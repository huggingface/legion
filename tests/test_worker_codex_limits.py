"""Codex `max_turns` / `max_budget_usd`: the mapper's counting on the recorded fixture, and the
adapter against a fake app-server whose turns make one model call per step."""

import json
import sys
from pathlib import Path

import pytest

from flock.controller.prices import Price
from flock.models import SwarmSpec, TypeSpec
from flock.worker.adapters.codex import CodexAdapter, CodexMapper

from .conftest import make_spec
from .test_api import apply

FIX = Path(__file__).parent / "fixtures"


def run_fixture(m: CodexMapper):
    evs, aborts = [], 0
    for rec in map(json.loads, (FIX / "codex_appserver.jsonl").read_text().splitlines()):
        msg = rec["msg"]
        if rec["dir"] != "in":
            continue
        if "method" in msg and "id" not in msg:
            evs += m.on_notification(msg["method"], msg.get("params") or {})
        elif "result" in msg and isinstance(msg["result"].get("turn"), dict):
            evs += m.on_turn_id(msg["result"]["turn"]["id"])
        if m.abort_requested:
            aborts, m.abort_requested = aborts + 1, False
    return evs, aborts


def test_mapper_counts_model_calls_on_the_fixture():
    """The fixture turn makes three model calls (one per token count): the steer cut the first,
    the second ran the shell, the third answered."""
    m = CodexMapper("gpt-6-luna")
    m.max_turns = 3
    evs, aborts = run_fixture(m)
    assert aborts == 0 and m.calls == 3
    assert [e.status for e in evs if e.kind == "turn_ended"] == ["completed"]
    m = CodexMapper("gpt-6-luna")
    m.max_turns = 1
    evs, aborts = run_fixture(m)
    assert aborts == 1  # the first call had no final answer, so Codex would call again
    [end] = [e for e in evs if e.kind == "turn_ended"]
    assert end.status == "error" and end.data["reason"] == "max_turns"
    assert end.data["subtype"] == "error_max_turns" and "max_turns exceeded (1)" in end.error
    # usage events are unchanged by the limits
    assert [e.usage for e in evs if e.kind == "usage"] == [
        e.usage for e in run_fixture(CodexMapper("gpt-6-luna"))[0] if e.kind == "usage"]


def test_mapper_budget_prices_each_call():
    m = CodexMapper("gpt-6-luna")
    m.max_budget_usd, m.prices = 0.02, {"gpt-6-luna": Price(1.0, 1.0, 1.0, 1.0)}  # $1/M tokens
    evs, aborts = run_fixture(m)
    # per-call `last`: 16805, 16846 and 16881 tokens -> $0.0168, $0.0337 (over, tool call: stop)
    assert aborts == 1 and m.calls == 3 and m.cost == pytest.approx(0.050532)
    [end] = [e for e in evs if e.kind == "turn_ended"]
    assert end.data["reason"] == "max_budget_usd" and end.data["num_turns"] == 3


FAKE = r'''
import json, sys, threading, time
LOG = sys.argv[-1]
active, stop, n, total = None, set(), 0, 0
lock = threading.Lock()
def send(m):
    with lock:
        sys.stdout.write(json.dumps(m) + "\n"); sys.stdout.flush()
def call(turn, items, final):
    """One model call: its items, then its token count (100 in, 10 out per call)."""
    global total
    open(LOG, "a").write(turn + "\n")
    for it in items:
        send({"method": "item/started", "params": {"turnId": turn, "item": it}})
        send({"method": "item/completed", "params": {"turnId": turn, "item": it}})
    total += 110
    last = {"inputTokens": 100, "cachedInputTokens": 0, "outputTokens": 10, "totalTokens": 110}
    tot = {"inputTokens": total // 11 * 10, "cachedInputTokens": 0, "outputTokens": total // 11, "totalTokens": total}
    send({"method": "thread/tokenUsage/updated", "params": {"turnId": turn, "tokenUsage": {"total": tot, "last": last}}})
def run(turn, text):
    global active
    k = 0
    tools = int(text.split(":")[0]) if text.split(":")[0].isdigit() else 0
    while True:
        time.sleep(0.15)
        if turn in stop:
            status = "interrupted"; break
        if k < tools:
            k += 1
            call(turn, [{"type": "commandExecution", "id": "c%d" % k, "command": "true", "exitCode": 0, "aggregatedOutput": ""}], False)
        else:
            call(turn, [{"type": "agentMessage", "id": "m", "text": "done", "phase": "final_answer"}], True)
            time.sleep(0.3)  # the turn lingers after its final answer (steer window)
            status = "completed"; break
    active = None
    send({"method": "turn/completed", "params": {"turn": {"id": turn, "status": status}}})
for line in sys.stdin:
    m = json.loads(line)
    meth, rid, p = m.get("method"), m.get("id"), m.get("params") or {}
    if meth == "initialize":
        send({"id": rid, "result": {}})
    elif meth == "thread/start":
        send({"id": rid, "result": {"model": p["model"], "thread": {"id": "thr-1", "path": None}}})
    elif meth == "turn/start":
        n += 1; active = "turn-%d" % n
        send({"id": rid, "result": {"turn": {"id": active, "status": "inProgress"}}})
        send({"method": "turn/started", "params": {"turn": {"id": active}}})
        text = p["input"][0]["text"]
        send({"method": "item/completed", "params": {"item": {"type": "userMessage", "id": "u", "content": p["input"]}}})
        threading.Thread(target=run, args=(active, text)).start()
    elif meth == "turn/steer":
        if active and p["expectedTurnId"] == active:
            send({"id": rid, "result": {"turnId": active}})
        else:
            send({"id": rid, "error": {"code": -32600, "message": "no active turn"}})
    elif meth == "turn/interrupt":
        stop.add(p["turnId"])
        send({"id": rid, "result": {}})
'''


@pytest.fixture
def codex(tmp_path):
    script = tmp_path / "fake-codex-impl.py"
    script.write_text(FAKE)
    calls = tmp_path / "calls.log"
    wrapper = tmp_path / "fake-codex"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\" {calls}\n")
    wrapper.chmod(0o755)
    (tmp_path / "wd").mkdir()
    a = CodexAdapter(tmp_path / "wd", home=tmp_path / "codex", codex_bin=str(wrapper))

    def model_calls() -> list[str]:
        return calls.read_text().split() if calls.exists() else []

    a.model_calls = model_calls
    yield a


async def until_ended(a):
    got = []
    async for ev in a.events():
        got.append(ev)
        if ev.kind == "turn_ended":
            return got
    raise AssertionError("exited before turn_ended")


def spec(**kw):
    return TypeSpec(harness="codex", model="gpt-6-luna", **kw)


async def test_codex_max_turns_interrupts_after_one_call(codex):
    await codex.start(codex.workdir, None, spec(max_turns=1))
    await codex.send("2: run two tools")
    evs = await until_ended(codex)
    end = evs[-1]
    assert end.status == "error" and end.data["reason"] == "max_turns"
    assert end.data["subtype"] == "error_max_turns" and end.data["num_turns"] == 1
    assert codex.model_calls() == ["turn-1"]
    # the session is still usable: a plain prompt is one call, under the limit
    await codex.send("hello")
    evs = await until_ended(codex)
    assert evs[-1].status == "completed" and "reason" not in evs[-1].data
    assert codex.model_calls() == ["turn-1", "turn-2"]
    await codex.stop()


async def test_codex_max_budget_interrupts(codex):
    """gpt-6-luna is not in the price table, so it is priced at the default row ($15/M input):
    one 110-token call is over a $0.000001 budget."""
    await codex.start(codex.workdir, None, spec(max_budget_usd=0.000001))
    await codex.send("3: tools")
    evs = await until_ended(codex)
    assert evs[-1].status == "error" and evs[-1].data["reason"] == "max_budget_usd"
    assert codex.model_calls() == ["turn-1"]
    await codex.stop()


async def test_codex_without_limits_is_unchanged(codex):
    await codex.start(codex.workdir, None, spec())
    await codex.send("2: tools")
    evs = await until_ended(codex)
    assert evs[-1].status == "completed" and evs[-1].data == {"harness_turn_id": "turn-1"}
    assert len(codex.model_calls()) == 3
    await codex.stop()


async def test_codex_limit_not_hit(codex):
    await codex.start(codex.workdir, None, spec(max_turns=3, max_budget_usd=1.0))
    await codex.send("2: tools")
    evs = await until_ended(codex)
    assert evs[-1].status == "completed" and len(codex.model_calls()) == 3
    await codex.stop()


async def test_codex_steer_after_last_allowed_call_becomes_the_next_turn(codex):
    """max_turns=1 and the final answer is in, but the turn is still open: a steer now would make
    a second call. It is held, the turn completes normally, and the steer runs as its own turn."""
    await codex.start(codex.workdir, None, spec(max_turns=1))
    await codex.send("hello")
    async for ev in codex.events():
        if ev.kind == "usage":
            break
    assert codex._mapper.exhausted()
    await codex.send("and another thing")
    first = await until_ended(codex)
    assert first[-1].status == "completed"
    second = await until_ended(codex)
    assert any(e.kind == "turn_started" and e.data["harness_turn_id"] == "turn-2" for e in second)
    assert second[-1].status == "completed"
    assert codex.model_calls() == ["turn-1", "turn-2"]
    await codex.stop()


async def test_codex_budget_uses_the_pinned_provider_price(codex):
    """A provider-backed type is priced at its pinned price, not the default row: at $0/M the
    same budget that stops a default-priced call is never reached."""
    spec_ = SwarmSpec.model_validate({
        "swarm": "s", "namespace": "n", "bucket": "b",
        "providers": {"hf": {"type": "openai_compat", "base_url": "https://router.example/v1",
                             "api_key_env": "HF_TOKEN", "pricing": {"*": [0, 0, 0]}}},
        "types": {"w": {"harness": "codex", "model": "hf:gpt-6-luna", "max_budget_usd": 0.000001}},
    }).types["w"]
    await codex.start(codex.workdir, None, spec_)
    assert codex._mapper.prices["gpt-6-luna"].input == 0
    await codex.send("2: tools")
    evs = await until_ended(codex)
    assert evs[-1].status == "completed" and len(codex.model_calls()) == 3
    await codex.stop()


async def test_apply_accepts_codex_limits(client):
    s = make_spec(reviewer={"harness": "codex", "model": "gpt-5.5-codex", "secrets": ["OPENAI_API_KEY"],
                            "max_turns": 3, "max_budget_usd": 0.05})
    await apply(client, s)
    got = (await client.get("/swarms/demo")).json()["spec"]["types"]["reviewer"]
    assert got["max_turns"] == 3 and got["max_budget_usd"] == 0.05
