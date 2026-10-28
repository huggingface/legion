"""Live changes (docs/live.md): the `flock update` diff, the `flock grant` argument syntax, the
folder watch, the grants API (checks, merge, events, the budget gate, what heartbeats carry),
the runtime side (prompts and grants at the next turn boundary, with a note), the CLI commands
against the in-process app, and `flock run --watch` in process."""

from __future__ import annotations

import asyncio
import io
import json
import textwrap
from pathlib import Path

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from flock import cli
from flock.comms.tools import tool_def
from flock.controller import localrun
from flock.live import FolderWatch, diff_specs, grant_patch, revoke_patch
from flock.models import (
    AgentHeartbeat,
    CommsSpec,
    Grant,
    GrantLimits,
    Harness,
    SwarmSpec,
    granted_spec,
    merge_grants,
    type_spec_rev,
)
from flock.tools.bundle import pack
from flock.tools.toolset import Toolset
from flock.worker.comms import CommsHub

from .test_cli import _wire
from .worker_fakes import runtime, spec, until

runner = CliRunner()

TOOLS = '''
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b
'''
TOOLS_V2 = TOOLS + '''

def mul(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
'''
PROVIDERS = {"gpu": {"type": "openai_compat", "base_url": "http://gpu.invalid/v1",
                     "api_key_env": "K", "self_hosted": True}}


def raw(**types) -> dict:
    base = {"researcher": {"harness": "claude", "model": "claude-opus-5-5", "replicas": 2,
                           "agents_per_worker": 2},
            "calc": {"harness": "llm", "model": "gpu:m", "replicas": 2, "agents_per_worker": 2,
                     "tools": ["add"]}}
    return {"swarm": "demo", "namespace": "ns", "bucket": "hf://buckets/ns/sh",
            "providers": PROVIDERS, "types": {**base, **types}}


# --------------------------------------------------------------------------- the diff


def test_diff_lists_tools_prompts_and_keys():
    comms = {"url": "http://board.invalid", "invite_env": "INV", "channel": "team",
             "kickoff": "hello"}
    old = SwarmSpec.model_validate({
        **raw(), "tools_version": "0" * 16, "limits": {"budget_usd": 10},
        "comms": {**comms, "channel_id": 7, "kickoff_id": 70}})
    old.types["calc"].system_prompt = "v1"
    old.types["x"] = old.types["researcher"].model_copy()
    new = SwarmSpec.model_validate({
        **raw(), "tools_version": "1" * 16, "limits": {"budget_usd": 15},
        "comms": {**comms, "kickoff": "hello again"}})
    new.types["calc"].system_prompt = "v2\nwith a second line"
    new.types["calc"].replicas = 3
    new.types["r"] = new.types["researcher"].model_copy(update={"replicas": 1})
    lines = diff_specs(old, new, (["add"], ["mul"]))
    assert lines == [
        f"tools: {'0' * 16} -> {'1' * 16} (+mul -add)",
        ("prompts changed: calc (system_prompt); "
         "comms (kickoff (posted once per channel: not posted again))"),
        "types.calc.replicas: 2 -> 3",
        "+ type r (claude, replicas 1)",
        "- type x (its agents are removed)",
        "limits.budget_usd: 10.0 -> 15.0",
    ]
    assert diff_specs(old, old) == []
    # what the controller writes (channel id, kickoff id, resolved models) is never a change
    same = SwarmSpec.model_validate({**raw(), "tools_version": "0" * 16,
                                     "limits": {"budget_usd": 10}, "comms": comms})
    same.types["calc"].system_prompt = "v1"
    same.types["x"] = same.types["researcher"].model_copy()
    assert diff_specs(old, same) == []


# --------------------------------------------------------------------------- grant arguments


def test_grant_and_revoke_arguments():
    g = grant_patch(Grant(), ["tools=+mul,-add", "comms.post=off", "+search", "bash=off",
                              "limits.max_turns=5", "limits.budget_usd=$0.5"])
    assert g.tools == {"mul": True, "add": False, "comms:post": False, "comms:search": True}
    assert (g.bash, g.limits.max_turns, g.limits.budget_usd) == (False, 5, 0.5)
    assert g.describe() == ("tools -add -comms:post +comms:search +mul, bash off, "
                            "limits.max_turns 5, limits.budget_usd 0.5")
    assert grant_patch(g, ["comms.post"]).tools["comms:post"] is True  # bare switch: on
    for bad in (["nope"], ["bash=maybe"], ["limits.max_turns=x"], ["tools="]):
        with pytest.raises(ValueError):
            grant_patch(Grant(), bad)
    r = revoke_patch(g, ["tools=mul", "comms.post", "limits.max_turns", "bash"])
    assert r.tools == {"add": False, "comms:search": True}
    assert (r.bash, r.limits.max_turns, r.limits.budget_usd) == (None, None, 0.5)
    assert revoke_patch(g, ["tools", "limits", "bash"]).empty()


def test_merge_and_granted_spec():
    t = Grant(tools={"add": False, "mul": True}, bash=False, limits=GrantLimits(max_turns=9))
    a = Grant(tools={"add": True}, limits=GrantLimits(budget_usd=1.0))
    m = merge_grants(t, a)
    assert m.tools == {"add": True, "mul": True} and m.bash is False
    assert (m.limits.max_turns, m.limits.budget_usd) == (9, 1.0)
    assert merge_grants(None, Grant()) is None and merge_grants(t, None) is t
    ts = SwarmSpec.model_validate(raw()).types["calc"]
    eff = granted_spec(ts.model_copy(update={"max_budget_usd": 0.5}), m, spent_usd=0.75)
    assert (eff.max_turns, eff.bash, eff.max_budget_usd) == (9, False, 0.25)
    assert granted_spec(ts, None) is ts


# --------------------------------------------------------------------------- the folder watch


def test_folder_watch_settles_and_follows_the_spec(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "system.md").write_text("v1")
    folder = tmp_path / "tools"
    folder.mkdir()
    (folder / "calc.py").write_text(TOOLS)
    f = tmp_path / "spec.yaml"
    r = raw()
    r["types"]["calc"]["system_prompt_file"] = "prompts/system.md"
    f.write_text(yaml.safe_dump(r))
    w = FolderWatch(f, cli.read_spec(str(f)), quiet=1.0)
    assert not w.poll(0.0)
    (tmp_path / "prompts" / "system.md").write_text("v2, longer")
    assert not w.poll(1.0)  # changed: wait until it is quiet
    assert not w.poll(1.5)
    assert w.poll(2.1) and not w.poll(3.0)  # once
    (folder / "_helpers.py").write_text("X = 1\n")  # a new file in the tools folder counts
    assert not w.poll(4.0) and w.poll(5.1)
    (tmp_path / "unrelated.txt").write_text("x")  # not referenced: ignored
    assert not w.poll(6.0) and not w.poll(7.5)
    # a spec that references a new file: retarget, then that file is watched
    (tmp_path / "prompts" / "round.md").write_text("round {round}")
    r["types"]["calc"]["loop"] = {"prompt_file": "prompts/round.md"}
    f.write_text(yaml.safe_dump(r))
    assert not w.poll(8.0) and w.poll(9.1)
    w.retarget(cli.read_spec(str(f)))
    assert not w.poll(10.0)
    (tmp_path / "prompts" / "round.md").write_text("round {round}, again")
    assert not w.poll(11.0) and w.poll(12.1)


# --------------------------------------------------------------------------- grants API


async def applied(client, tmp_path, tick, top: dict | None = None, **types) -> SwarmSpec:
    b = pack(_tools(tmp_path, TOOLS_V2))
    r = await client.post("/swarms/demo/tools", content=b.data,
                          headers={"content-type": "application/gzip"})
    assert r.status_code == 201, r.text
    spec = SwarmSpec.model_validate({**raw(**types), "tools_version": b.version, **(top or {})})
    r = await client.put("/swarms/demo", json=spec.model_dump(mode="json"))
    assert r.status_code == 200, r.text
    tick()
    return spec


def _tools(root: Path, src: str) -> Path:
    d = root / "tools"
    d.mkdir(parents=True, exist_ok=True)
    (d / "calc.py").write_text(textwrap.dedent(src))
    return d


async def test_grants_are_checked_merged_evented_and_survive_apply(client, store, tick, tmp_path):
    spec = await applied(client, tmp_path, tick)
    r = await client.put("/swarms/demo/grants/calc", json={"tools": {"add": False}, "bash": False})
    assert r.status_code == 200, r.text
    assert r.json() == {"swarm": "demo", "target": "calc", "kind": "type",
                        "grant": {"tools": {"add": False}, "bash": False}}
    r = await client.put("/swarms/demo/grants/calc-0000",
                         json={"tools": {"add": True, "mul": True}, "limits": {"max_turns": 3}})
    assert r.status_code == 200 and r.json()["kind"] == "agent"
    a0 = (await client.get("/agents/calc-0000")).json()
    a1 = (await client.get("/agents/calc-0001")).json()
    assert a0["grants"] == {"tools": {"add": True, "mul": True}, "limits": {"max_turns": 3},
                            "bash": False}
    assert a1["grants"] == {"tools": {"add": False}, "bash": False}
    assert (await client.get("/agents/researcher-0000")).json()["grants"] is None
    listed = (await client.get("/swarms/demo/agents", params={"type": "calc"})).json()["agents"]
    assert [x["grants"] for x in listed] == [a0["grants"], a1["grants"]]

    for target, body, want in [
        ("calc", {"tools": {"nope": True}}, "unknown tool 'nope'; bundle"),
        ("researcher", {"bash": False}, "only the llm harness has a bash tool"),
        ("calc", {"tools": {"post": True}}, "+post: the swarm has no comms block"),
        ("calc", {"limits": {"max_turns": 0}}, "max_turns must be >= 1"),
        ("calc", {"mystery": 1}, "Extra inputs are not permitted"),
    ]:
        r = await client.put(f"/swarms/demo/grants/{target}", json=body)
        assert r.status_code in (400, 422) and want in r.text, (target, r.text)
    assert (await client.put("/swarms/demo/grants/ghost", json={})).status_code == 404

    rows = (await client.get("/swarms/demo/grants")).json()["grants"]
    assert [(g["kind"], g["target"]) for g in rows] == [("type", "calc"), ("agent", "calc-0000")]
    # grants are an overlay: a re-apply (what `flock update` does) keeps them
    spec.types["calc"].replicas = 3
    assert (await client.put("/swarms/demo", json=spec.model_dump(mode="json"))).status_code == 200
    assert len((await client.get("/swarms/demo/grants")).json()["grants"]) == 2

    assert (await client.delete("/swarms/demo/grants/calc")).json()["revoked"] is True
    assert (await client.delete("/swarms/demo/grants/calc")).status_code == 404
    r = await client.put("/swarms/demo/grants/calc-0000", json={})  # empty: revoked too
    assert r.status_code == 200 and r.json()["grant"] == {}
    assert (await client.get("/swarms/demo/grants")).json()["grants"] == []
    evs = (await client.get("/swarms/demo/events")).json()["events"]
    kinds = [(e["type"], e["data"]["target"]) for e in evs if e["type"].startswith("grant")]
    assert kinds == [("grant", "calc"), ("grant", "calc-0000"), ("grant_revoked", "calc"),
                     ("grant_revoked", "calc-0000")]
    mine = (await client.get("/agents/calc-0000/events")).json()["events"]
    assert [e["type"] for e in mine] == ["grant", "grant_revoked"]


async def test_comms_tools_can_be_granted_only_to_a_type_on_the_board(client, store, tick,
                                                                      tmp_path, monkeypatch):
    import flock.controller.api as api_mod

    monkeypatch.setattr(api_mod, "comms_resolve", lambda spec, prev, env, transport=None: None)
    comms = {"url": "http://board.invalid", "invite_env": "INV", "channel": "team",
             "channel_id": 7, "search": None}
    await applied(client, tmp_path, tick, {"comms": comms},
                  calc={"harness": "llm", "model": "gpu:m", "replicas": 1,
                        "tools": ["add", "comms:read"]})
    r = await client.put("/swarms/demo/grants/calc", json={"tools": {"post": True}})
    assert r.status_code == 200 and r.json()["grant"] == {"tools": {"comms:post": True}}
    r = await client.put("/swarms/demo/grants/researcher", json={"tools": {"comms:post": True}})
    assert r.status_code == 400 and "has no comms tools" in r.text
    r = await client.put("/swarms/demo/grants/calc", json={"tools": {"search": True}})
    assert r.status_code == 400 and "comms.search is off" in r.text
    r = await client.put("/swarms/demo/grants/researcher", json={"tools": {"comms:read": False}})
    assert r.status_code == 200  # taking away is always possible


async def test_heartbeat_carries_a_changed_type_spec_and_the_grants(client, store, tick, tmp_path):
    spec = await applied(client, tmp_path, tick)
    w = next(w for w in store.list_workers("demo") if w["type"] == "calc")
    store.update_worker(w["id"], job_id="job-1", incarnation="inc-1")
    rev = type_spec_rev(spec.types["calc"])

    async def beat(rev: str | None, agents=("calc-0000", "calc-0001")) -> dict:
        hb = {"worker": w["id"], "deadline": "2026-10-15T09:00:00Z", "spec_rev": rev,
              "agents": [AgentHeartbeat(name=a, state="idle").model_dump(mode="json")
                         for a in agents]}
        r = await client.post(f"/workers/{w['id']}/heartbeat", json=hb,
                              params={"incarnation": "inc-1"})
        assert r.status_code == 200, r.text
        return r.json()

    out = await beat(rev)
    assert "type_spec" not in out and "grants" not in out
    spec.types["calc"].system_prompt = "You are calc v2."
    assert (await client.put("/swarms/demo", json=spec.model_dump(mode="json"))).status_code == 200
    out = await beat(rev)
    assert out["type_spec"]["system_prompt"] == "You are calc v2."
    assert out["spec_rev"] == type_spec_rev(spec.types["calc"]) != rev
    assert "type_spec" not in await beat(out["spec_rev"])
    await client.put("/swarms/demo/grants/calc", json={"bash": False})
    await client.put("/swarms/demo/grants/calc-0001", json={"limits": {"max_turns": 2}})
    out = await beat(out["spec_rev"])
    assert out["grants"] == {"calc-0000": {"bash": False},
                             "calc-0001": {"bash": False, "limits": {"max_turns": 2}}}


async def test_a_granted_budget_gates_prompts(client, store, tick, tmp_path):
    await applied(client, tmp_path, tick)
    store.update_agent("calc-0000", cost=0.6)
    r = await client.put("/swarms/demo/grants/calc-0000", json={"limits": {"budget_usd": 0.5}})
    assert r.status_code == 200
    r = await client.post("/agents/calc-0000/prompt", json={"text": "hi"})
    assert r.status_code == 409 and "reached its granted limits.budget_usd" in r.text
    assert (await client.post("/agents/calc-0001/prompt", json={"text": "hi"})).status_code == 200
    r = await client.post("/swarms/demo/prompts", json={"type": "calc", "text": "all"})
    assert r.json()["skipped_budget"] == 1 and r.json()["matched"] == 2
    await client.put("/swarms/demo/grants/calc-0000", json={"limits": {"budget_usd": 1}})
    assert (await client.post("/agents/calc-0000/prompt", json={"text": "hi"})).status_code == 200


async def test_apply_refuses_a_harness_change_for_a_running_type(client, store, tick, tmp_path):
    spec = await applied(client, tmp_path, tick)
    spec.types["researcher"].harness = Harness.codex
    r = await client.put("/swarms/demo", json=spec.model_dump(mode="json"))
    assert r.status_code == 400 and "types.researcher.harness: a running type's harness" in r.text
    spec.types["fresh"] = spec.types["researcher"].model_copy(update={"replicas": 0})
    spec.types["researcher"].harness = Harness.claude
    assert (await client.put("/swarms/demo", json=spec.model_dump(mode="json"))).status_code == 200
    spec.types["fresh"].harness = Harness.codex  # a type without agents may still change
    assert (await client.put("/swarms/demo", json=spec.model_dump(mode="json"))).status_code == 200


async def test_an_effective_apply_bumps_the_generation_and_keeps_the_spec(client, store, tick,
                                                                         tmp_path):
    spec = await applied(client, tmp_path, tick)
    src = {"git_sha": "abc123", "ref": "main", "dirty": 1, "by": "carlos", "extra": "dropped"}
    hdr = {"x-flock-source": json.dumps(src)}
    r = await client.put("/swarms/demo", json=spec.model_dump(mode="json"), headers=hdr)
    assert r.json()["generation"] == 1  # nothing changed: same generation, no history entry
    n = spec.types["researcher"].replicas
    spec.types["researcher"].replicas = n + 1
    r = await client.put("/swarms/demo", json=spec.model_dump(mode="json"), headers=hdr)
    assert r.json()["generation"] == 2
    hist = (await client.get("/swarms/demo/history")).json()
    assert [(h["generation"], h["diff_summary"]) for h in hist] == [
        (1, ["created"]), (2, [f"types.researcher.replicas: {n} -> {n + 1}"])]
    assert hist[0]["source"] == {}
    assert hist[1]["source"] == {"git_sha": "abc123", "ref": "main", "dirty": True, "by": "carlos"}
    got = (await client.get("/swarms/demo/history/1")).json()
    assert got["spec"]["types"]["researcher"]["replicas"] == n and got["ts"]
    assert (await client.get("/swarms/demo/history/9")).status_code == 404
    evs = [e["data"] for e in store.read_events("_swarm.demo", 0) if e["type"] == "spec_applied"]
    assert [e["generation"] for e in evs] == [1, 2] and evs[1]["source"]["by"] == "carlos"
    assert evs[1]["diff_summary"] == hist[1]["diff_summary"]


def test_bash_switch_is_llm_only_in_the_spec():
    with pytest.raises(ValueError, match="only the llm harness"):
        spec(bash=False)
    with pytest.raises(ValueError, match="contradicts tool_choice"):
        SwarmSpec.model_validate(raw(calc={"harness": "llm", "model": "gpu:m", "bash": False,
                                           "tool_choice": "bash"}))


# --------------------------------------------------------------------------- the runtime side


async def test_a_new_system_prompt_applies_at_the_next_turn_with_a_note(tmp_path):
    rt, adapter = runtime(tmp_path, system_prompt="v1")
    await rt.setup(restore=False)
    rt.set_live(spec=spec(system_prompt="v0"))  # before any turn: applied, no note
    await rt.prompt("one")
    await until(lambda: rt.state.turns == 1)
    assert adapter.sent == ["one"] and (rt.workdir / "CLAUDE.md").read_text() == "v0"
    adapter.turn_time = 0.3
    await rt.prompt("two")
    await until(lambda: rt.turn_id is not None)
    rt.set_live(spec=spec(system_prompt="v2"))
    await rt.prompt("steer")  # joins the turn in flight: no boundary, nothing applied
    assert (rt.workdir / "CLAUDE.md").read_text() == "v0" and adapter.sent[-1] == "steer"
    await until(lambda: rt.state.turns == 2)
    adapter.turn_time = 0.05
    await rt.prompt("three")
    await until(lambda: adapter.sent[-1].endswith("three"))
    assert (rt.workdir / "CLAUDE.md").read_text() == "v2"
    assert adapter.sent[-1] == ("[flock] Since your last turn: your instructions in CLAUDE.md "
                                "changed; re-read them.\n\nthree")
    await until(lambda: rt.state.turns == 3)
    await rt.prompt("four")
    await until(lambda: adapter.sent[-1].endswith("four"))
    assert adapter.sent[-1] == "four"  # once
    await rt.stop()


async def test_grants_apply_at_the_boundary_and_a_budget_defers(tmp_path):
    rt, adapter = runtime(tmp_path, max_turns=20)
    await rt.setup(restore=False)
    await rt.prompt("one")
    await until(lambda: rt.state.turns == 1)
    rt.set_live(grant=Grant(limits=GrantLimits(max_turns=3)))
    assert rt.spec.max_turns == 20  # not before the boundary
    await rt.prompt("two")
    await until(lambda: adapter.sent[-1].endswith("two"))
    assert rt.spec.max_turns == 3 and rt.base_spec.max_turns == 20
    assert adapter.sent[-1].startswith("[flock] Since your last turn: your grants changed: "
                                       "limits.max_turns 3.")
    await until(lambda: rt.state.turns == 2)
    rt.set_live(grant=None)
    await rt.prompt("three")
    await until(lambda: adapter.sent[-1].endswith("three"))
    assert rt.spec.max_turns == 20 and "all revoked" in adapter.sent[-1]
    await until(lambda: rt.state.turns == 3)
    rt.set_live(grant=Grant(limits=GrantLimits(budget_usd=0)))  # spent >= 0: no new turn
    await rt.prompt("four", prompt_id="p4")
    deferred = [e.data.get("data", e.data) for e in rt.outbox._q if e.type == "prompt_deferred"]
    assert deferred and deferred[-1]["reason"] == "budget"
    assert deferred[-1]["prompt_id"] == "p4" and adapter.sent[-1].endswith("three")
    await rt.stop()


async def test_cli_harness_gets_a_tools_note_when_its_tools_change(tmp_path):
    tools = [tool_def("comms:read")]

    def box() -> Toolset:
        return Toolset(version=None, dir=None, swarm="sw", type="researcher",
                       data_dir=tmp_path / "data", tools=tuple(tools))

    rt, adapter = runtime(tmp_path)
    rt.toolbox = box
    await rt.setup(restore=False)
    await rt.prompt("one")
    await until(lambda: rt.state.turns == 1)
    assert "`read`" in (rt.workdir / "CLAUDE.md").read_text() and adapter.sent == ["one"]
    tools.append(tool_def("comms:post"))
    await rt.prompt("two")
    await until(lambda: adapter.sent[-1].endswith("two"))
    assert "`post`" in (rt.workdir / "CLAUDE.md").read_text()
    assert adapter.sent[-1] == ("[flock] Since your last turn: your tools changed; re-read the "
                                "Tools section of CLAUDE.md.\n\ntwo")
    await rt.stop()


async def test_a_changed_board_section_gets_a_note(tmp_path):
    class Hub:  # the parts of CommsHub a runtime calls
        text = "<!-- flock:comms -->\nrules v1\n<!-- /flock:comms -->"

        async def on_prompt(self, agent, harness):
            pass

        def section(self, agent):
            return self.text

        def set_grant(self, agent, tools):
            pass

        def deny_env(self):
            return []

    rt, adapter = runtime(tmp_path, system_prompt="base")
    rt.comms = hub = Hub()
    rt.toolbox = lambda: Toolset(version=None, dir=None, swarm="sw", type="researcher",
                                 data_dir=tmp_path / "data")  # as the worker always sets one
    await rt.setup(restore=False)
    await rt.prompt("one")
    await until(lambda: rt.state.turns == 1)
    hub.text = hub.text.replace("v1", "v2")  # a re-apply changed the rules or protocols
    await rt.prompt("two")
    await until(lambda: adapter.sent[-1].endswith("two"))
    assert "rules v2" in (rt.workdir / "CLAUDE.md").read_text()
    assert adapter.sent[-1] == ("[flock] Since your last turn: the Team board section of "
                                "CLAUDE.md changed; re-read it.\n\ntwo")
    await rt.stop()


def test_comms_hub_applies_an_agents_grant():
    hub = CommsHub(swarm="sw", env={})
    hub.configure(CommsSpec(url="http://b.invalid", invite_env="INV", channel="team",
                            channel_id=7), ["comms:read", "comms:post"])
    hub.set_grant("a-0001", {"comms:post": False, "mul": True})
    hub.set_grant("a-0002", {"comms:search": True})
    assert hub.entries_for("a-0000") == ["comms:read", "comms:post"]
    assert hub.entries_for("a-0001") == ["comms:read"]
    assert hub.entries_for("a-0002") == ["comms:read", "comms:post", "comms:search"]
    assert [d.name for d in hub.tooldefs("a-0001")] == ["read"]
    assert "`post`" not in hub.section("a-0001") and "`post`" in hub.section("a-0000")
    res = asyncio.run(hub.call("a-0001", "comms:post", {"text": "x", "thread": "new"}))
    assert res.error and "not enabled" in res.output
    hub.set_grant("a-0001", None)
    assert hub.entries_for("a-0001") == ["comms:read", "comms:post"]


# --------------------------------------------------------------------------- the CLI


def _experiment(root: Path, *, replicas: int = 2, prompt: str = "You are a researcher.",
                tools: str = TOOLS, **type_extra) -> Path:
    (root / "prompts").mkdir(parents=True, exist_ok=True)
    (root / "prompts" / "system.md").write_text(prompt)
    _tools(root, tools)
    t = {"harness": "claude", "model": "claude-opus-5-5", "replicas": replicas,
         "agents_per_worker": 2, "system_prompt_file": "prompts/system.md", "tools": ["add"],
         **type_extra}
    f = root / "spec.yaml"
    f.write_text(yaml.safe_dump({"swarm": "demo", "namespace": "ns",
                                 "bucket": "hf://buckets/ns/sh", "types": {"researcher": t}}))
    return f


def test_cli_update_prints_what_changed_and_refuses_cleanly(tmp_path, monkeypatch, app, store,
                                                           tick):
    _wire(monkeypatch, app)
    exp = tmp_path / "exp"
    f = _experiment(exp)
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 1 and "no swarm 'demo' is running" in r.output
    assert runner.invoke(cli.app, ["apply", str(f)]).exit_code == 0
    tick()
    v1 = store.get_swarm("demo")["spec"].tools_version
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 0 and r.output.strip() == "no changes", r.output

    _experiment(exp, replicas=3, prompt="You are a careful researcher.", tools=TOOLS_V2)
    v2 = pack(exp / "tools").version
    r = runner.invoke(cli.app, ["update", str(f)])  # the spec file works as well as the folder
    assert r.exit_code == 0, r.output
    assert r.output.splitlines() == [f"tools: {v1} -> {v2} (+mul)",
                                     "prompts changed: researcher (system_prompt)",
                                     "types.researcher.replicas: 2 -> 3"]
    applied = store.get_swarm("demo")["spec"]
    assert applied.tools_version == v2 and applied.types["researcher"].replicas == 3
    assert applied.types["researcher"].system_prompt == "You are a careful researcher."

    _experiment(exp, replicas=3, prompt="You are a careful researcher.", tools=TOOLS_V2,
                harness="codex")
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 1 and "refused: types.researcher.harness" in r.output, r.output
    (exp / "spec.yaml").write_text("types: [unclosed\n")
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 1 and "spec.yaml" in r.output
    _experiment(exp, tools="def broken(:\n")
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 1 and "tools:" in r.output
    assert store.get_swarm("demo")["spec"] == applied  # nothing was applied


def test_cli_grant_revoke_and_list(tmp_path, monkeypatch, app, store, tick):
    _wire(monkeypatch, app)
    f = _experiment(tmp_path / "exp", tools=TOOLS_V2)
    assert runner.invoke(cli.app, ["apply", str(f)]).exit_code == 0
    tick()
    r = runner.invoke(cli.app, ["grant", "researcher", "tools=-add"])
    assert r.exit_code == 0 and r.output.strip() == (
        "type researcher: tools -add (from each agent's next turn)"), r.output
    r = runner.invoke(cli.app, ["grant", "researcher-0000", "+mul", "limits.max_turns=4"])
    assert r.exit_code == 0 and "agent researcher-0000: tools +mul, limits.max_turns 4" in r.output
    r = runner.invoke(cli.app, ["grant", "researcher-0000", "limits.budget_usd=2"])
    assert "tools +mul, limits.max_turns 4, limits.budget_usd 2" in r.output  # merged
    r = runner.invoke(cli.app, ["grants"])
    assert r.exit_code == 0
    assert r.output.splitlines()[1:] == [
        "demo   type   researcher       tools -add",
        "demo   agent  researcher-0000  tools +mul, limits.max_turns 4, limits.budget_usd 2"]
    st = json.loads(runner.invoke(cli.app, ["agent", "researcher-0000"]).output)
    assert st["grants"] == {"tools": {"add": False, "mul": True},
                            "limits": {"max_turns": 4, "budget_usd": 2.0}}
    r = runner.invoke(cli.app, ["revoke", "researcher-0000", "limits"])
    assert r.exit_code == 0 and "agent researcher-0000: tools +mul (" in r.output
    r = runner.invoke(cli.app, ["revoke", "researcher"])
    assert r.exit_code == 0 and "all grants revoked" in r.output
    r = runner.invoke(cli.app, ["grant", "researcher", "bash=off"])
    assert r.exit_code == 1 and "only the llm harness" in r.output
    r = runner.invoke(cli.app, ["grant", "researcher", "bogus"])
    assert r.exit_code == 2 and "unknown capability" in r.output
    r = runner.invoke(cli.app, ["grant", "nobody", "+mul"])
    assert r.exit_code == 1 and "pass --swarm" in r.output
    runner.invoke(cli.app, ["revoke", "researcher-0000"])
    assert runner.invoke(cli.app, ["grants", "demo"]).output.strip() == "no grants"


# --------------------------------------------------------------------------- flock run --watch


async def test_run_watch_applies_saved_changes_and_survives_a_broken_spec(tmp_path):
    exp = tmp_path / "exp"
    exp.mkdir()
    f = exp / "spec.yaml"

    def write(budget: float = 10, **extra) -> None:
        f.write_text(yaml.safe_dump({
            "swarm": "watched", "compute": "local", "limits": {"budget_usd": budget},
            "types": {"w": {"harness": "claude", "model": "m", "replicas": 0, **extra}}}))

    write(labels={"v": "1"})
    out = io.StringIO()
    task = asyncio.create_task(localrun.run_local(localrun.RunOptions(
        spec=cli.read_spec(str(f)), data_dir=tmp_path / ".flock", until=60, refresh=0.05,
        grace=0, out=out, tty=False, watch=f, watch_interval=0.05, watch_quiet=0.15)))
    current = tmp_path / ".flock" / "current"
    while not current.exists():
        assert not task.done(), out.getvalue()
        await asyncio.sleep(0.02)
    info = json.loads(current.read_text())
    async with httpx.AsyncClient(base_url=info["url"],
                                 headers={"X-Flock-Token": info["token"]}) as c:
        async def labels() -> dict:
            r = await c.get("/swarms/watched")
            return r.json()["spec"]["types"]["w"]["labels"] if r.status_code == 200 else {}

        while await labels() != {"v": "1"}:
            await asyncio.sleep(0.02)
        write(labels={"v": "2"})
        await until(lambda: "types.w.labels.v: 1 -> 2" in out.getvalue(), timeout=10)
        assert await labels() == {"v": "2"}
        f.write_text("types: [unclosed\n")
        await until(lambda: "not applied: the swarm keeps running" in out.getvalue(), timeout=10)
        assert await labels() == {"v": "2"} and not task.done()
        write(labels={"v": "3"}, bash=False)  # refused by the spec: bash is llm only
        await until(lambda: out.getvalue().count("not applied") == 2, timeout=10)
        assert await labels() == {"v": "2"}
    write(budget=0, labels={"v": "3"})  # a live budget of 0 ends the run
    try:
        result = await asyncio.wait_for(task, 20)
    except TimeoutError:
        raise AssertionError(out.getvalue()) from None
    assert result.reason == "budget", out.getvalue()
    doc = json.loads((result.run_dir / "run.json").read_text())
    assert doc["spec"]["limits"]["budget_usd"] == 0
    assert doc["spec"]["types"]["w"]["labels"] == {"v": "3"}
