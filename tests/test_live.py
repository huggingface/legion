"""Live changes (docs/live.md): the `flock update` diff, what heartbeats carry, the
runtime side (prompts at the next turn boundary, with a note), the CLI commands against the
in-process app."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import yaml
from typer.testing import CliRunner

from flock import cli
from flock.live import (
    diff_specs,
    keep_replicas,
    world_tool_changes,
)
from flock.models import (
    AgentHeartbeat,
    SwarmSpec,
    type_spec_rev,
)
from flock.tools.bundle import pack

from .conftest import HB, make_spec, wire

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
GPU = {"urls": ["http://gpu.invalid/v1"], "api_key_env": "K"}


def raw(**types) -> dict:
    base = {"researcher": {**HB, "model": "m", "replicas": 2,
                           "agents_per_worker": 2},
            "calc": {**HB, "model": "m", "replicas": 2, "agents_per_worker": 2,
                     "tools": ["add"]}}
    return {"swarm": "demo", "namespace": "ns", "bucket": "hf://buckets/ns/sh",
            "inference": GPU, "types": {**base, **types}}


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
        "+ type r (harbor, replicas 1)",
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


def test_diff_lists_element_wise_and_skips_world_keys():
    world = {"url": "http://w.invalid", "endpoint": "http://w.invalid", "credential_id": "a",
             "manifest": {"world_api": "v1", "name": "w", "version": "1",
                          "tools": [{"name": "foo", "description": "d"}]}}
    old = SwarmSpec.model_validate({**raw(), "world": world})
    new = SwarmSpec.model_validate({**raw(), "world": {"url": "http://w.invalid"}})
    assert diff_specs(old, new) == []  # blanked by the file side, refilled by the controller
    long = [f"world:tool_{i}" for i in range(12)]
    old.types["calc"].tools = long
    new.types["calc"].tools = [*long[1:], "world:get_pr"]
    assert diff_specs(old, new) == [f"types.calc.tools: +world:get_pr -{long[0]}"]
    new.types["calc"].tools = long[::-1]
    assert diff_specs(old, new) == ["types.calc.tools: reordered"]
    assert world_tool_changes(old, new) == "world tools: -foo"
    assert world_tool_changes(old, old) is None


def test_keep_replicas_copies_live_counts_of_existing_types():
    old, new = SwarmSpec.model_validate(raw()), SwarmSpec.model_validate(raw())
    new.types["calc"].replicas = 20
    new.types["fresh"] = new.types["calc"].model_copy(update={"replicas": 7})
    assert keep_replicas(old, new) == ["calc: replicas kept (spec 20, live 2)"]
    assert (new.types["calc"].replicas, new.types["fresh"].replicas) == (2, 7)


# --------------------------------------------------------------------------- the API


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


async def test_heartbeat_carries_a_changed_type_spec(client, store, tick, tmp_path):
    spec = await applied(client, tmp_path, tick)
    w = next(w for w in store.list_workers("demo") if w["type"] == "calc")
    store.update_worker(w["id"], job_id="job-1", incarnation="inc-1")
    stored = lambda: store.get_swarm("demo")["spec"].types["calc"]  # with its pinned context window
    rev = type_spec_rev(stored())

    async def beat(rev: str | None, agents=("calc-0000", "calc-0001")) -> dict:
        hb = {"worker": w["id"], "deadline": "2026-10-15T09:00:00Z", "spec_rev": rev,
              "agents": [AgentHeartbeat(name=a, state="idle").model_dump(mode="json")
                         for a in agents]}
        r = await client.post(f"/workers/{w['id']}/heartbeat", json=hb,
                              params={"incarnation": "inc-1"})
        assert r.status_code == 200, r.text
        return r.json()

    out = await beat(rev)
    assert "type_spec" not in out
    spec.types["calc"].system_prompt = "You are calc v2."
    assert (await client.put("/swarms/demo", json=spec.model_dump(mode="json"))).status_code == 200
    out = await beat(rev)
    assert out["type_spec"]["system_prompt"] == "You are calc v2."
    assert out["spec_rev"] == type_spec_rev(stored()) != rev
    assert "type_spec" not in await beat(out["spec_rev"])


async def test_an_update_changes_the_worker_liveness_timings(client, store, tick, clock):
    spec = make_spec()
    assert (await client.put("/swarms/demo", json=spec.model_dump(mode="json"))).status_code == 200
    tick()
    wid = store.list_workers("demo")[0]["id"]
    store.update_worker(wid, last_heartbeat=clock().timestamp())
    clock.advance(100)

    async def state() -> str:
        return (await client.get(f"/workers/{wid}")).json()["state"]

    assert await state() == "unknown"  # the 90 s default
    spec.compute.worker_unknown_after, spec.compute.worker_dead_after = "2m", "4m"
    assert (await client.put("/swarms/demo", json=spec.model_dump(mode="json"))).status_code == 200
    assert await state() == "running"
    clock.advance(100)  # 200 s silent: dead by the old 180 s, not by the new 4 m
    assert not [a for a in tick() if a.kind in ("cancel", "crash")]
    assert await state() == "unknown"


async def test_an_effective_apply_bumps_the_generation_and_keeps_the_spec(client, store, tick,
                                                                         tmp_path):
    spec = await applied(client, tmp_path, tick)
    src = {"git_sha": "abc123", "ref": "main", "dirty": 1, "by": "alice", "extra": "dropped"}
    hdr = {"x-flock-source": json.dumps(src)}
    r = await client.put("/swarms/demo", json=spec.model_dump(mode="json"), headers=hdr)
    assert r.json()["generation"] == 1  # nothing changed: same generation, no history entry
    n = spec.types["researcher"].replicas
    spec.types["researcher"].replicas = n + 1
    r = await client.put("/swarms/demo?replicas=spec&dry_run=1", json=spec.model_dump(mode="json"),
                         headers=hdr)
    assert r.json()["dry_run"] and store.generation("demo") == 1  # a check stores nothing
    assert len(store.spec_history("demo")) == 1
    r = await client.put("/swarms/demo?replicas=spec", json=spec.model_dump(mode="json"),
                         headers=hdr)
    assert r.json()["generation"] == 2
    hist = store.spec_history("demo")
    assert [(h["generation"], h["diff"]) for h in hist] == [
        (1, ["created"]), (2, [f"types.researcher.replicas: {n} -> {n + 1}"])]
    assert hist[0]["source"] == {}
    assert hist[1]["source"] == {"git_sha": "abc123", "ref": "main", "dirty": True, "by": "alice"}
    evs = [e["data"] for e in store.read_events("_swarm.demo", 0) if e["type"] == "spec_applied"]
    assert [e["generation"] for e in evs] == [1, 2] and evs[1]["source"]["by"] == "alice"
    assert evs[1]["diff_summary"] == hist[1]["diff"]


# --------------------------------------------------------------------------- the CLI


def _experiment(root: Path, *, replicas: int = 2, prompt: str = "You are a researcher.",
                tools: str = TOOLS, **type_extra) -> Path:
    (root / "prompts").mkdir(parents=True, exist_ok=True)
    (root / "prompts" / "system.md").write_text(prompt)
    _tools(root, tools)
    t = {**HB, "model": "m", "replicas": replicas,
         "agents_per_worker": 2, "system_prompt_file": "prompts/system.md", "tools": ["add"],
         **type_extra}
    f = root / "spec.yaml"
    f.write_text(yaml.safe_dump({"swarm": "demo", "namespace": "ns",
                                 "bucket": "hf://buckets/ns/sh", "types": {"researcher": t},
                                 "inference": GPU}))
    return f


def test_cli_update_prints_what_changed_and_refuses_cleanly(tmp_path, monkeypatch, app, store,
                                                           tick):
    wire(monkeypatch, app)
    exp = tmp_path / "exp"
    f = _experiment(exp)
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 1 and "no swarm 'demo' is running" in r.output
    assert runner.invoke(cli.app, ["launch", str(f)]).exit_code == 0
    tick()
    v1 = store.get_swarm("demo")["spec"].tools_version
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 0 and r.output.strip() == "no changes", r.output

    _experiment(exp, replicas=3, prompt="You are a careful researcher.", tools=TOOLS_V2)
    v2 = pack(exp / "tools").version
    r = runner.invoke(cli.app, ["update", str(f), "--replicas", "spec"])  # file or folder
    assert r.exit_code == 0, r.output
    assert r.output.splitlines() == [f"tools: {v1} -> {v2} (+mul)",
                                     "prompts changed: researcher (system_prompt)",
                                     "types.researcher.replicas: 2 -> 3"]
    applied = store.get_swarm("demo")["spec"]
    assert applied.tools_version == v2 and applied.types["researcher"].replicas == 3
    assert applied.types["researcher"].system_prompt == "You are a careful researcher."

    (exp / "spec.yaml").write_text("types: [unclosed\n")
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 1 and "spec.yaml" in r.output
    _experiment(exp, tools="def broken(:\n")
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 1 and "tools:" in r.output
    assert store.get_swarm("demo")["spec"] == applied  # nothing was applied


def test_cli_update_probes_a_model_sent_somewhere_new(tmp_path, monkeypatch, app, tick):
    """An update probes again when what a model is sent to changes (new urls under the same
    block name), not only when a model or block is added."""
    wire(monkeypatch, app)
    exp = tmp_path / "exp"
    f = _experiment(exp)
    assert runner.invoke(cli.app, ["launch", str(f), "--skip-preflight"]).exit_code == 0
    tick()
    # --strict: a probe without the key (unverified) refuses the update, so it shows when one ran
    r = runner.invoke(cli.app, ["update", str(exp), "--strict"])  # the same spec: no probe
    assert r.exit_code == 0 and "preflight" not in r.output, r.output
    d = yaml.safe_load(f.read_text())
    d["inference"]["urls"] = ["http://gpu2.invalid/v1"]
    f.write_text(yaml.safe_dump(d))
    r = runner.invoke(cli.app, ["update", str(exp), "--strict"])
    assert r.exit_code == 1 and "preflight inference:m: unverified" in r.output, r.output


def test_cli_update_keeps_live_replicas_and_checks(tmp_path, monkeypatch, app, store, tick):
    wire(monkeypatch, app)
    exp = tmp_path / "exp"
    f = _experiment(exp, replicas=4)
    assert runner.invoke(cli.app, ["launch", str(f)]).exit_code == 0
    tick()
    _experiment(exp, replicas=2, prompt="v2")
    before = store.get_swarm("demo")["spec"]
    r = runner.invoke(cli.app, ["update", str(exp), "--check"])
    assert r.exit_code == 0, r.output
    assert r.output.splitlines() == ["preflight inference:m: unverified",
                                     "researcher: replicas kept (spec 2, live 4)",
                                     "prompts changed: researcher (system_prompt)",
                                     "check ok: nothing was applied"]
    assert store.get_swarm("demo")["spec"] == before
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 0 and "replicas kept (spec 2, live 4)" in r.output, r.output
    assert store.get_swarm("demo")["spec"].types["researcher"].replicas == 4
    r = runner.invoke(cli.app, ["update", str(exp), "--replicas", "spec"])
    assert r.exit_code == 0 and "types.researcher.replicas: 4 -> 2" in r.output, r.output
    assert store.get_swarm("demo")["spec"].types["researcher"].replicas == 2
    # --check names where each changed value came from
    (exp / "run.env").write_text("APW=3\n")
    _experiment(exp, replicas=2, prompt="v2", agents_per_worker="${APW:-2}", max_turns=7)
    r = runner.invoke(cli.app, ["update", str(exp), "--check", "--env", str(exp / "run.env")])
    assert r.exit_code == 0, r.output
    assert "types.researcher.agents_per_worker: 2 -> 3 (run.env APW)" in r.output
    assert "types.researcher.max_turns: - -> 7 (spec.yaml:" in r.output, r.output
    _experiment(exp, replicas=2, prompt="v2")
    # --check exits 1 on what the controller refuses (a tool the bundle lacks)
    d = yaml.safe_load(f.read_text())
    d["types"]["researcher"]["tools"] = ["add", "nope"]
    f.write_text(yaml.safe_dump(d))
    r = runner.invoke(cli.app, ["update", str(exp), "--check"])
    assert r.exit_code == 1 and "unknown tool(s) nope" in r.output, r.output
    assert runner.invoke(cli.app, ["update", str(exp), "--replicas", "x"]).exit_code == 1


async def test_put_replicas_keep_and_dry_run(client, store):
    spec = SwarmSpec.model_validate(raw(calc={**HB, "model": "m", "replicas": 2,
                                              "agents_per_worker": 2}))
    assert (await client.put("/swarms/demo", json=spec.model_dump(mode="json"))).status_code == 200
    spec.types["calc"].replicas = 5
    body = spec.model_dump(mode="json")
    r = await client.put("/swarms/demo", params={"dry_run": 1}, json=body)
    assert r.status_code == 200 and r.json()["dry_run"] is True
    assert store.get_swarm("demo")["spec"].types["calc"].replicas == 2  # stored nothing
    r = await client.put("/swarms/demo", params={"replicas": "keep"}, json=body)
    assert r.status_code == 200 and store.get_swarm("demo")["spec"].types["calc"].replicas == 2
    assert (await client.put("/swarms/demo", json=body)).status_code == 200  # default: spec
    assert store.get_swarm("demo")["spec"].types["calc"].replicas == 5


