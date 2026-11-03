"""Run 4's spawner wave: agent-spawner agents may spawn up to four agent-child peers, children
cannot spawn, both get the shared system prompt plus their group section, a child's first
prompt is the spawner's context, and an isolated Pi agent gets what `flock spawn` needs."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import TypeAdapter

from flock.cli import load_spec
from flock.controller import spawn as spawning
from flock.models import AssignCommand, Command

from .test_isolation import kernel_namespaces  # noqa: F401 - the fixture, for the namespace test

EXP = Path(__file__).resolve().parents[1] / "experiments/run4"
CMD = TypeAdapter(Command)
SPAWNER_TEXT = "You can start up to 4 peers of your own kind"
CHILD_TEXT = "You were started by another agent (your parent, `$FLOCK_PARENT`)"


def run4_spec(spawners: int = 1):
    spec = load_spec(str(EXP / "spec.yaml"))  # as `flock apply` reads it: prompts inlined
    spec.types["agent-spawner"].replicas = spawners
    # The in-process controller has no world or board; the types are what is under test.
    data = spec.model_dump(mode="json")
    data.update(world=None, comms=None)
    for t in data["types"].values():
        t.update(tools=[], comms=False)
    return type(spec).model_validate(data)


def as_agent(app, token) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                             headers={"Authorization": f"Bearer {token}"})


def test_spec_types_are_clones_of_agent_with_the_spawn_policy_and_caps():
    spec = load_spec(str(EXP / "spec.yaml"))
    agent = spec.types["agent"]
    skip = {"system_prompt", "system_prompt_append_file", "workers", "replicas_ramp"}
    for name in ("agent-spawner", "agent-child"):
        t = spec.types[name]
        assert t.replicas == 0
        assert t.model_dump(exclude=skip) == agent.model_dump(exclude=skip), name
        assert t.system_prompt.startswith(agent.system_prompt.rstrip("\n") + "\n\n")
    assert SPAWNER_TEXT in spec.types["agent-spawner"].system_prompt
    assert CHILD_TEXT in spec.types["agent-child"].system_prompt
    assert SPAWNER_TEXT not in agent.system_prompt and CHILD_TEXT not in agent.system_prompt
    assert spec.types["agent-spawner"].workers.model_dump() == {"initial": 2, "max_extra": 0}
    assert spec.types["agent-child"].workers.model_dump() == {"initial": 0, "max_extra": 7}
    assert 2 * 12 >= 20 and 7 * 12 >= 80
    assert set(spec.spawn) == {"agent-spawner"}
    rule = spec.spawn["agent-spawner"]
    assert rule.types == {"agent-child": 4} and rule.depth == 1
    assert spec.limits.alive == 400 and spec.limits.spawned_total == 100


def test_slurm_spec_has_both_types_without_aliases(monkeypatch):
    from flock.specrender import render

    monkeypatch.delenv("LOOP_UNTIL", raising=False)
    data = render(EXP / "spec.yaml")
    assert {"agent", "agent-spawner", "agent-child"} <= set(data["types"])
    assert data["types"]["agent-child"]["loop"] is not data["types"]["agent"]["loop"]
    assert data["types"]["agent-child"]["system_prompt_append_file"] == "prompts/group-child.md"
    assert data["types"]["agent"]["loop"]["while"]["until"] == "48h"
    assert data["types"]["agent-spawner"]["replicas_ramp"] is None  # only `agent` ramps
    assert "&id" not in yaml.safe_dump(data)
    monkeypatch.setenv("LOOP_UNTIL", "30m")
    smoke = render(EXP / "spec.yaml")
    assert all(t["loop"]["while"]["until"] == "30m" for t in smoke["types"].values())


async def test_spawner_spawns_four_children_and_the_fifth_is_refused(client, app, store, tick):
    spec = run4_spec()
    r = await client.put(f"/swarms/{spec.swarm}", json=spec.model_dump(mode="json"))
    assert r.status_code == 200, r.text
    tick()
    agents = (await client.get(f"/swarms/{spec.swarm}/agents")).json()["agents"]
    [spawner] = [a["name"] for a in agents if a["type"] == "agent-spawner"]
    tok = spawning.agent_token(store, spawner, spec.swarm, 0)
    async with as_agent(app, tok) as c:
        kids = []
        for i in range(4):
            r = await c.post(f"/agents/{spawner}/spawn", json={
                "type": "agent-child", "prompt": f"context {i}: PR 12, open: tests", "key": f"k{i}"})
            assert r.status_code == 201, r.text
            assert r.json()["depth"] == 0 and r.json()["parent"] == spawner
            kids.append(r.json()["name"])
        r = await c.post(f"/agents/{spawner}/spawn", json={"type": "agent-child", "key": "k4"})
        assert r.status_code == 409 and r.json()["reason"] == "max_children"
        r = await c.post(f"/agents/{spawner}/spawn", json={"type": "agent", "key": "k5"})
        assert r.status_code == 409 and r.json()["reason"] == "type_not_allowed"
        # the same key is the same child, so a retry does not take a place
        r = await c.post(f"/agents/{spawner}/spawn", json={"type": "agent-child", "key": "k0"})
        assert r.status_code == 200 and r.json()["name"] == kids[0]
    child_tok = spawning.agent_token(store, kids[0], spec.swarm, 0)
    async with as_agent(app, child_tok) as c:
        r = await c.post(f"/agents/{kids[0]}/spawn", json={"type": "agent-child"})
        assert r.status_code == 409 and r.json()["reason"] in {"depth", "type_not_allowed"}
        assert (await c.post(f"/agents/{spawner}/prompt", json={"text": "split?"})).status_code == 200
    tick()
    rows = {n: store.get_agent(n) for n in kids}
    workers = {r["worker"] for r in rows.values()}
    assert len(workers) == 1 and next(iter(workers)).endswith("agent-child-w0000")
    wid = next(iter(workers))
    cmds = [CMD.validate_json(c["body"]) for c in store.commands_for(wid)]
    assigned = {c.agent: c for c in cmds if isinstance(c, AssignCommand)}
    for i, name in enumerate(kids):
        cmd = assigned[name]
        assert cmd.parent == spawner and cmd.agent_token.startswith("flock_at_")
        assert CHILD_TEXT in cmd.spec.system_prompt
        assert SPAWNER_TEXT not in cmd.spec.system_prompt
        assert "# Project: Python Chatbot" in cmd.spec.system_prompt
        assert cmd.spec.max_turns == 150 and cmd.spec.harness.value == "pi"
    first = {p["agent"]: json.loads(p["body"]) for p in store.prompts_for_worker(wid)}
    assert {n: first[n]["text"] for n in kids} == {
        n: f"context {i}: PR 12, open: tests" for i, n in enumerate(kids)}
    # the spawner's own assign carries the spawner section
    swid = store.get_agent(spawner)["worker"]
    scmds = [CMD.validate_json(c["body"]) for c in store.commands_for(swid)]
    [sassign] = [c for c in scmds if isinstance(c, AssignCommand) and c.agent == spawner]
    assert SPAWNER_TEXT in sassign.spec.system_prompt and CHILD_TEXT not in sassign.spec.system_prompt


async def test_isolated_pi_env_carries_the_agent_token_and_the_flock_cli(tmp_path, monkeypatch):
    """What `flock spawn` needs inside the Pi namespace: FLOCK_URL and the agent's own token
    (never FLOCK_TOKEN) in the environment, and the venv's `flock` on PATH under a read-only
    mount of the chroot."""
    import shutil
    import sys

    from flock import isolation
    from flock.models import TypeSpec
    from flock.worker.adapters import pi

    venv_bin = str(Path(sys.executable).parent)
    flock_cli = Path(venv_bin) / "flock"
    if not flock_cli.exists():
        pytest.skip("the flock console script is not installed next to this interpreter")
    monkeypatch.setenv("FLOCK_PROCESS_ISOLATION", "1")
    monkeypatch.setenv("FLOCK_TOKEN", "worker-token")
    monkeypatch.setenv("PATH", f"{venv_bin}:/usr/bin:/bin")
    workspace, home = tmp_path / "cedar", tmp_path / "home"
    workspace.mkdir()
    adapter = pi.PiAdapter(workspace, agent="cedar", home=home, pi_bin="/usr/bin/false",
                           model_config=pi.PiModelConfig("http://fake/v1", "MODEL_KEY", "m", "fake"))
    calls = []

    async def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("stop before exec")

    monkeypatch.setattr(pi.asyncio, "create_subprocess_exec", spawn)
    lineage = {"FLOCK_URL": "http://controller:7860", "FLOCK_AGENT": "cedar",
               "FLOCK_PARENT": "amber-otter", "FLOCK_AGENT_TOKEN": "flock_at_abc",
               "FLOCK_TOKEN": ""}  # as the worker's _set_lineage sets them
    try:
        with pytest.raises(RuntimeError, match="stop before exec"):
            await adapter.start(workspace, None, TypeSpec(harness="pi", model="m"),
                                env={"MODEL_KEY": "k", **lineage})
        args, kwargs = calls[0]
        env = kwargs["env"]
        for k in ("FLOCK_URL", "FLOCK_AGENT", "FLOCK_PARENT", "FLOCK_AGENT_TOKEN"):
            assert env[k] == lineage[k], k
        assert "FLOCK_TOKEN" not in env
        found = shutil.which("flock", path=env["PATH"])
        assert found and Path(found).resolve() == flock_cli.resolve()
        config = json.loads(args[args.index(str(Path(isolation.__file__).resolve())) + 1])
        roots = config["readonly"]
        assert any(str(flock_cli.resolve()).startswith(r.rstrip("/") + "/") for r in roots)
        import flock as pkg  # the package the script imports (an editable checkout or the venv)
        assert any(str(Path(pkg.__file__).resolve()).startswith(r.rstrip("/") + "/") for r in roots)
    finally:
        await adapter.stop()


def _fake_controller():
    """A local HTTP server answering POST /agents/<name>/spawn; records each request."""
    import http.server
    import threading

    seen: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
            out = json.dumps({"name": "brisk-lantern", "created": True, "parent": "cedar",
                              "type": body["type"], "depth": 0}).encode()
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def _agent_env(url: str, home: Path) -> dict[str, str]:
    """The env an isolated Pi agent gets (harness_env, then the run4 allow-list), with the
    worker's own token set in the worker's environment."""
    import sys

    from flock.worker.adapters import pi

    worker = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "FLOCK_TOKEN": "worker-token",
              "HF_TOKEN": "hf-secret"}
    lineage = {"FLOCK_URL": url, "FLOCK_AGENT": "cedar", "FLOCK_PARENT": "",
               "FLOCK_AGENT_TOKEN": "flock_at_cedar", "FLOCK_TOKEN": ""}
    env = pi.isolated_env(pi.harness_env(lineage, home, base=worker), None)[0]
    env["HOME"] = str(home)
    return env


def _check_spawn(tmp_path: Path, wrap=None) -> None:
    """`flock spawn` as an agent runs it (`wrap`: inside the namespace), against a fake
    controller: it posts under the agent's own name with the agent token, never the worker's."""
    import subprocess
    import sys

    flock_cli = Path(sys.executable).parent / "flock"
    if not flock_cli.exists():
        pytest.skip("the flock console script is not installed next to this interpreter")
    server, seen = _fake_controller()
    workspace, home = tmp_path / "cedar", tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    try:
        env = _agent_env(f"http://127.0.0.1:{server.server_address[1]}", home)
        assert "FLOCK_TOKEN" not in env and "HF_TOKEN" not in env
        argv = [str(flock_cli), "spawn", "agent-child", "--prompt", "goal: PR 7", "--key", "k1"]
        cmd = wrap(argv, workspace, home) if wrap else argv
        r = subprocess.run(cmd, env=env, cwd=workspace, capture_output=True, text=True,
                           timeout=60, check=False)
        assert r.returncode == 0 and r.stdout.strip() == "brisk-lantern", r.stderr
        [req] = seen
        assert req["path"] == "/agents/cedar/spawn"
        assert req["body"] == {"type": "agent-child", "prompt": "goal: PR 7", "key": "k1"}
        assert req["headers"]["X-Flock-Token"] == "flock_at_cedar"
        assert "worker-token" not in json.dumps(req["headers"])
    finally:
        server.shutdown()


def test_flock_spawn_works_with_the_isolated_agent_env(tmp_path):
    _check_spawn(tmp_path)


@pytest.mark.usefixtures("kernel_namespaces")
def test_flock_spawn_works_inside_the_pi_namespace(tmp_path):
    from flock import isolation

    def wrap(argv, workspace, home):
        return isolation.command(argv, tmp_path / "root", cwd=workspace,
                                 writable=[workspace, home], network=True)

    _check_spawn(tmp_path, wrap)
