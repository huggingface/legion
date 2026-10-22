"""Worker side of spawn: the harness env carries the agent's identity and scoped token, never
the worker's FLOCK_TOKEN; session_started names the parent."""

from __future__ import annotations

import asyncio

from flock.controller.jobs import bootstrap_command
from flock.models import AssignCommand, PromptCommand

from .test_worker_main import FakeController, make_worker
from .worker_fakes import spec, until

CHILD = "leader-0000-analyst-0001"


async def test_harness_env_and_session_started_carry_lineage(tmp_path):
    ctrl = FakeController()
    ctrl.batches = [
        [AssignCommand(agent=CHILD, type="analyst", spec=spec(env={"X": "1"}), restore=False,
                       parent="leader-0000", agent_token="flock_at_child"),
         AssignCommand(agent="leader-0000", type="analyst", spec=spec(), restore=False,
                       agent_token="flock_at_leader")],
        [PromptCommand(agent=CHILD, prompt_id="p1", text="go"),
         PromptCommand(agent="leader-0000", prompt_id="p2", text="go")],
    ]
    worker, adapters = make_worker(tmp_path, ctrl)
    run = asyncio.create_task(worker.run())
    await until(lambda: sum(e.type == "session_started" for e in ctrl.events) == 2, timeout=5)
    env = adapters[CHILD].envs[0]
    assert env["X"] == "1"
    assert env["FLOCK_AGENT"] == CHILD and env["FLOCK_PARENT"] == "leader-0000"
    assert env["FLOCK_URL"] == "http://controller"
    assert env["FLOCK_AGENT_TOKEN"] == "flock_at_child"
    assert env["FLOCK_TOKEN"] == ""  # never the worker's own token
    assert adapters["leader-0000"].envs[0]["FLOCK_PARENT"] == ""
    # worker events carry the pi entry; custom data sits under "data"
    started = {e.agent: e.data["data"] for e in ctrl.events if e.type == "session_started"}
    assert started[CHILD]["parent"] == "leader-0000"
    assert started["leader-0000"]["parent"] is None
    run.cancel()
    try:
        await run
    except asyncio.CancelledError:
        pass
    await worker.client.aclose()


def test_bootstrap_puts_the_flock_cli_on_path():
    script = bootstrap_command()[2]
    assert 'export PATH="$HOME/.local/bin:$PATH"' in script  # where `uv tool install` puts it
    check = script.index("command -v flock")
    assert script.index("uv tool install") < check < script.index("exec flock-worker")


async def test_unassign_purges_an_ephemeral_agent_and_keeps_others(tmp_path):
    """docs/spawn.md "Ephemeral types": after its final sync, the worker deletes an ephemeral
    agent's workdir, session log, state and pin; a normal agent's files stay."""
    from flock.models import UnassignCommand

    keep = "leader-0000-analyst-0002"
    ctrl = FakeController()
    ctrl.batches = [
        [AssignCommand(agent=CHILD, type="analyst", spec=spec(ephemeral=True), restore=False,
                       parent="leader-0000", agent_token="t1"),
         AssignCommand(agent=keep, type="analyst", spec=spec(), restore=False,
                       parent="leader-0000", agent_token="t2")],
        [PromptCommand(agent=CHILD, prompt_id="p1", text="go"),
         PromptCommand(agent=keep, prompt_id="p2", text="go")],
    ]
    worker, _ = make_worker(tmp_path, ctrl)
    run = asyncio.create_task(worker.run())
    await until(lambda: sum(e.type == "turn_ended" for e in ctrl.events) == 2, timeout=5)
    root = tmp_path / "agents"
    for name in (CHILD, keep):
        rt = worker.agents[name]
        rt.save_state()
        assert rt.workdir.is_dir() and rt.state_path.is_file()
    ctrl.batches.append([UnassignCommand(agent=CHILD), UnassignCommand(agent=keep)])
    await until(lambda: not worker.agents, timeout=5)
    assert not (root / CHILD).exists()
    assert not list((root / "_sessions").glob(f"{CHILD}.*"))
    assert (root / keep).is_dir() and list((root / "_sessions").glob(f"{keep}.*"))
    run.cancel()
    try:
        await run
    except asyncio.CancelledError:
        pass
    await worker.client.aclose()
