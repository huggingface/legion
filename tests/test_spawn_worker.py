"""Worker side of spawn: the bootstrap puts the legion CLI on PATH, and unassigning an ephemeral
agent purges it. The harness env's identity (never the worker's LEGION_TOKEN) and the parent in
session_started: tests/test_runs.py::test_spawn."""

from __future__ import annotations

import asyncio

from legion.models import AssignCommand, PromptCommand

from .test_worker_main import FakeController, make_worker
from .worker_fakes import spec, until

CHILD = "leader-0000-analyst-0001"


async def test_unassign_purges_an_ephemeral_agent_and_keeps_others(tmp_path):
    """docs/spawn.md "Ephemeral types": after its final sync, the worker deletes an ephemeral
    agent's workdir, session log and state; a normal agent's files stay."""
    from legion.models import UnassignCommand

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
