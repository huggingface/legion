"""A harness SIGKILLed mid-turn (the OOM killer) is a transient failure."""

from __future__ import annotations

import asyncio

from flock.worker.adapters.base import AdapterEvent

from .test_loop_worker import custom, start
from .worker_fakes import ScriptedAdapter, runtime, until


class OomOnce(ScriptedAdapter):
    """The first turn's process is SIGKILLed mid-turn (returncode -9)."""

    killed = False

    async def _finish(self):
        if not OomOnce.killed:
            OomOnce.killed = True
            await asyncio.sleep(0.02)
            self.live = False
            self._turn = None
            self._q.put(AdapterEvent("exited", data={"returncode": -9}))
            return
        await super()._finish()


async def test_oom_killed_turn_is_transient(tmp_path):
    ad = OomOnce(tmp_path / "home")
    rt, _ = runtime(tmp_path, adapter=ad, idle_grace="5s")
    await rt.setup(restore=False)
    rt.start_loop(start(max_rounds=2))
    await until(lambda: custom(rt, "loop_stopped"), timeout=10)
    assert [d["reason"] for d in custom(rt, "turn_interrupted")] == ["oom_killed"]
    assert rt._loop_errors == 0 and rt.state.turns == 2  # retried, not counted as an error
