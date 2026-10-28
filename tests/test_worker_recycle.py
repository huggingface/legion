"""Single-agent recycle (docs/pi.md "Memory"): a fresh harness process on the same session at a
turn boundary, from `recycle_after`, the memory watchdog or a changed model endpoint; and a
harness SIGKILLed mid-turn (the OOM killer) as a transient failure."""

from __future__ import annotations

import asyncio

from flock.worker.adapters.base import AdapterEvent

from .test_loop_worker import custom, start
from .worker_fakes import ScriptedAdapter, event_types, runtime, until


async def test_recycle_after_turns_restarts_on_the_same_session(tmp_path):
    rt, ad = runtime(tmp_path, idle_grace="5s", recycle_after={"turns": 2})
    await rt.setup(restore=False)
    rt.start_loop(start(max_rounds=5))
    await until(lambda: custom(rt, "loop_stopped"), timeout=5)
    assert len(ad.sent) == 5 and rt.state.turns == 5
    # rounds 1-2, 3-4 and 5 each on a fresh process; the same session throughout
    assert ad.starts == [None, "s-1", "s-1"]
    types = event_types(rt)
    assert "turn_interrupted" not in types
    assert [d["reason"] for d in custom(rt, "harness_recycled")] == ["turns", "turns"]


async def test_recycle_now_ends_an_idle_process(tmp_path):
    rt, ad = runtime(tmp_path, idle_grace="5s")
    await rt.setup(restore=False)
    assert not rt.recycle("memory")  # no process yet
    await rt.prompt("hi")
    await until(lambda: rt.state.turns == 1)
    rt.pss_mb = 900.0
    assert rt.recycle("memory")
    await until(lambda: rt.phase == "none")
    assert custom(rt, "harness_recycled") == [{"reason": "memory", "pss_mb": 900.0}]
    await rt.prompt("again")
    await until(lambda: rt.state.turns == 2)
    assert ad.starts == [None, "s-1"]


class EndpointAdapter(ScriptedAdapter):
    """update_spec reports a changed model endpoint once, as PiAdapter does."""

    model = "m-1"

    def update_spec(self, spec):
        changed, self.model = spec.model != self.model, spec.model
        return changed


async def test_endpoint_change_recycles_before_the_next_turn(tmp_path):
    ad = EndpointAdapter(tmp_path / "home")
    rt, _ = runtime(tmp_path, adapter=ad, idle_grace="5s")
    await rt.setup(restore=False)
    await rt.prompt("one")
    await until(lambda: rt.state.turns == 1)
    rt.set_live(spec=rt.spec.model_copy(update={"model": "m-2"}))
    await rt.prompt("two")  # a turn boundary: the new spec applies, then a fresh process runs it
    await until(lambda: rt.state.turns == 2)
    assert ad.starts == [None, "s-1"] and ad.sent == ["one", "two"]
    assert [d["reason"] for d in custom(rt, "harness_recycled")] == ["endpoint"]


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
