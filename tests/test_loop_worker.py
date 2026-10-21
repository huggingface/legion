"""Worker side of `loop:` (docs/loop.md): `every: turn_end` re-prompts, `loop_stop`, three failed
turns in a row, rendering of the built-in placeholders and `{last_summary}`."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

from flock.models import LoopMeta, LoopStartCommand, TokenUsage
from flock.worker.adapters.base import AdapterEvent
from flock.worker.loop import render

from .worker_fakes import ScriptedAdapter, runtime, until

T0 = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


def meta(**kw) -> LoopMeta:
    return LoopMeta(**{"kind": "loop", "type": "researcher", **kw})


def start(text: str = "round {round} ({why})", **kw) -> LoopStartCommand:
    """A loop_start for a new run of the loop (each call: a fresh run id, as `--start` sends)."""
    kw.setdefault("run", uuid.uuid4().hex[:12])
    return LoopStartCommand(agent="researcher-0001", text=text,
                            meta=meta(round=0, why="turn_end", started_at=datetime.now(UTC)), **kw)


def custom(rt, kind: str) -> list[dict]:
    return [e.data["data"] for e in list(rt.outbox._q) if e.type == kind]


def test_render_fills_builtins_and_leaves_the_rest():
    m = meta(round=4, started_at=T0, wall_s=7200, cap_usd=3.6, why="every 120s")
    out = render("r{round} {elapsed}/{wall} min ${spend} of ${cap} ({why}) {last_summary} "
                 "{other} {cap:.2f} {ROUND} {{x}}", m, spend=0.91234, last_summary="did it",
                 now=T0 + timedelta(minutes=12, seconds=20))
    assert out == "r4 12/120 min $0.912 of $3.60 (every 120s) did it {other} {cap:.2f} {ROUND} {{x}}"
    bare = render("{round}|{wall}|{cap}|{last_summary}", meta(), spend=0, last_summary=None, now=T0)
    assert bare == "|none|none|(none yet)"


async def test_turn_end_loop_reprompts_until_max_rounds(tmp_path):
    rt, ad = runtime(tmp_path, idle_grace="5s")
    await rt.setup(restore=False)
    rt.start_loop(start(max_rounds=3))
    await until(lambda: custom(rt, "loop_stopped"), timeout=5)
    assert ad.sent == ["round 1 (start)", "round 2 (turn_end)", "round 3 (turn_end)"]
    assert rt.state.turns == 3  # one turn per prompt: nothing folded, nothing stacked
    sends = [i for i, (k, _) in enumerate(ad.log) if k == "send"]
    assert len(sends) == 3
    assert [d["round"] for d in custom(rt, "loop_prompt")] == [1, 2, 3]
    assert custom(rt, "loop_stopped") == [{"reason": "max_rounds", "round": 3, "active": True}]
    ended = custom(rt, "turn_ended")
    assert [d["tool_calls"] for d in ended] == [0, 0, 0]
    await asyncio.sleep(0.2)
    assert len(ad.sent) == 3 and rt.loop is None
    await rt.stop()


async def test_loop_stop_ends_the_worker_loop(tmp_path):
    rt, ad = runtime(tmp_path, idle_grace="5s", adapter=ScriptedAdapter(tmp_path / "h", turn_time=0.1))
    await rt.setup(restore=False)
    rt.start_loop(start())
    await until(lambda: len(ad.sent) >= 2, timeout=5)
    rt.stop_loop("until")
    n = len(ad.sent)
    await asyncio.sleep(0.4)  # the turn in flight finishes, nothing follows it
    assert len(ad.sent) == n and rt.turn_id is None
    assert custom(rt, "loop_stopped") == [{"reason": "until", "round": n, "active": True}]
    rt.stop_loop("again")  # always answered, so the controller never waits for an ack in vain
    assert custom(rt, "loop_stopped")[-1] == {"reason": "again", "round": n, "active": False}
    await rt.stop()


class FailingAdapter(ScriptedAdapter):
    async def _finish(self):
        await asyncio.sleep(self.turn_time)
        self._q.put(AdapterEvent("turn_ended", status="error", error="api_error"))
        self._turn = None


async def test_three_failed_turns_in_a_row_end_the_loop(tmp_path, monkeypatch):
    monkeypatch.setattr("flock.worker.agents.LOOP_ERROR_BACKOFF", 0.01)
    rt, ad = runtime(tmp_path, idle_grace="5s", adapter=FailingAdapter(tmp_path / "h"))
    await rt.setup(restore=False)
    rt.start_loop(start())
    await until(lambda: custom(rt, "loop_stopped"), timeout=5)
    assert len(ad.sent) == 3 and custom(rt, "loop_stopped")[0]["reason"] == "errors"
    assert [d["why"] for d in custom(rt, "loop_prompt")] == ["start", "retry after error",
                                                              "retry after error"]
    await rt.stop()


async def test_deadline_ends_the_worker_loop(tmp_path):
    rt, ad = runtime(tmp_path, idle_grace="5s")
    await rt.setup(restore=False)
    rt.start_loop(start(deadline=datetime.now(UTC) - timedelta(seconds=1)))
    await until(lambda: custom(rt, "loop_stopped"), timeout=5)
    assert ad.sent == [] and custom(rt, "loop_stopped")[0]["reason"] == "deadline"


class DoneAdapter(ScriptedAdapter):
    """Each turn calls one tool, then `done` with a summary naming the prompt."""

    async def _finish(self):
        await asyncio.sleep(self.turn_time)
        text = self.sent[-1]
        self._q.put(AdapterEvent("message", message={"role": "assistant", "content": [
            {"type": "toolCall", "id": "c1", "name": "bash", "arguments": {}},
            {"type": "toolCall", "id": "c2", "name": "done", "arguments": {}}]}))
        self._q.put(AdapterEvent("message", message={"role": "assistant", "fromDoneTool": True,
                                                     "content": [{"type": "text", "text": f"did {text}"}]}))
        self._q.put(AdapterEvent("usage", model="m-1", usage=TokenUsage(output=1000)))
        self._q.put(AdapterEvent("turn_ended", status="completed"))
        self._turn = None


async def test_controller_loop_prompts_render_with_summary_and_spend(tmp_path):
    rt, ad = runtime(tmp_path, idle_grace="5s", adapter=DoneAdapter(tmp_path / "h"))
    await rt.setup(restore=False)
    m = meta(round=1, why="every 30s", started_at=datetime.now(UTC), cap_usd=1)
    await rt.prompt("r{round}: {last_summary}", loop=m)
    await until(lambda: rt.state.turns == 1)
    await rt.prompt("r{round}: {last_summary} ${spend}/${cap}", loop=m.model_copy(update={"round": 2}))
    await until(lambda: rt.state.turns == 2)
    assert ad.sent[0] == "r1: (none yet)"
    # m-1 is priced as the default model (75 USD per 1M output tokens): 1000 tokens = $0.075
    assert ad.sent[1] == "r2: did r1: (none yet) $0.075/$1.00"
    assert [d["tool_calls"] for d in custom(rt, "turn_ended")] == [1, 1]  # `done` not counted
    assert rt.state.last_summary == "did " + ad.sent[1] and rt.state.loop_round == 2
    # a final prompt ends a worker-side loop first
    rt.loop = start()
    await rt.prompt("final: {last_summary}", loop=meta(kind="final", round=2, why="final: until",
                                                       add_tools=["submit_final"]))
    assert rt.loop is None and custom(rt, "loop_stopped")[-1]["reason"] == "final"
    assert custom(rt, "loop_prompt")[-1] == {"kind": "final", "round": 2, "why": "final: until",
                                             "add_tools": ["submit_final"]}
    await rt.stop()


async def test_a_new_run_restarts_from_round_one_a_repeat_does_not(tmp_path):
    """Finding 4: `flock loop --start` after an exhausted loop (new run id) starts at round
    1; the same run sent again (a lost command, a live update) keeps its count."""
    rt, ad = runtime(tmp_path, idle_grace="5s")
    await rt.setup(restore=False)
    first = start(max_rounds=1)
    rt.start_loop(first)
    await until(lambda: len(custom(rt, "loop_stopped")) == 1, timeout=5)
    rt.start_loop(first.model_copy())  # same run again: already at max_rounds
    await until(lambda: len(custom(rt, "loop_stopped")) == 2, timeout=5)
    assert ad.sent == ["round 1 (start)"]
    rt.start_loop(start(max_rounds=1))  # a new run
    await until(lambda: len(custom(rt, "loop_stopped")) == 3, timeout=5)
    assert ad.sent == ["round 1 (start)", "round 1 (start)"]
    await rt.stop()


async def test_a_deferred_loop_prompt_is_not_a_round(tmp_path):
    rt, ad = runtime(tmp_path, idle_grace="5s")
    await rt.setup(restore=False)
    rt.allow_starts = False  # draining
    await rt.prompt("r{round}", loop=meta(round=None, why="turn_end"))
    [d] = custom(rt, "prompt_deferred")
    assert d["loop"] == {"kind": "loop", "round": 1} and d["text"] == "r1"
    assert rt.state.loop_round == 0 and ad.sent == []
