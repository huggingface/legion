"""Worker side of `loop:` (docs/loop.md): `every: turn_end` re-prompts, `loop_stop`, three failed
turns in a row, rendering of the built-in placeholders and `{last_summary}`."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from flock.models import LoopMeta, LoopStartCommand, TokenUsage
from flock.tools.loader import ToolDef
from flock.tools.runner import Result
from flock.tools.toolset import Toolset
from flock.worker import agents as agents_mod
from flock.worker.adapters.base import AdapterEvent
from flock.worker.adapters.llm import LLMAdapter
from flock.worker.agents import AgentRuntime
from flock.worker.events import Outbox
from flock.worker.loop import render

from .test_worker_llm import llm_spec
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
    ws = render("{workspace}", meta(), spend=0, last_summary=None, now=T0,
                workspace="182k files, 900 MB; largest: snaps/ 140k")
    assert ws == "182k files, 900 MB; largest: snaps/ 140k"
    assert render("{workspace}", meta(), spend=0, last_summary=None, now=T0) == "not measured yet"


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


class MaxTurnsAdapter(ScriptedAdapter):
    def __init__(self, *a, data: dict, **kw):
        super().__init__(*a, **kw)
        self.data = data

    async def _finish(self):
        await asyncio.sleep(self.turn_time)
        self._q.put(AdapterEvent("turn_ended", status="error", error="max_turns exceeded",
                                 data=self.data))
        self._turn = None


@pytest.mark.parametrize("data", [{"reason": "max_turns"},             # llm, pi, codex
                                  {"subtype": "error_max_turns"}])     # claude
async def test_max_turns_is_not_a_failed_loop_turn(tmp_path, data):
    """Defect 5: with tool_choice required, ending at max_turns is normal; 5 in a row run the
    loop to max_rounds instead of ending it with `errors`."""
    rt, ad = runtime(tmp_path, idle_grace="5s", adapter=MaxTurnsAdapter(tmp_path / "h", data=data))
    await rt.setup(restore=False)
    rt.start_loop(start(max_rounds=5))
    await until(lambda: custom(rt, "loop_stopped"), timeout=5)
    assert custom(rt, "loop_stopped")[0]["reason"] == "max_rounds" and len(ad.sent) == 5
    assert [d["why"] for d in custom(rt, "loop_prompt")] == ["start"] + ["turn_end"] * 4
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
    await rt.prompt("final: {last_summary}", loop=meta(kind="final", round=2, why="final: until"))
    assert rt.loop is None and custom(rt, "loop_stopped")[-1]["reason"] == "final"
    assert custom(rt, "loop_prompt")[-1] == {"kind": "final", "round": 2, "why": "final: until"}
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


async def test_a_wake_outside_the_round_count_keeps_the_round(tmp_path):
    """`counted: false` (wake.counts_as_round false): the prompt carries the current round,
    the counter does not move, and a deferral gives nothing back."""
    rt, ad = runtime(tmp_path, idle_grace="5s")
    await rt.setup(restore=False)
    rt.state.loop_round = 3
    wake = meta(round=None, why="wake: comms.mention", counted=False)
    rt.allow_starts = False  # draining
    await rt.prompt("r{round}", loop=wake)
    [d] = custom(rt, "prompt_deferred")
    assert d["loop"] == {"kind": "loop", "round": 3, "counted": False} and rt.state.loop_round == 3
    rt.allow_starts = True
    await rt.prompt("r{round} {why}", loop=wake)
    await until(lambda: rt.state.turns == 1)
    assert ad.sent == ["r3 wake: comms.mention"] and rt.state.loop_round == 3
    assert custom(rt, "loop_prompt")[-1] == {"kind": "loop", "round": 3, "why": "wake: comms.mention",
                                             "counted": False}
    await rt.stop()


async def test_a_call_runs_with_the_turns_pinned_tools(tmp_path):
    """Review of #19: a tools re-apply lands after the `llm` adapter pinned v1 for a turn but
    before the worker consumed its `turn_started`. The call runs with v1."""
    called: list[str] = []

    def tools(version: str) -> Toolset:
        async def call(entry, args):
            called.append(version)
            return Result("ok", False)

        return Toolset(version, tmp_path, "s", "a", tmp_path / "data", tools=(
            ToolDef("action", "Dummy action", {"type": "object", "properties": {}}, "dummy",
                    "action", builtin="dummy:action"),), builtin_call=call)

    current = [tools("v1")]
    toolbox = lambda: current[0]
    adapter = LLMAdapter(tmp_path / "agent", agent="a-0001", home=tmp_path / "home", toolbox=toolbox)
    outbox = Outbox()
    rt = AgentRuntime("a-0001", "a", llm_spec(), swarm="s", work_root=tmp_path / "agents",
                      adapter=adapter, outbox=outbox, toolbox=toolbox)
    rt.workdir.mkdir(parents=True)
    await rt.setup(restore=False)
    entered, release = asyncio.Event(), asyncio.Event()
    n = 0

    async def fake_complete():
        nonlocal n
        n += 1
        if n == 1:
            entered.set()
            await release.wait()
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function", "function": {"name": "action", "arguments": "{}"}}]}
        else:
            msg = {"role": "assistant", "content": "Done."}
        return {"choices": [{"message": msg}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    adapter._complete = fake_complete
    await adapter.start(rt.workdir, None, rt.spec, env={})
    pump = None
    try:
        await adapter.send("Take one action.")
        await asyncio.wait_for(entered.wait(), 3)
        current[0] = tools("v2")  # a heartbeat installs v2 mid-turn
        pump = asyncio.create_task(rt._pump())
        await until(lambda: rt.turn_id is not None)
        release.set()
        await until(lambda: rt.state.turns == 1)
        evs = outbox.take()
        ended = next(e.data["data"] for e in evs if e.type == "turn_ended")
        assert called == ["v1"] and ended["tools_version"] == "v1"
        assert ended["tool_calls"] == 1
    finally:
        await adapter.stop()
        if pump is not None:
            await asyncio.wait_for(pump, 3)


class StoppedAdapter(ScriptedAdapter):
    """Turn 1 calls `done` with a summary; later turns call tools and stop at max_turns."""

    async def _finish(self):
        await asyncio.sleep(self.turn_time)
        if len(self.sent) == 1:
            self._q.put(AdapterEvent("message", message={"role": "assistant", "fromDoneTool": True,
                                                         "content": [{"type": "text", "text": "built v1"}]}))
            self._q.put(AdapterEvent("turn_ended", status="completed"))
        else:
            for name in ("edit_file", "run_bot", "read"):
                self._q.put(AdapterEvent("message", message={"role": "assistant", "content": [
                    {"type": "text", "text": "let me check"},
                    {"type": "toolCall", "id": name, "name": name, "arguments": {}}]}))
            self._q.put(AdapterEvent("turn_ended", status="error", error="max_turns exceeded",
                                     data={"reason": "max_turns"}))
        self._turn = None


async def test_a_turn_stopped_without_done_replaces_the_stale_summary(tmp_path):
    """A loop turn that ends at max_turns used to leave `last_summary` at an older turn's
    `done` summary, so every later prompt repeated it."""
    rt, ad = runtime(tmp_path, idle_grace="5s", adapter=StoppedAdapter(tmp_path / "h"))
    await rt.setup(restore=False)
    rt.start_loop(start("round {round}: {last_summary}", max_rounds=3))
    await until(lambda: custom(rt, "loop_stopped"), timeout=5)
    line = ("previous turn ended (max_turns) after 3 tool call(s) without finishing "
            "(last calls: edit_file, run_bot, read)")
    assert ad.sent == ["round 1: (none yet)", "round 2: built v1", f"round 3: {line}"]
    assert rt.state.last_summary == line
    await rt.stop()


class TransportFailureAdapter(ScriptedAdapter):
    async def _finish(self):
        await asyncio.sleep(self.turn_time)
        self._q.put(AdapterEvent("turn_ended", status="error", error="fetch failed",
                                data={"reason": "model_transport"}))
        self._turn = None


async def test_transport_stalls_do_not_exhaust_worker_loop(tmp_path, monkeypatch):
    monkeypatch.setattr("flock.worker.agents.LOOP_ERROR_BACKOFF", 0.01)
    monkeypatch.setattr("flock.worker.agents.LOOP_TRANSIENT_BACKOFF", 0.01)
    rt, ad = runtime(tmp_path, idle_grace="5s", adapter=TransportFailureAdapter(tmp_path / "h"))
    await rt.setup(restore=False)
    rt.start_loop(start(max_rounds=5))
    await until(lambda: custom(rt, "loop_stopped"), timeout=5)
    assert len(ad.sent) == 5
    assert custom(rt, "loop_stopped")[0]["reason"] == "max_rounds"
    assert rt._loop_errors == 0
    await rt.stop()


def test_transient_backoff_doubles_with_jitter_up_to_the_cap():
    from flock.worker.agents import transient_backoff
    top = [transient_backoff(n, rng=lambda: 1.0) for n in range(1, 10)]
    assert top == [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 120.0, 120.0, 120.0]
    assert [transient_backoff(n, rng=lambda: 0.0) for n in (1, 3, 50)] == [1.0, 4.0, 60.0]


async def test_consecutive_transport_failures_back_off_then_count_as_errors(tmp_path, monkeypatch):
    """A dead endpoint: retries back off (growing delays), and after LOOP_TRANSIENT_LIMIT in
    a row each further failure counts toward LOOP_ERROR_LIMIT, so the loop ends."""
    delays: list[float] = []
    real = agents_mod.transient_backoff

    def spy(n, rng=None):
        delays.append(real(n, rng=lambda: 1.0))
        return 0.001 * n

    monkeypatch.setattr("flock.worker.agents.LOOP_ERROR_BACKOFF", 0.01)
    monkeypatch.setattr("flock.worker.agents.LOOP_TRANSIENT_LIMIT", 4)
    monkeypatch.setattr("flock.worker.agents.transient_backoff", spy)
    rt, ad = runtime(tmp_path, idle_grace="5s", adapter=TransportFailureAdapter(tmp_path / "h"))
    await rt.setup(restore=False)
    rt.start_loop(start())
    await until(lambda: custom(rt, "loop_stopped"), timeout=5)
    assert custom(rt, "loop_stopped")[0]["reason"] == "errors"
    assert delays == [2.0, 4.0, 8.0, 16.0]  # 4 backed-off retries, then 3 counted errors
    assert len(ad.sent) == 4 + 3
    await rt.stop()
