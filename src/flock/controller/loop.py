"""The loop engine (docs/loop.md): per-type prompting loops, final rounds and the budget stop.

`LoopEngine.tick()` runs every second next to the reconcile loop. Per swarm it first checks
`limits.budget_usd` (at `budget_usd - sum(final.reserve_usd)`: stop loops, drop pending
prompts, run final rounds; at `budget_usd`: stop every agent), then per type with a `loop:`
evaluates `while` and sends what is due. Every prompt goes through the bulk-prompt path
(`state: idle, skip_pending: true`), so a loop never stacks prompts and every turn is admitted
under `max_working` like any other.

Who sends the loop prompt (`mode`):
- `worker`: `every: turn_end` on a provider without `max_working`. The controller sends one
  `loop_start` per agent; the worker re-sends the prompt each time a turn completes, until
  `max_rounds`, the deadline, a `loop_stop` or 3 failed turns in a row (a turn that hit
  `max_turns` is not a failure; the controller logs it as `loop_max_turns`).
- `controller`: a duration `every`, or `turn_end` under `max_working` (a worker-side loop
  would keep its slot from turn to turn and starve the agents queued behind it). The controller
  prompts on `turn_ended` events or when `every` has passed since the agent's last loop prompt.

`first` goes alone, and the loop prompt only after the turn that followed it ended. Wake-ups
(`wake(agent, reason)`, fed by the workers' comms events for `on: comms.*`) queue the loop
prompt after a 2 s debounce, at most one per `wake.min_gap` per agent and `wake.max` per run
(or `wake.window`); by default a wake prompt is not a round (`wake.counts_as_round`). Each one
sent is a `loop_wake` event.

State lives in the store (`loops`, `loop_agents`, `swarm_flags`), so a controller restart
picks the loops up where they were. Swarm-level events go to the stream `_swarm.<swarm>`
(`GET /swarms/{swarm}/events`); per-agent ones (`final_round`) also to the agent's stream.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from flock.controller.reconcile import agent_state, swarm_stream
from flock.controller.store import Row, Store
from flock.models import (
    AgentState,
    Event,
    LoopMeta,
    LoopSpec,
    LoopStartCommand,
    LoopStatus,
    LoopStopCommand,
    PromptCommand,
    StopCommand,
    SwarmSpec,
    TypeSpec,
    agent_cap,
    hit_max_turns,
    parse_duration,
    until_deadline,
)

log = logging.getLogger("flock.loop")

WAKE_DEBOUNCE = 2.0  # s of quiet after the last wake before the prompt goes out
WAKE_MAX_DELAY = 10.0  # s, a wake is never held longer than this by a stream of new wakes
STALL_AFTER = 30.0  # s idle in a worker-side loop with nothing in flight: send loop_start again
ACK_TIMEOUT = 60.0  # s to wait for a worker's loop_stopped before the final prompt goes anyway
ERROR_LIMIT = 3  # failed loop turns in a row (not max_turns) end an agent's loop (both sides)
RESTOP_EVERY = 30.0  # s, hard budget stop: min gap between repeated stops to one working agent
LOOP_KINDS = ("first", "loop")
NOTICE_KEEP = 100  # pending notices kept per agent (the oldest go first)
NOTICE_GROUPS = 5  # (kind, thread) groups named in the notice line; the rest are counted
NOTICE_NAMES = 3  # names shown per group
NOTICE_HEAD = "New for you since your last round: "
# s after the controller saw the armed loop_start leave the queue before a `loop_prompt` counts
# as the round that carried it: one emitted just before the fetch can arrive up to a worker
# event flush (3 s) later. Erring late shows the line twice, never drops it.
NOTICE_FETCH_MARGIN = 5.0


def _ts(t: float) -> datetime:
    return datetime.fromtimestamp(t, UTC)


def _spec_hash(lp: LoopSpec) -> str:
    return hashlib.sha256(lp.model_dump_json().encode()).hexdigest()[:12]


def _custom(ev: Event) -> dict[str, Any]:
    """A worker custom event's own payload (pi `custom` entry: data.data)."""
    inner = ev.data.get("data") if isinstance(ev.data, dict) else None
    return inner if isinstance(inner, dict) else {}


def _is_loop_body(body: str, kinds: tuple[str, ...] = LOOP_KINDS) -> bool:
    try:
        meta = json.loads(body).get("loop") or {}
    except ValueError:
        return False
    return meta.get("kind") in kinds


def _wake_capped(lp: LoopSpec, st: dict, now: float) -> bool:
    """The agent has had `wake.max` wake prompts in this run (or in the last `wake.window`)."""
    cap = lp.wake.max
    if cap is None:
        return False
    if lp.wake.window is None:
        return st.get("wakes", 0) >= cap
    window = parse_duration(lp.wake.window).total_seconds()
    return sum(1 for t in st.get("wake_times", []) if now - t < window) >= cap


def _is_wake(meta: dict) -> bool:
    """A loop prompt's meta (or a worker's `loop` report) names a wake prompt. A worker older
    than `wake` reports only `counted: false`, which is always a wake."""
    return meta.get("wake") is True or meta.get("counted") is False


def _unwake(st: dict) -> None:
    """A wake prompt was dropped or deferred: not a wake after all (its wake allowance comes
    back; the round, when it counted one, is rolled back by the caller)."""
    st["wakes"] = max(0, st.get("wakes", 0) - 1)
    if st.get("wake_times"):
        st["wake_times"] = st["wake_times"][:-1]
    st["wake_last"] = st["wake_times"][-1] if st.get("wake_times") else None


def _who(n: dict) -> str:
    # no braces (the worker's renderer would take `{round}` in a name for a placeholder)
    name = "".join(c for c in str(n.get("from") or "?") if c not in "{}\n\r")[:64]
    return f"@{name}" if n.get("human") else name


def notice_line(notices: list[dict]) -> str:
    """The one line appended to a round prompt (`loop.notices`): the board notifications since
    the agent's last round, grouped by (kind, thread): mentions by humans first, then other
    mentions, then replies in the agent's threads, each in the order they came; at most
    NOTICE_GROUPS groups and NOTICE_NAMES names per group (humans first), the rest counted."""
    if any(n.get("channel") or n["kind"] in ("dm", "channel") for n in notices):
        return extended_notice_line(notices)
    groups: dict[tuple[str, int], list[dict]] = {}
    for n in notices:
        groups.setdefault((n["kind"], n["thread"]), []).append(n)

    def rank(item: tuple[tuple[str, int], list[dict]]) -> int:
        (kind, _), ns = item
        return (0 if kind == "mention" else 2) + (0 if any(n.get("human") for n in ns) else 1)

    ordered = sorted(groups.items(), key=rank)  # stable: first come first within a rank
    parts = []
    for (kind, thread), ns in ordered[:NOTICE_GROUPS]:
        names = list(dict.fromkeys(_who(n) for n in sorted(ns, key=lambda n: not n.get("human"))))
        shown = ", ".join(names[:NOTICE_NAMES])
        if len(names) > NOTICE_NAMES:
            shown += f", +{len(names) - NOTICE_NAMES} more"
        k = len(ns)
        if kind == "mention":
            parts.append(f"{shown} mentioned you in thread {thread}" if k == 1
                         else f"{k} mentions of you in thread {thread} ({shown})")
        else:
            parts.append(f"{k} {'reply' if k == 1 else 'replies'} in your thread {thread} ({shown})")
    rest = ordered[NOTICE_GROUPS:]
    if rest:
        more = sum(len(ns) for _, ns in rest)
        threads = list(dict.fromkeys(str(t) for (_, t), _ in rest))
        parts.append(f"+{more} more in thread{'s' if len(threads) > 1 else ''} "
                     + ", ".join(threads[:NOTICE_GROUPS]) + (", ..." if len(threads) > NOTICE_GROUPS else ""))
    return NOTICE_HEAD + "; ".join(parts) + ". Read them if useful."


def with_notices(text: str, line: str | None) -> str:
    return f"{text.rstrip()}\n\n{line}" if line else text


def with_banner(text: str, banner: str | None) -> str:
    return f"{banner}\n\n{text}" if banner else text


def _new_agent() -> dict[str, Any]:
    return {"phase": "new", "rounds": 0, "last_sent": None, "first_delivered": False,
            "wstarted": None, "stop_sent": None, "acked": False, "last_activity": None,
            "turn_done": False, "idle_streak": 0, "errors": 0, "done_reason": None,
            "final": None, "final_sent": None,
            "wakes": 0, "wake_last": None, "wake_times": [],  # wake prompts sent (`on:`)
            "inflight": None}  # {kind, round} of a loop prompt the worker took, until its turn starts


def extended_notice_line(notices: list[dict]) -> str:
    groups = {}
    for n in notices:
        groups.setdefault((n["kind"], n.get("channel", ""), n["thread"]), []).append(n)
    parts = []
    ordered = sorted(groups.items(), key=lambda x: {"dm": 0, "mention": 1, "reply": 2}.get(x[0][0], 3))
    for (kind, channel, thread), ns in ordered[:NOTICE_GROUPS]:
        names = list(dict.fromkeys(_who(n) for n in ns))
        shown = ", ".join(names[:NOTICE_NAMES])
        if len(names) > NOTICE_NAMES:
            shown += f", +{len(names) - NOTICE_NAMES} more"
        where = "".join(c for c in str(channel) if c not in "{}\n\r")[:64]
        suffix = f" in #{where}" if where else ""
        if kind == "dm":
            parts.append(f"DM from {shown} (channel {where})")
        elif kind == "mention":
            parts.append(f"{shown} mentioned you{suffix}, thread {thread}")
        elif kind == "channel":
            parts.append(f"new topic{suffix}, thread {thread} ({shown})")
        else:
            parts.append(f"{len(ns)} replies{suffix}, thread {thread} ({shown})")
    if len(ordered) > NOTICE_GROUPS:
        parts.append(f"+{len(ordered) - NOTICE_GROUPS} more groups")
    return NOTICE_HEAD + "; ".join(parts) + ". Read them if useful."


# A prompt sender: (swarm, type, agents, build) -> names of the agents that got a prompt.
# `build(agent_row)` makes the agent's PromptCommand. It is the bulk-prompt path with
# state=idle and skip_pending=true (api.py `prompt_many`).
Sender = Callable[[str, str, list[Row], Callable[[Row], PromptCommand]], list[str]]
# (swarm spec, type) -> True when the type's provider enforces max_working.
Capped = Callable[[SwarmSpec, str], bool]
# (agent row, worker row, now) -> working/idle as the controller knows it (api.py turn_state).
TurnState = Callable[[Row, Row | None, float], AgentState]


class LoopEngine:
    def __init__(self, store: Store, *, send: Sender, clock: Callable[[], datetime],
                 capped: Capped | None = None, turn_state: TurnState | None = None) -> None:
        self.store = store
        self.send = send
        self.clock = clock
        self.capped = capped or (lambda spec, t: False)
        self.turn_state = turn_state or agent_state
        self._lock = threading.RLock()
        # agent -> first, last, why, events: queued wakes not yet sent
        self._wakes: dict[str, tuple[float, float, list[str], int]] = {}
        self._stops: dict[str, float] = {}  # agent -> last StopCommand under the hard budget stop

    # ------------------------------------------------------------------ events out
    def _event(self, swarm: str, kind: str, data: dict[str, Any], agent: str | None = None) -> None:
        now = _ts(self.clock().timestamp())
        self.store.append_events(swarm, swarm_stream(swarm),
                                 [Event(type=kind, ts=now, agent=agent or "", data=data)])
        if agent:
            self.store.append_events(swarm, agent, [Event(type=kind, ts=now, agent=agent, data=data)])

    # ------------------------------------------------------------------ public seams
    def wake(self, agent: str, reason: str) -> bool:
        """Queue the agent's loop prompt (debounced WAKE_DEBOUNCE s, held until `wake.min_gap`
        after its last wake prompt, skip_pending), e.g. for a comms mention. Only for a running
        loop whose `on:` lists `reason` (or its prefix before ':') in a swarm the hard budget
        stop has not stopped, and while the agent is under `wake.max`; returns False when the
        wake was ignored."""
        with self._lock:
            a = self.store.get_agent(agent)
            if a is None or self.hard_stopped(a["swarm"]):
                return False
            sw = self.store.get_swarm(a["swarm"])
            ts = sw["spec"].types.get(a["type"]) if sw else None
            if ts is None or ts.loop is None:
                return False
            if reason not in ts.loop.on and reason.split(":", 1)[0] not in ts.loop.on:
                return False
            L = self.store.get_loops(a["swarm"]).get(a["type"])
            st = self.store.get_loop_agent(agent)
            if not L or L["state"] != "running" or not st or st["phase"] != "looping":
                return False
            now = self.clock().timestamp()
            if _wake_capped(ts.loop, st, now):
                self._wakes.pop(agent, None)
                return False
            first, _, why, n = self._wakes.get(agent, (now, now, [], 0))
            self._wakes[agent] = (first, now, [*why, reason][-10:], n + 1)
            return True

    def notice(self, agent: str, reason: str, data: dict[str, Any]) -> bool:
        """A board notification for the agent (`comms_mention` / `comms_thread_reply` from its
        worker): kept, with the loop agent's state in the store, until its next round prompt
        names it (`loop.notices`). Ignored for a type without `loop.notices`, an agent out of
        its loop and a message already kept. `reason`: mention or thread."""
        with self._lock:
            a = self.store.get_agent(agent)
            if a is None:
                return False
            sw = self.store.get_swarm(a["swarm"])
            ts = sw["spec"].types.get(a["type"]) if sw else None
            if ts is None or ts.loop is None or not ts.loop.notices:
                return False
            st = self.store.get_loop_agent(agent)
            thread, mid = data.get("thread"), data.get("msg_id")
            if st is None or st["phase"] == "done" or not isinstance(thread, int):
                return False
            kind = reason if reason in ("mention", "dm", "channel") else "reply"
            pending = st.get("notices") or []
            if mid is not None and any(n["msg_id"] == mid and n["kind"] == kind for n in pending):
                return False
            sender = str(data.get("from") or "?")
            human = data.get("identity_type") == "user" if "identity_type" in data else (
                sender != "?" and self.store.get_agent(sender) is None)  # not one of ours
            pending.append({"msg_id": mid, "thread": thread, "from": sender, "kind": kind,
                            "human": human})
            if data.get("channel"):
                pending[-1]["channel"] = str(data["channel"])
            st["notices"] = pending[-NOTICE_KEEP:]
            self.store.put_loop_agent(agent, a["swarm"], a["type"], st)
            return True

    def control(self, swarm: str, type_name: str, action: str, final: bool = True) -> LoopStatus:
        """`flock loop --start|--stop`: stop (optionally without the final round) or start
        the loop again from round 0 (agents that had `first` do not get it again)."""
        with self._lock:
            sw = self.store.get_swarm(swarm)
            ts = sw["spec"].types.get(type_name) if sw else None
            if ts is None or ts.loop is None:
                raise KeyError(f"type {type_name!r} in swarm {swarm!r} has no loop")
            L = self.store.get_loops(swarm).get(type_name)
            now = self.clock().timestamp()
            if action == "stop":
                if L is None:
                    L = self._new_loop(sw["spec"], type_name, ts, now)
                if L["state"] == "running":
                    self._stop(swarm, type_name, ts, L, "manual", final=final, now=now)
                elif L["state"] == "final" and not final:
                    self._finish(swarm, type_name, L, now, "final round skipped")
            else:
                if self.store.swarm_flags(swarm).get("budget_stop"):
                    raise ValueError(f"swarm {swarm!r} is stopped by limits.budget_usd; raise it "
                                     "(flock apply) before starting a loop again")
                if L is None or L["state"] in ("final", "done"):
                    self._start(sw["spec"], type_name, ts, now, restart=L is not None)
            return self.status(swarm, type_name, detail=False)

    def _put_budget(self, swarm: str, body: dict[str, Any]) -> None:
        """Write the budget-stop flags, keeping the swarm's pause holds."""
        holds = self.store.swarm_flags(swarm).get("holds")
        self.store.put_swarm_flags(swarm, {**body, **({"holds": holds} if holds else {})})

    # ------------------------------------------------------------------ pause / resume
    def pause(self, swarm: str, types: list[str] | None = None, agents: list[str] | None = None,
              *, interrupt: bool = False, reason: str = "", by: str = "operator",
              auto: bool = False) -> dict[str, Any]:
        """Hold types and/or agents (both None: every type): no new prompt reaches them and
        their loops skip them without counting rounds, until `resume`. `interrupt` also stops
        the turns in flight (`stop`). An automatic hold never replaces an operator's one.
        Returns the swarm's holds."""
        with self._lock:
            sw = self.store.get_swarm(swarm)
            if sw is None:
                raise KeyError(f"no swarm {swarm!r}")
            if types is None and agents is None:
                types = sorted(sw["spec"].types)
            unknown = [t for t in types or [] if t not in sw["spec"].types]
            unknown += [n for n in agents or [] if (a := self.store.get_agent(n)) is None
                        or a["swarm"] != swarm]
            if unknown:
                raise KeyError(f"not in swarm {swarm!r}: {', '.join(unknown)}")
            flags = self.store.swarm_flags(swarm)
            h = flags.setdefault("holds", {"types": {}, "agents": {}})
            entry = {"since": self.clock().timestamp(), "reason": reason, "by": by, "auto": auto}
            new: dict[str, list[str]] = {"types": [], "agents": []}
            for key, names in (("types", types or []), ("agents", agents or [])):
                for n in names:
                    if n not in h[key] or (not auto and h[key][n].get("auto")):
                        h[key][n] = entry
                        new[key].append(n)
            if not new["types"] and not new["agents"]:
                return h
            self.store.put_swarm_flags(swarm, flags)
            self._event(swarm, "paused", {**new, "reason": reason, "by": by, "auto": auto,
                                          "interrupt": interrupt})
            if interrupt:
                workers = {w["id"]: w for w in self.store.list_workers(swarm)}
                loops = self.store.get_loops(swarm)
                now = self.clock().timestamp()
                for a in self.store.list_agents(swarm):
                    if (a["worker"] and not a["removing"] and self.store.held(a, flags)
                            and self.turn_state(a, workers.get(a["worker"]), now)
                            == AgentState.working):
                        # a worker-side loop would re-prompt the stopped turn: end it first
                        st = self.store.get_loop_agent(a["name"])
                        if ((loops.get(a["type"]) or {}).get("mode") == "worker" and st
                                and st["wstarted"] and not st["acked"]):
                            self.store.enqueue_command(a["worker"], LoopStopCommand(
                                agent=a["name"], reason="held").model_dump_json())
                            st["stop_sent"] = now
                            self.store.put_loop_agent(a["name"], swarm, a["type"], st)
                        self.store.enqueue_command(a["worker"],
                                                   StopCommand(agent=a["name"]).model_dump_json())
            return h

    def resume(self, swarm: str, types: list[str] | None = None, agents: list[str] | None = None,
               *, by: str = "operator", auto_only: bool = False,
               reason: str | None = None) -> dict[str, Any]:
        """Lift holds: the named ones, or all (both None). `auto_only`: only the holds the
        controller made itself (world down), never an operator's. `reason`: only
        holds with that reason (each automatic source lifts its own). Queued prompts and wakes
        go out on the next poll / tick. Returns the swarm's holds."""
        with self._lock:
            flags = self.store.swarm_flags(swarm)
            h = flags.get("holds")
            if not h:
                return {"types": {}, "agents": {}}
            gone: dict[str, list[str]] = {"types": [], "agents": []}
            for key, names in (("types", types), ("agents", agents)):
                pick = names if names is not None else (list(h[key]) if types is None
                                                        and agents is None else [])
                for n in pick:
                    if n in h[key] and (not auto_only or h[key][n].get("auto")) and (
                            reason is None or h[key][n].get("reason") == reason):
                        del h[key][n]
                        gone[key].append(n)
            if not gone["types"] and not gone["agents"]:
                return h
            if not h["types"] and not h["agents"]:
                del flags["holds"]
            self.store.put_swarm_flags(swarm, flags)
            self._event(swarm, "resumed", {**gone, "by": by, "auto": auto_only})
            for w in self.store.list_workers(swarm):  # wake parked command polls
                self.store.listener("commands", w["id"])
            return flags.get("holds") or {"types": {}, "agents": {}}

    def hard_stopped(self, swarm: str) -> bool:
        return self.store.swarm_flags(swarm).get("budget_stop") == "hard"

    # ------------------------------------------------------------------ status
    def status(self, swarm: str, type_name: str, *, detail: bool = False) -> LoopStatus | None:
        return loop_status(self.store, swarm, type_name, detail=detail)

    # ------------------------------------------------------------------ events in
    def on_events(self, a: Row, events: list[Event], spec: SwarmSpec | None = None) -> None:
        """Turn and loop events from the agent's worker (post_events), in worker order. `spec`
        is the agent's swarm spec when the caller has it at hand."""
        if not any(ev.type in ("loop_prompt", "loop_stopped", "turn_started", "turn_ended",
                               "turn_interrupted", "prompt_deferred") for ev in events):
            return
        with self._lock:
            st = self.store.get_loop_agent(a["name"])
            if st is None:
                return
            if spec is None:
                sw = self.store.get_swarm(a["swarm"])
                spec = sw["spec"] if sw else None
            ts = spec.types.get(a["type"]) if spec else None
            L = self.store.get_loops(a["swarm"]).get(a["type"])
            if ts is None or ts.loop is None or L is None:
                return
            now = self.clock().timestamp()
            stop_reason = None
            for ev in events:
                d = _custom(ev)
                if ev.type == "loop_prompt":
                    st["last_activity"] = now
                    arm = st.get("notice_armed")
                    if (arm and not arm.get("used") and d.get("kind") == "loop" and not d.get("wake")
                            and self._fetched(a, arm, now)
                            and now - arm["fetched"] >= NOTICE_FETCH_MARGIN):
                        arm["used"] = True  # this round carried the line: plain again next tick
                    st["inflight"] = {"kind": d.get("kind"), "round": d.get("round"),
                                      "counted": d.get("counted", True), "wake": d.get("wake", False)}
                    if d.get("kind") == "first":
                        st["first_delivered"] = True
                    elif d.get("kind") == "final":
                        st["final"] = "delivered"
                    elif isinstance(d.get("round"), int):
                        st["rounds"] = max(st["rounds"], d["round"])
                elif ev.type == "turn_started":
                    st["inflight"] = None
                elif ev.type == "prompt_deferred":
                    self._undeliver(st, d.get("loop") or st.get("inflight"), now)
                elif ev.type == "loop_stopped":
                    st["acked"] = True
                    st["wstarted"] = None
                    if (d.get("reason") in ("max_rounds", "errors", "deadline", "done")
                            and st["phase"] == "looping"):
                        st["phase"], st["done_reason"] = "done", d["reason"]
                elif ev.type in ("turn_ended", "turn_interrupted"):
                    st["last_activity"] = now
                    max_turns = ev.type == "turn_ended" and hit_max_turns(d)
                    if max_turns:  # not an error (below), but the operator sees each one
                        kind = ("final" if st["final"] == "delivered"
                                else "first" if st["phase"] == "first" else "loop")
                        self._event(a["swarm"], "loop_max_turns",
                                    {"type": a["type"], "kind": kind, "round": st["rounds"],
                                     "calls": d.get("num_turns")}, agent=a["name"])
                    if st["final"] == "delivered":
                        st["final"] = "done"
                        continue
                    # `done(finished: true)` under `stop_when.all_agents_done`: out of the loop
                    finished = (ts.loop.stop_when.all_agents_done and ev.type == "turn_ended"
                                and d.get("finished") is True)
                    if st["phase"] == "first" and st["first_delivered"]:
                        st["phase"] = "looping"
                        stop_reason = "done" if finished else stop_reason
                        continue
                    if st["phase"] != "looping":
                        continue
                    held = self.store.held(a)  # a turn cut or ended under a pause hold: not counted
                    status = d.get("status") if ev.type == "turn_ended" else "interrupted"
                    w = ts.loop.while_
                    if w.idle_rounds is not None and ev.type == "turn_ended" and not held:
                        # read-only calls (`done`, comms read/search, `readonly` folder tools)
                        # are not activity; a worker older than `state_calls` sends tool_calls
                        busy = d.get("state_calls", d.get("tool_calls", 1))
                        st["idle_streak"] = 0 if busy else st["idle_streak"] + 1
                        if st["idle_streak"] >= w.idle_rounds:
                            stop_reason = "idle_rounds"
                    if finished:
                        stop_reason = "done"
                    if L.get("mode") == "controller" and held:
                        st["turn_done"] = True
                    elif L.get("mode") == "controller":
                        st["errors"] = (0 if status == "completed" or max_turns else
                                        st["errors"] if d.get("reason") in ("timeout", "model_transport")
                                        else st["errors"] + 1)
                        if st["errors"] >= ERROR_LIMIT:
                            stop_reason = "errors"
                        st["turn_done"] = True
            if stop_reason and st["phase"] == "looping":
                self._agent_done(a, st, L, stop_reason, now)
            self.store.put_loop_agent(a["name"], a["swarm"], a["type"], st)

    def _fetched(self, a: Row, arm: dict, now: float) -> bool:
        """The armed loop_start left its worker's command queue (`arm["fetched"]`: when the
        controller first saw it gone)."""
        if arm.get("fetched") is None:
            for c in self.store.commands_for(a["worker"]) if a["worker"] else []:
                try:
                    body = json.loads(c["body"])
                except ValueError:
                    continue
                if body.get("kind") == "loop_start" and body.get("agent") == a["name"]:
                    return False
            arm["fetched"] = now
        return True

    def _undeliver(self, st: dict, took: dict | None, now: float) -> None:
        """The worker refused a loop prompt it had taken (draining, stopping): undo what its
        `loop_prompt` advanced, so the same prompt goes again on a later tick. Not counted as
        another round."""
        st["inflight"] = None
        if not took:
            return
        kind = took.get("kind")
        if kind == "loop" and _is_wake(took):
            _unwake(st)
        if kind == "first" and st["phase"] == "first":
            st["first_delivered"] = False
            st["last_activity"] = None  # due again at once
        elif kind == "final":
            if st["final"] in ("sent", "delivered"):
                st["final"], st["final_sent"] = None, None
        elif kind == "loop" and took.get("counted") is False:
            st["turn_done"] = True  # no turn runs; the cadence (`last_sent`) is left as it was
        elif kind == "loop" and isinstance(took.get("round"), int):
            st["rounds"] = min(st["rounds"], max(0, took["round"] - 1))
            st["last_sent"], st["turn_done"] = None, True
            if st["phase"] == "done" and st["done_reason"] == "max_rounds":
                st["phase"], st["done_reason"] = "looping", None

    # ------------------------------------------------------------------ the tick
    def tick(self) -> None:
        with self._lock:
            for sw in self.store.list_swarms():
                if sw["removing"]:
                    continue
                try:
                    self._tick_swarm(sw["spec"])
                except Exception:
                    log.exception("loop tick for %s failed", sw["spec"].swarm)

    def _tick_swarm(self, spec: SwarmSpec) -> None:
        now = self.clock().timestamp()
        loops = self.store.get_loops(spec.swarm)
        phase = self._budget(spec, loops, now)
        for type_name in set(loops) - set(spec.types):
            self.store.delete_loop(spec.swarm, type_name)
        for type_name, ts in sorted(spec.types.items()):
            L = loops.get(type_name)
            if ts.loop is None:
                if L is not None:
                    if L["state"] in ("running", "final"):
                        self._event(spec.swarm, "loop_stopped", {"type": type_name, "reason": "loop_removed"})
                        self._stop_workers(spec.swarm, type_name, L, "loop_removed", now)
                    self.store.delete_loop(spec.swarm, type_name)
                continue
            if L is None:
                L = self._start(spec, type_name, ts, now)
            elif (L.get("hash") != _spec_hash(ts.loop)
                  or L.get("mode") != self._mode(spec, type_name, ts.loop)):
                self._updated(spec, type_name, ts, L, now)
            if L["state"] == "final" and ts.loop.final is None:  # final removed by a re-apply
                self._finish(spec.swarm, type_name, L, now, "final removed from the spec")
            if L["state"] == "running":
                self._run(spec, type_name, ts, L, now, budget=phase)
            if L["state"] == "final":
                self._final(spec, type_name, ts, L, now)

    # ------------------------------------------------------------------ budget
    def _budget(self, spec: SwarmSpec, loops: dict[str, dict], now: float) -> str | None:
        flags = self.store.swarm_flags(spec.swarm)
        phase = flags.get("budget_stop")
        budget = spec.limits.budget_usd
        reserve = sum(ts.loop.final.reserve_usd for ts in spec.types.values()
                      if ts.loop is not None and ts.loop.final is not None)
        cost = self.store.swarm_cost(spec.swarm)
        if phase and (budget is None or budget != flags.get("budget_usd")):
            limit = None if budget is None else budget - reserve if phase == "final_reserve" else budget
            if limit is None or cost < limit:  # re-applied with more room: lift the stop
                self._put_budget(spec.swarm, {})
                self._event(spec.swarm, "budget_cleared", {"cost_usd": round(cost, 6),
                                                           "budget_usd": budget})
                phase = None
        if budget is None:
            return phase
        if phase == "hard":
            self._keep_stopped(spec, now)
            return phase
        if cost >= budget and phase != "hard":
            self._event(spec.swarm, "budget_stop", {"phase": "hard", "cost_usd": round(cost, 6),
                                                    "budget_usd": budget})
            self._put_budget(spec.swarm, {"budget_stop": "hard", "budget_usd": budget,
                                                    "at": now})
            for type_name, L in loops.items():
                ts = spec.types.get(type_name)
                if ts is not None and L["state"] == "running":
                    self._stop(spec.swarm, type_name, ts, L, "budget", final=False, now=now)
                elif L["state"] == "final":
                    self._finish(spec.swarm, type_name, L, now, "budget exhausted")
            dropped = self._drop(spec.swarm)
            stopped = 0
            for a in self.store.list_agents(spec.swarm):
                if a["worker"] and not a["removing"]:
                    self.store.enqueue_command(a["worker"], StopCommand(agent=a["name"]).model_dump_json())
                    self._stops[a["name"]] = now
                    stopped += 1
            log.warning("%s: budget $%.2f reached ($%.4f): stopped %d agents, dropped %d prompts",
                        spec.swarm, budget, cost, stopped, dropped)
            return "hard"
        if phase is None and cost >= budget - reserve:
            self._event(spec.swarm, "budget_stop", {
                "phase": "final_reserve", "cost_usd": round(cost, 6), "budget_usd": budget,
                "threshold_usd": round(budget - reserve, 6)})
            self._put_budget(spec.swarm, {"budget_stop": "final_reserve",
                                                    "budget_usd": budget, "at": now})
            dropped = self._drop(spec.swarm)
            for type_name, L in loops.items():
                ts = spec.types.get(type_name)
                if ts is not None and L["state"] == "running":
                    self._stop(spec.swarm, type_name, ts, L, "budget", final=True, now=now)
            log.warning("%s: budget $%.2f minus final reserve $%.2f reached ($%.4f): loops "
                        "stopped, %d prompts dropped", spec.swarm, budget, reserve, cost, dropped)
            return "final_reserve"
        return phase

    def _drop(self, swarm: str, type_name: str | None = None,
              kinds: tuple[str, ...] = LOOP_KINDS) -> int:
        """Drop undelivered prompts: every prompt of the swarm, or one type's loop prompts of
        `kinds`. A dropped loop prompt was never a round (docs/loop.md: a round is one prompt
        to one agent), so its agent's counters go back to before it was queued. Prompts a
        worker already fetched are not pending any more, so they are never dropped twice or
        sent again."""
        out: list[tuple[str, str]] = []
        keep = None if type_name is None else (lambda b: not _is_loop_body(b, kinds))
        n = self.store.drop_prompts(swarm, keep=keep, type_name=type_name, dropped=out)
        for agent, body in out:
            try:
                meta = json.loads(body).get("loop") or {}
            except ValueError:
                continue
            st = self.store.get_loop_agent(agent)
            if st is None or not meta:
                continue
            kind, rnd = meta.get("kind"), meta.get("round")
            if kind == "loop" and _is_wake(meta):
                _unwake(st)
            if kind == "loop" and meta.get("counted") is False:
                st["turn_done"] = True  # a wake outside the round count: the cadence stays
            elif kind == "loop" and isinstance(rnd, int):
                st["rounds"] = min(st["rounds"], max(0, rnd - 1))
                st["last_sent"], st["turn_done"] = None, True
                if st["phase"] == "done" and st["done_reason"] == "max_rounds":
                    st["phase"], st["done_reason"] = "looping", None
            elif kind == "first" and st["phase"] == "first":
                st["phase"], st["first_delivered"], st["last_activity"] = "new", False, None
            elif kind == "final" and st["final"] == "sent":
                st["final"], st["final_sent"] = None, None
            else:
                continue
            a = self.store.get_agent(agent)
            if a is not None:
                self.store.put_loop_agent(agent, a["swarm"], a["type"], st)
        return n

    def _keep_stopped(self, spec: SwarmSpec, now: float) -> None:
        """Hard-stopped: any agent seen working again (a relaunched worker, a stop that never
        arrived) gets `stop` again, at most every RESTOP_EVERY s."""
        workers = {w["id"]: w for w in self.store.list_workers(spec.swarm)}
        for a in self.store.list_agents(spec.swarm):
            if not a["worker"] or a["removing"]:
                continue
            if self.turn_state(a, workers.get(a["worker"]), now) != AgentState.working:
                continue
            if now - self._stops.get(a["name"], -RESTOP_EVERY) < RESTOP_EVERY:
                continue
            self.store.enqueue_command(a["worker"], StopCommand(agent=a["name"]).model_dump_json())
            self._stops[a["name"]] = now
            self._event(spec.swarm, "budget_stop", {"phase": "hard", "restop": a["name"],
                                                    "budget_usd": spec.limits.budget_usd})

    # ------------------------------------------------------------------ lifecycle
    def _mode(self, spec: SwarmSpec, type_name: str, lp: LoopSpec) -> str:
        """Who sends the loop prompt: the worker for `turn_end` on an uncapped provider."""
        return "worker" if lp.every == "turn_end" and not self.capped(spec, type_name) else "controller"

    def _new_loop(self, spec: SwarmSpec, type_name: str, ts: TypeSpec, now: float) -> dict:
        lp = ts.loop
        assert lp is not None
        return {"state": "running", "mode": self._mode(spec, type_name, lp), "every": lp.every,
                "started_at": now, "stopped_at": None, "reason": None, "hash": _spec_hash(lp),
                "final_at": None, "run": uuid.uuid4().hex[:12]}

    def _start(self, spec: SwarmSpec, type_name: str, ts: TypeSpec, now: float,
               restart: bool = False) -> dict:
        L = self._new_loop(spec, type_name, ts, now)
        self.store.put_loop(spec.swarm, type_name, L)
        if restart:  # a new run of the loop: rounds from 0, `first` only for those never sent
            for name, st in self.store.loop_agents(spec.swarm, type_name).items():
                fresh = _new_agent()
                if st["phase"] != "new":
                    fresh["phase"], fresh["first_delivered"] = "looping", True
                self.store.put_loop_agent(name, spec.swarm, type_name, fresh)
                self._wakes.pop(name, None)
        self._event(spec.swarm, "loop_started", {"type": type_name, "mode": L["mode"],
                                                 "every": L["every"], "restart": restart})
        return L

    def _updated(self, spec: SwarmSpec, type_name: str, ts: TypeSpec, L: dict, now: float) -> None:
        """A re-applied loop or provider: controller-side settings apply from this tick. When
        the owner of the cadence changes (`every`, or `max_working` on the provider), the old
        owner stops first: worker loops get `loop_stop` and the controller waits for their
        `loop_stopped` before it prompts; queued controller prompts are dropped before the
        workers take over. Worker loops that stay worker loops get a fresh `loop_start`."""
        assert ts.loop is not None
        old, new = L.get("mode"), self._mode(spec, type_name, ts.loop)
        L["hash"], L["mode"], L["every"] = _spec_hash(ts.loop), new, ts.loop.every
        self.store.put_loop(spec.swarm, type_name, L)
        self._event(spec.swarm, "loop_updated", {"type": type_name, "mode": new,
                                                 "every": ts.loop.every, "previous_mode": old})
        if L["state"] != "running":
            return
        if old == "worker" and new == "controller":
            for a in self.store.list_agents(spec.swarm, type_name):
                st = self.store.get_loop_agent(a["name"])
                if st is None:
                    continue
                if st["wstarted"] and not st["acked"] and a["worker"]:
                    self.store.enqueue_command(a["worker"], LoopStopCommand(
                        agent=a["name"], reason="mode_change").model_dump_json())
                    st["stop_sent"] = now
                st["wstarted"], st["turn_done"], st["last_sent"] = None, True, now
                self.store.put_loop_agent(a["name"], spec.swarm, type_name, st)
        elif old == "controller" and new == "worker":
            self._drop(spec.swarm, type_name)
        elif new == "worker":
            for a in self.store.list_agents(spec.swarm, type_name):
                st = self.store.get_loop_agent(a["name"])
                if st and st["phase"] == "looping" and st["wstarted"]:
                    self._loop_start(spec, type_name, ts, L, a, st, now)
                    self.store.put_loop_agent(a["name"], spec.swarm, type_name, st)

    def _stop(self, swarm: str, type_name: str, ts: TypeSpec, L: dict, reason: str, *,
              final: bool, now: float) -> None:
        L["stopped_at"], L["reason"] = now, reason
        dropped = self._drop(swarm, type_name)
        rounds = [st["rounds"] for st in self.store.loop_agents(swarm, type_name).values()]
        self._event(swarm, "loop_stopped", {"type": type_name, "reason": reason,
                                            "rounds_total": sum(rounds), "dropped": dropped})
        self._stop_workers(swarm, type_name, L, reason, now)
        if final and ts.loop is not None and ts.loop.final is not None:
            L["state"], L["final_at"] = "final", now
        else:
            L["state"] = "done"
        self.store.put_loop(swarm, type_name, L)

    def _stop_workers(self, swarm: str, type_name: str, L: dict, reason: str, now: float) -> None:
        if L.get("mode") != "worker":
            return
        for a in self.store.list_agents(swarm, type_name):
            st = self.store.get_loop_agent(a["name"])
            if st and st["wstarted"] and not st["acked"] and a["worker"]:
                self.store.enqueue_command(
                    a["worker"], LoopStopCommand(agent=a["name"], reason=reason).model_dump_json())
                st["stop_sent"] = now
                self.store.put_loop_agent(a["name"], swarm, type_name, st)

    def _finish(self, swarm: str, type_name: str, L: dict, now: float, note: str = "") -> None:
        """The final round is over (or cut short): its undelivered final prompts are dropped."""
        dropped = self._drop(swarm, type_name, kinds=("final",))
        agents = self.store.loop_agents(swarm, type_name).values()
        L["state"] = "done"
        self.store.put_loop(swarm, type_name, L)
        self._event(swarm, "final_done", {
            "type": type_name, "sent": sum(1 for s in agents if s["final"]),
            "done": sum(1 for s in agents if s["final"] == "done"),
            "skipped": sum(1 for s in agents if s["final"] == "skipped"), "dropped": dropped,
            "note": note})

    def _agent_done(self, a: Row, st: dict, L: dict, reason: str, now: float) -> None:
        st["phase"], st["done_reason"] = "done", reason
        if L.get("mode") == "worker" and st["wstarted"] and not st["acked"] and a["worker"]:
            self.store.enqueue_command(
                a["worker"], LoopStopCommand(agent=a["name"], reason=reason).model_dump_json())
            st["stop_sent"] = now

    # ------------------------------------------------------------------ running
    def _meta(self, ts: TypeSpec, type_name: str, L: dict, kind: str, *, round: int | None,
              why: str, counted: bool = True, wake: bool = False) -> LoopMeta:
        assert ts.loop is not None
        until = ts.loop.while_.until
        return LoopMeta(kind=kind, type=type_name, round=round, started_at=_ts(L["started_at"]),
                        wall_s=max(0.0, until_deadline(until, L["started_at"]) - L["started_at"]) if until else None,
                        cap_usd=agent_cap(ts), why=why, counted=counted, wake=wake,
                        add_tools=list(ts.loop.final.add_tools) if kind == "final" and ts.loop.final else [])

    def _loop_start(self, spec: SwarmSpec, type_name: str, ts: TypeSpec, L: dict, a: Row,
                    st: dict, now: float) -> None:
        lp = ts.loop
        assert lp is not None and lp.prompt is not None
        until = lp.while_.until
        arm = st.get("notice_armed")
        line = arm["line"] if arm and not arm.get("used") and lp.notices else None
        if line:
            arm["fetched"] = None  # a new command to wait for
        st["banner"] = self.banner(spec.swarm, type_name)
        cmd = LoopStartCommand(
            agent=a["name"], text=with_banner(with_notices(lp.prompt, line), st["banner"]),
            meta=self._meta(ts, type_name, L, "loop", round=st["rounds"], why="turn_end"),
            max_rounds=lp.while_.max_rounds, run=L.get("run"),
            deadline=_ts(until_deadline(until, L["started_at"])) if until else None)
        self.store.enqueue_command(a["worker"], cmd.model_dump_json())
        st["wstarted"], st["acked"], st["last_activity"] = now, False, now
        st["stop_sent"] = None  # an earlier stop (a pause hold) is answered by this start

    def _type_reason(self, spec: SwarmSpec, type_name: str, ts: TypeSpec, L: dict, now: float,
                     budget: str | None) -> str | None:
        w = ts.loop.while_ if ts.loop else None
        if budget is not None:
            return "budget"
        if w is None:
            return None
        if w.until and now >= until_deadline(w.until, L["started_at"]):
            return "until"
        if w.budget_usd is not None:
            cost = sum(a["cost"] for a in self.store.list_agents(spec.swarm, type_name))
            if cost >= w.budget_usd:
                return "budget_usd"
        return None

    def _run(self, spec: SwarmSpec, type_name: str, ts: TypeSpec, L: dict, now: float,
             budget: str | None) -> None:
        lp = ts.loop
        assert lp is not None
        if reason := self._type_reason(spec, type_name, ts, L, now, budget):
            self._stop(spec.swarm, type_name, ts, L, reason, final=budget != "hard", now=now)
            return
        agents = [a for a in self.store.list_agents(spec.swarm, type_name) if not a["removing"]]
        states = self.store.loop_agents(spec.swarm, type_name)
        for name in set(states) - {a["name"] for a in agents}:
            self.store.delete_loop_agent(name)
            self._wakes.pop(name, None)
        workers = {w["id"]: w for w in self.store.list_workers(spec.swarm, type_name)}
        stagger = parse_duration(lp.stagger).total_seconds() if lp.stagger else 0.0
        cap = agent_cap(ts)
        max_rounds = lp.while_.max_rounds
        every = lp.every_s
        flags = self.store.swarm_flags(spec.swarm)
        due: dict[str, PromptCommand] = {}
        wakes: dict[str, int] = {}  # agent -> wake events folded into its due prompt
        changed: set[str] = set()
        banner = self.banner(spec.swarm, type_name)
        for i, a in enumerate(sorted(agents, key=lambda r: r["name"])):
            name = a["name"]
            st = states.get(name)
            if st is None:
                st = states[name] = _new_agent()
                changed.add(name)
            if st["phase"] == "done":
                continue
            if self.store.held(a, flags):  # paused: no prompt, no round; wakes stay queued
                if (L["mode"] == "worker" and st["wstarted"] and not st["acked"] and a["worker"]
                        and (st["stop_sent"] or 0) < st["wstarted"]):
                    self.store.enqueue_command(a["worker"], LoopStopCommand(
                        agent=name, reason="held").model_dump_json())
                    st["stop_sent"] = now
                    changed.add(name)
                continue
            if cap is not None and a["cost"] >= cap:
                self._agent_done(a, st, L, "agent_usd", now)
                changed.add(name)
                continue
            if st["phase"] == "new":
                if now < L["started_at"] + i * stagger:
                    continue
                if lp.first is not None:
                    due[name] = self._prompt(a, lp.first, self._meta(ts, type_name, L, "first",
                                                                     round=0, why="first"))
                    continue
                st["phase"], st["first_delivered"] = "looping", True
                changed.add(name)
            if st["phase"] == "first":
                if not st["first_delivered"] and now - (st["last_activity"] or 0.0) >= STALL_AFTER:
                    due[name] = self._prompt(a, lp.first or "", self._meta(  # never taken: again
                        ts, type_name, L, "first", round=0, why="first"))
                continue
            if st["phase"] != "looping":
                continue
            woken = self._woken(name, st, lp, now)
            if L["mode"] == "worker":
                if not a["worker"]:
                    continue
                w = workers.get(a["worker"])
                if not st["wstarted"]:
                    if self.turn_state(a, w, now) == AgentState.idle:
                        self._loop_start(spec, type_name, ts, L, a, st, now)
                        changed.add(name)
                    continue
                idle = self.turn_state(a, w, now) == AgentState.idle
                if idle and now - (st["last_activity"] or now) >= STALL_AFTER:
                    self._loop_start(spec, type_name, ts, L, a, st, now)  # lost, or a new worker
                    changed.add(name)
                elif woken and idle:
                    due[name] = self._prompt(a, lp.prompt or "", self._meta(
                        ts, type_name, L, "loop", round=None, why=woken[0],
                        counted=lp.wake.counts_as_round, wake=True))
                    wakes[name] = woken[1]
                elif (st.get("banner") != banner and not st["acked"]
                      and not (st["stop_sent"] and not st["acked"])):
                    self._loop_start(spec, type_name, ts, L, a, st, now)  # the banner changed
                    changed.add(name)
                else:
                    arm = st.get("notice_armed")
                    if (arm and not arm.get("used") and arm.get("fetched") is None
                            and self._fetched(a, arm, now)):
                        changed.add(name)  # seen leaving the queue: when, for `used`
                    if self._arm_notices(lp, st, waiting=bool(st["stop_sent"] and not st["acked"]),
                                         now=now):
                        self._loop_start(spec, type_name, ts, L, a, st, now)
                        changed.add(name)
                continue
            if st["stop_sent"] and not st["acked"] and now - st["stop_sent"] < ACK_TIMEOUT:
                continue  # its worker-side loop (before a mode change) has not confirmed the stop
            if max_rounds is not None and st["rounds"] >= max_rounds:
                if st["turn_done"]:  # its last round's turn ended (a pending one is never dropped)
                    self._agent_done(a, st, L, "max_rounds", now)
                    changed.add(name)
                continue
            why = None
            if woken:
                why, wakes[name] = woken
            elif st["rounds"] == 0:
                why = "start"
            elif every is None and st["turn_done"]:
                why = "turn_end"
            elif every is None and now - (st["last_activity"] or 0.0) >= STALL_AFTER:
                why = "turn_end"  # its prompt never started a turn (deferred, or lost): again
            elif every is not None and now - (st["last_sent"] or 0.0) >= every:
                why = f"every {lp.every}"
            if why:
                counted = name not in wakes or lp.wake.counts_as_round
                due[name] = self._prompt(a, lp.prompt or "", self._meta(
                    ts, type_name, L, "loop", round=st["rounds"] + int(counted), why=why,
                    counted=counted, wake=name in wakes))
        noticed: dict[str, int] = {}  # agent -> pending notices its due loop prompt names
        for name, cmd in due.items():
            pending = states[name].get("notices") if lp.notices else None
            if pending and cmd.loop is not None and cmd.loop.kind == "loop":
                due[name] = cmd.model_copy(update={"text": with_notices(cmd.text, notice_line(pending))})
                noticed[name] = len(pending)
            elif states[name].get("notices") and not lp.notices:
                states[name]["notices"] = []  # switched off: nothing kept for later
                changed.add(name)
        sent = set(self.send(spec.swarm, type_name, [a for a in agents if a["name"] in due],
                             lambda a: due[a["name"]])) if due else set()
        for name in sent:
            st, meta = states[name], due[name].loop
            assert meta is not None
            if name in noticed:  # named once; what came in meanwhile waits for the next round
                st["notices"] = st["notices"][noticed[name]:]
            if meta.kind == "first":
                st["phase"], st["first_delivered"] = "first", False
            else:
                st["rounds"] = meta.round if meta.round is not None else st["rounds"]
                st["last_sent"], st["turn_done"] = now, False
            st["last_activity"] = now
            self._wakes.pop(name, None)
            changed.add(name)
            if name in wakes:
                st["wakes"] = st.get("wakes", 0) + 1
                st["wake_last"] = now
                st["wake_times"] = [*st.get("wake_times", []), now][-(lp.wake.max or 1):]
                self._event(spec.swarm, "loop_wake", {
                    "type": type_name, "agent": name, "reason": meta.why.removeprefix("wake: "),
                    "events": wakes[name], "coalesced": wakes[name] - 1, "round": meta.round,
                    "counted": meta.counted, "wakes": st["wakes"]}, agent=name)
        for name in changed:
            self.store.put_loop_agent(name, spec.swarm, type_name, states[name])
        if sent:
            self._event(spec.swarm, "loop_tick", {
                "type": type_name, "prompted": len(sent),
                "first": sum(1 for n in sent if due[n].loop.kind == "first"),
                "rounds": {n: due[n].loop.round for n in sorted(sent) if due[n].loop.kind == "loop"}})
        live = [states[a["name"]] for a in agents]
        if live and all(st["phase"] == "done" for st in live):
            reasons = {st["done_reason"] for st in live}
            self._stop(spec.swarm, type_name, ts, L, reasons.pop() if len(reasons) == 1
                       else "agents_done", final=True, now=now)

    def _arm_notices(self, lp: LoopSpec, st: dict, *, waiting: bool, now: float) -> bool:
        """Worker-side loop (`loop.notices`): the worker renders each round from the text of its
        last loop_start, so the line rides in a fresh loop_start (same run: the round count
        stays) and, once a round carried it (its `loop_prompt` after the command was fetched),
        another loop_start puts the plain prompt back, or the line of what came in meanwhile.
        True when a loop_start is due now (`notice_armed` updated)."""
        arm = st.get("notice_armed")
        if not st["wstarted"] or st["acked"] or waiting:
            return False
        if not lp.notices:  # switched off: nothing kept for later
            if st.get("notices"):
                st["notices"] = []
            if arm:
                st["notice_armed"] = None
                return not arm.get("used")  # take the line out of the worker's loop text
            return False
        if arm and not arm.get("used"):
            return False  # its round has not come yet
        pending = st.get("notices") or []
        if pending:
            st["notice_armed"] = {"line": notice_line(pending), "at": now, "n": len(pending),
                                  "fetched": None}
            st["notices"] = []
            return True
        if arm:  # used, nothing new: the plain prompt again
            st["notice_armed"] = None
            return True
        return False

    def _woken(self, name: str, st: dict, lp: LoopSpec, now: float) -> tuple[str, int] | None:
        """The queued wake that is due now, as (why, events): after WAKE_DEBOUNCE s of quiet
        (at most WAKE_MAX_DELAY s after the first), and `wake.min_gap` after the agent's last
        wake prompt; wakes held by the gap are coalesced into this one."""
        w = self._wakes.get(name)
        if w is None:
            return None
        if _wake_capped(lp, st, now):  # e.g. `wake.max` lowered by a re-apply
            self._wakes.pop(name, None)
            return None
        first, last, why, n = w
        if now - last < WAKE_DEBOUNCE and now - first < WAKE_MAX_DELAY:
            return None
        gap = parse_duration(lp.wake.min_gap).total_seconds()
        if st.get("wake_last") is not None and now - st["wake_last"] < gap:
            return None
        return "wake: " + ", ".join(dict.fromkeys(why)), n

    def _prompt(self, a: Row, text: str, meta: LoopMeta) -> PromptCommand:
        return PromptCommand(agent=a["name"], prompt_id=uuid.uuid4().hex, loop=meta,
                             text=with_banner(text, self.banner(a["swarm"], a["type"])))

    def banner(self, swarm: str, type_name: str) -> str | None:
        """`flock notice set` (docs/loop.md "Banner"): the type's line, else the swarm's."""
        n = (self.store.get_control(swarm, f"notice:{type_name}")
             or self.store.get_control(swarm, "notice:*"))
        return n["text"] if n else None

    # ------------------------------------------------------------------ final round
    def _final(self, spec: SwarmSpec, type_name: str, ts: TypeSpec, L: dict, now: float) -> None:
        lp = ts.loop
        if lp is None or lp.final is None:
            self._finish(spec.swarm, type_name, L, now, "final removed from the spec")
            return
        agents = [a for a in self.store.list_agents(spec.swarm, type_name) if not a["removing"]]
        states = self.store.loop_agents(spec.swarm, type_name)
        workers = {w["id"]: w for w in self.store.list_workers(spec.swarm, type_name)}
        due: dict[str, PromptCommand] = {}
        changed: set[str] = set()
        for a in agents:
            name = a["name"]
            st = states.setdefault(name, _new_agent())
            never_taken = st["final"] == "sent" and now - (st["final_sent"] or now) >= STALL_AFTER
            if st["final"] is not None and not never_taken:
                continue
            w = workers.get(a["worker"])
            state = self.turn_state(a, w, now)
            if state == AgentState.dead or not a["worker"]:
                st["final"] = "skipped"
                changed.add(name)
                continue
            if st["stop_sent"] and not st["acked"] and now - st["stop_sent"] < ACK_TIMEOUT:
                continue  # the worker-side loop has not confirmed its stop yet
            if state != AgentState.idle:
                continue
            why = f"final: {L.get('reason') or 'stop'}"
            due[name] = self._prompt(a, lp.final.prompt or "", self._meta(
                ts, type_name, L, "final", round=st["rounds"], why=why))
        sent = self.send(spec.swarm, type_name, [a for a in agents if a["name"] in due],
                         lambda a: due[a["name"]]) if due else []
        for name in sent:
            again = states[name]["final"] == "sent"
            states[name]["final"], states[name]["final_sent"] = "sent", now
            changed.add(name)
            if again:
                continue
            self._event(spec.swarm, "final_round", {"type": type_name, "agent": name,
                                                    "add_tools": lp.final.add_tools}, agent=name)
        for name in changed:
            self.store.put_loop_agent(name, spec.swarm, type_name, states[name])
        live = [states[a["name"]] for a in agents]
        if all(st["final"] in ("done", "skipped") for st in live):
            self._finish(spec.swarm, type_name, L, now)


def loop_status(store: Store, swarm: str, type_name: str, *, detail: bool = False) -> LoopStatus | None:
    L = store.get_loops(swarm).get(type_name)
    if L is None:
        return None
    agents = store.loop_agents(swarm, type_name)
    rounds = [st["rounds"] for st in agents.values()]
    per = None
    if detail:
        per = {name: {**{k: st[k] for k in ("phase", "rounds", "done_reason", "final", "errors",
                                            "idle_streak")}, "wakes": st.get("wakes", 0)}
               for name, st in sorted(agents.items())}
    return LoopStatus(
        type=type_name, state=L["state"], mode=L["mode"], every=L["every"],
        started_at=_ts(L["started_at"]) if L.get("started_at") else None,
        stopped_at=_ts(L["stopped_at"]) if L.get("stopped_at") else None,
        reason=L.get("reason"), agents=len(agents),
        agents_done=sum(1 for st in agents.values() if st["phase"] == "done"),
        rounds_min=min(rounds, default=0), rounds_max=max(rounds, default=0),
        rounds_total=sum(rounds), wakes_total=sum(st.get("wakes", 0) for st in agents.values()),
        final_sent=sum(1 for st in agents.values() if st["final"] in ("sent", "delivered", "done")),
        final_done=sum(1 for st in agents.values() if st["final"] == "done"),
        per_agent=per)
