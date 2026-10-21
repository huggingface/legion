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
  `max_rounds`, the deadline, a `loop_stop` or 3 failed turns in a row.
- `controller`: a duration `every`, or `turn_end` under `max_working` (a worker-side loop
  would keep its slot from turn to turn and starve the agents queued behind it). The controller
  prompts on `turn_ended` events or when `every` has passed since the agent's last loop prompt.

`first` goes alone, and the loop prompt only after the turn that followed it ended. Wake-ups
(`wake(agent, reason)`, the seam for `on: comms.*`) queue the loop prompt after a 2 s debounce.

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

from flock.controller.reconcile import agent_state
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
    parse_duration,
)

log = logging.getLogger("flock.loop")

WAKE_DEBOUNCE = 2.0  # s of quiet after the last wake before the prompt goes out
WAKE_MAX_DELAY = 10.0  # s, a wake is never held longer than this by a stream of new wakes
STALL_AFTER = 30.0  # s idle in a worker-side loop with nothing in flight: send loop_start again
ACK_TIMEOUT = 60.0  # s to wait for a worker's loop_stopped before the final prompt goes anyway
ERROR_LIMIT = 3  # failed loop turns in a row end an agent's loop (worker and controller side)
LOOP_KINDS = ("first", "loop")


def swarm_stream(swarm: str) -> str:
    """Event stream key for swarm-level events (agent names never contain a dot)."""
    return f"_swarm.{swarm}"


def _ts(t: float) -> datetime:
    return datetime.fromtimestamp(t, UTC)


def _spec_hash(lp: LoopSpec) -> str:
    return hashlib.sha256(lp.model_dump_json().encode()).hexdigest()[:12]


def _custom(ev: Event) -> dict[str, Any]:
    """A worker custom event's own payload (pi `custom` entry: data.data)."""
    inner = ev.data.get("data") if isinstance(ev.data, dict) else None
    return inner if isinstance(inner, dict) else {}


def _is_loop_body(body: str) -> bool:
    try:
        meta = json.loads(body).get("loop") or {}
    except ValueError:
        return False
    return meta.get("kind") in LOOP_KINDS


def _new_agent() -> dict[str, Any]:
    return {"phase": "new", "rounds": 0, "last_sent": None, "first_delivered": False,
            "wstarted": None, "stop_sent": None, "acked": False, "last_activity": None,
            "turn_done": False, "idle_streak": 0, "errors": 0, "done_reason": None,
            "final": None, "final_sent": None}


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
        self._wakes: dict[str, tuple[float, float, list[str]]] = {}  # agent -> first, last, why

    # ------------------------------------------------------------------ events out
    def _event(self, swarm: str, kind: str, data: dict[str, Any], agent: str | None = None) -> None:
        now = _ts(self.clock().timestamp())
        self.store.append_events(swarm, swarm_stream(swarm),
                                 [Event(type=kind, ts=now, agent=agent or "", data=data)])
        if agent:
            self.store.append_events(swarm, agent, [Event(type=kind, ts=now, agent=agent, data=data)])

    # ------------------------------------------------------------------ public seams
    def wake(self, agent: str, reason: str) -> bool:
        """Queue the agent's loop prompt (debounced WAKE_DEBOUNCE s, skip_pending), e.g. for a
        comms mention. Only for a running loop whose `on:` lists `reason` (or its prefix
        before ':'); returns False when the wake was ignored."""
        with self._lock:
            a = self.store.get_agent(agent)
            sw = self.store.get_swarm(a["swarm"]) if a else None
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
            first, _, why = self._wakes.get(agent, (now, now, []))
            self._wakes[agent] = (first, now, [*why, reason][-10:])
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

    def hard_stopped(self, swarm: str) -> bool:
        return self.store.swarm_flags(swarm).get("budget_stop") == "hard"

    # ------------------------------------------------------------------ status
    def status(self, swarm: str, type_name: str, *, detail: bool = False) -> LoopStatus | None:
        return loop_status(self.store, swarm, type_name, detail=detail)

    # ------------------------------------------------------------------ events in
    def on_events(self, a: Row, events: list[Event], spec: SwarmSpec | None = None) -> None:
        """Turn and loop events from the agent's worker (post_events), in worker order. `spec`
        is the agent's swarm spec when the caller has it at hand."""
        if not any(ev.type in ("loop_prompt", "loop_stopped", "turn_ended", "turn_interrupted")
                   for ev in events):
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
                    if d.get("kind") == "first":
                        st["first_delivered"] = True
                    elif d.get("kind") == "final":
                        st["final"] = "delivered"
                    elif isinstance(d.get("round"), int):
                        st["rounds"] = max(st["rounds"], d["round"])
                elif ev.type == "loop_stopped":
                    st["acked"] = True
                    st["wstarted"] = None
                    if d.get("reason") in ("max_rounds", "errors", "deadline") and st["phase"] == "looping":
                        st["phase"], st["done_reason"] = "done", d["reason"]
                elif ev.type in ("turn_ended", "turn_interrupted"):
                    st["last_activity"] = now
                    if st["final"] == "delivered":
                        st["final"] = "done"
                        continue
                    if st["phase"] == "first" and st["first_delivered"]:
                        st["phase"] = "looping"
                        continue
                    if st["phase"] != "looping":
                        continue
                    status = d.get("status") if ev.type == "turn_ended" else "interrupted"
                    w = ts.loop.while_
                    if w.idle_rounds is not None and ev.type == "turn_ended":
                        st["idle_streak"] = 0 if d.get("tool_calls", 1) else st["idle_streak"] + 1
                        if st["idle_streak"] >= w.idle_rounds:
                            stop_reason = "idle_rounds"
                    if L.get("mode") == "controller":
                        st["errors"] = 0 if status == "completed" else st["errors"] + 1
                        if st["errors"] >= ERROR_LIMIT:
                            stop_reason = "errors"
                        st["turn_done"] = True
            if stop_reason and st["phase"] == "looping":
                self._agent_done(a, st, L, stop_reason, now)
            self.store.put_loop_agent(a["name"], a["swarm"], a["type"], st)

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
            elif L.get("hash") != _spec_hash(ts.loop):
                self._updated(spec, type_name, ts, L, now)
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
                self.store.put_swarm_flags(spec.swarm, {})
                self._event(spec.swarm, "budget_cleared", {"cost_usd": round(cost, 6),
                                                           "budget_usd": budget})
                phase = None
        if budget is None:
            return phase
        if cost >= budget and phase != "hard":
            self._event(spec.swarm, "budget_stop", {"phase": "hard", "cost_usd": round(cost, 6),
                                                    "budget_usd": budget})
            self.store.put_swarm_flags(spec.swarm, {"budget_stop": "hard", "budget_usd": budget,
                                                    "at": now})
            for type_name, L in loops.items():
                ts = spec.types.get(type_name)
                if ts is not None and L["state"] == "running":
                    self._stop(spec.swarm, type_name, ts, L, "budget", final=False, now=now)
                elif L["state"] == "final":
                    self._finish(spec.swarm, type_name, L, now, "budget exhausted")
            dropped = self.store.drop_prompts(spec.swarm)
            stopped = 0
            for a in self.store.list_agents(spec.swarm):
                if a["worker"] and not a["removing"]:
                    self.store.enqueue_command(a["worker"], StopCommand(agent=a["name"]).model_dump_json())
                    stopped += 1
            log.warning("%s: budget $%.2f reached ($%.4f): stopped %d agents, dropped %d prompts",
                        spec.swarm, budget, cost, stopped, dropped)
            return "hard"
        if phase is None and cost >= budget - reserve:
            self._event(spec.swarm, "budget_stop", {
                "phase": "final_reserve", "cost_usd": round(cost, 6), "budget_usd": budget,
                "threshold_usd": round(budget - reserve, 6)})
            self.store.put_swarm_flags(spec.swarm, {"budget_stop": "final_reserve",
                                                    "budget_usd": budget, "at": now})
            dropped = self.store.drop_prompts(spec.swarm)
            for type_name, L in loops.items():
                ts = spec.types.get(type_name)
                if ts is not None and L["state"] == "running":
                    self._stop(spec.swarm, type_name, ts, L, "budget", final=True, now=now)
            log.warning("%s: budget $%.2f minus final reserve $%.2f reached ($%.4f): loops "
                        "stopped, %d prompts dropped", spec.swarm, budget, reserve, cost, dropped)
            return "final_reserve"
        return phase

    # ------------------------------------------------------------------ lifecycle
    def _new_loop(self, spec: SwarmSpec, type_name: str, ts: TypeSpec, now: float) -> dict:
        lp = ts.loop
        assert lp is not None
        mode = "worker" if lp.every == "turn_end" and not self.capped(spec, type_name) else "controller"
        return {"state": "running", "mode": mode, "every": lp.every, "started_at": now,
                "stopped_at": None, "reason": None, "hash": _spec_hash(lp), "final_at": None}

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
        """A re-applied loop: controller-side settings apply from this tick; worker-side loops
        get a fresh loop_start with the new prompt and limits."""
        assert ts.loop is not None
        L["hash"] = _spec_hash(ts.loop)
        self.store.put_loop(spec.swarm, type_name, L)
        self._event(spec.swarm, "loop_updated", {"type": type_name})
        if L["state"] == "running" and L["mode"] == "worker":
            for a in self.store.list_agents(spec.swarm, type_name):
                st = self.store.get_loop_agent(a["name"])
                if st and st["phase"] == "looping" and st["wstarted"]:
                    self._loop_start(spec, type_name, ts, L, a, st, now)
                    self.store.put_loop_agent(a["name"], spec.swarm, type_name, st)

    def _stop(self, swarm: str, type_name: str, ts: TypeSpec, L: dict, reason: str, *,
              final: bool, now: float) -> None:
        L["stopped_at"], L["reason"] = now, reason
        dropped = self.store.drop_prompts(swarm, keep=lambda b: not _is_loop_body(b),
                                          type_name=type_name)
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
        agents = self.store.loop_agents(swarm, type_name).values()
        L["state"] = "done"
        self.store.put_loop(swarm, type_name, L)
        self._event(swarm, "final_done", {
            "type": type_name, "sent": sum(1 for s in agents if s["final"]),
            "done": sum(1 for s in agents if s["final"] == "done"),
            "skipped": sum(1 for s in agents if s["final"] == "skipped"), "note": note})

    def _agent_done(self, a: Row, st: dict, L: dict, reason: str, now: float) -> None:
        st["phase"], st["done_reason"] = "done", reason
        if L.get("mode") == "worker" and st["wstarted"] and not st["acked"] and a["worker"]:
            self.store.enqueue_command(
                a["worker"], LoopStopCommand(agent=a["name"], reason=reason).model_dump_json())
            st["stop_sent"] = now

    # ------------------------------------------------------------------ running
    def _meta(self, ts: TypeSpec, type_name: str, L: dict, kind: str, *, round: int | None,
              why: str) -> LoopMeta:
        assert ts.loop is not None
        until = ts.loop.while_.until
        return LoopMeta(kind=kind, type=type_name, round=round, started_at=_ts(L["started_at"]),
                        wall_s=parse_duration(until).total_seconds() if until else None,
                        cap_usd=agent_cap(ts), why=why,
                        add_tools=list(ts.loop.final.add_tools) if kind == "final" and ts.loop.final else [])

    def _loop_start(self, spec: SwarmSpec, type_name: str, ts: TypeSpec, L: dict, a: Row,
                    st: dict, now: float) -> None:
        lp = ts.loop
        assert lp is not None and lp.prompt is not None
        until = lp.while_.until
        cmd = LoopStartCommand(
            agent=a["name"], text=lp.prompt,
            meta=self._meta(ts, type_name, L, "loop", round=st["rounds"], why="turn_end"),
            max_rounds=lp.while_.max_rounds,
            deadline=_ts(L["started_at"] + parse_duration(until).total_seconds()) if until else None)
        self.store.enqueue_command(a["worker"], cmd.model_dump_json())
        st["wstarted"], st["acked"], st["last_activity"] = now, False, now

    def _type_reason(self, spec: SwarmSpec, type_name: str, ts: TypeSpec, L: dict, now: float,
                     budget: str | None) -> str | None:
        w = ts.loop.while_ if ts.loop else None
        if budget is not None:
            return "budget"
        if w is None:
            return None
        if w.until and now - L["started_at"] >= parse_duration(w.until).total_seconds():
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
        due: dict[str, PromptCommand] = {}
        changed: set[str] = set()
        for i, a in enumerate(sorted(agents, key=lambda r: r["name"])):
            name = a["name"]
            st = states.get(name)
            if st is None:
                st = states[name] = _new_agent()
                changed.add(name)
            if st["phase"] == "done":
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
            if st["phase"] != "looping":
                continue
            woken = self._woken(name, now)
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
                        ts, type_name, L, "loop", round=None, why=woken))
                continue
            if max_rounds is not None and st["rounds"] >= max_rounds:
                if st["turn_done"]:  # its last round's turn ended (a pending one is never dropped)
                    self._agent_done(a, st, L, "max_rounds", now)
                    changed.add(name)
                continue
            why = None
            if woken:
                why = woken
            elif st["rounds"] == 0:
                why = "start"
            elif every is None and st["turn_done"]:
                why = "turn_end"
            elif every is not None and now - (st["last_sent"] or 0.0) >= every:
                why = f"every {lp.every}"
            if why:
                due[name] = self._prompt(a, lp.prompt or "", self._meta(
                    ts, type_name, L, "loop", round=st["rounds"] + 1, why=why))
        sent = set(self.send(spec.swarm, type_name, [a for a in agents if a["name"] in due],
                             lambda a: due[a["name"]])) if due else set()
        for name in sent:
            st, meta = states[name], due[name].loop
            assert meta is not None
            if meta.kind == "first":
                st["phase"] = "first"
            else:
                st["rounds"] = meta.round if meta.round is not None else st["rounds"]
                st["last_sent"], st["turn_done"] = now, False
            st["last_activity"] = now
            self._wakes.pop(name, None)
            changed.add(name)
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

    def _woken(self, name: str, now: float) -> str | None:
        w = self._wakes.get(name)
        if w is None:
            return None
        first, last, why = w
        if now - last >= WAKE_DEBOUNCE or now - first >= WAKE_MAX_DELAY:
            return "wake: " + ", ".join(dict.fromkeys(why))
        return None

    def _prompt(self, a: Row, text: str, meta: LoopMeta) -> PromptCommand:
        return PromptCommand(agent=a["name"], prompt_id=uuid.uuid4().hex, text=text, loop=meta)

    # ------------------------------------------------------------------ final round
    def _final(self, spec: SwarmSpec, type_name: str, ts: TypeSpec, L: dict, now: float) -> None:
        lp = ts.loop
        assert lp is not None and lp.final is not None
        agents = [a for a in self.store.list_agents(spec.swarm, type_name) if not a["removing"]]
        states = self.store.loop_agents(spec.swarm, type_name)
        workers = {w["id"]: w for w in self.store.list_workers(spec.swarm, type_name)}
        due: dict[str, PromptCommand] = {}
        changed: set[str] = set()
        for a in agents:
            name = a["name"]
            st = states.setdefault(name, _new_agent())
            if st["final"] is not None:
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
            states[name]["final"], states[name]["final_sent"] = "sent", now
            changed.add(name)
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
        per = {name: {k: st[k] for k in ("phase", "rounds", "done_reason", "final", "errors",
                                         "idle_streak")} for name, st in sorted(agents.items())}
    return LoopStatus(
        type=type_name, state=L["state"], mode=L["mode"], every=L["every"],
        started_at=_ts(L["started_at"]) if L.get("started_at") else None,
        stopped_at=_ts(L["stopped_at"]) if L.get("stopped_at") else None,
        reason=L.get("reason"), agents=len(agents),
        agents_done=sum(1 for st in agents.values() if st["phase"] == "done"),
        rounds_min=min(rounds, default=0), rounds_max=max(rounds, default=0),
        rounds_total=sum(rounds),
        final_sent=sum(1 for st in agents.values() if st["final"] in ("sent", "delivered", "done")),
        final_done=sum(1 for st in agents.values() if st["final"] == "done"),
        per_agent=per)
