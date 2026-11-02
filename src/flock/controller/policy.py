"""Policies and the periodic digest (docs/policies.md).

`Policies.evaluate(spec, body, now)` runs on every health tick with the swarm's new snapshot
body. Each rule reads one metric path (`worker.*` per worker, the rest once per swarm) and
keeps per (rule, target) state: consecutive fresh breaching samples (a worker's sample is fresh
when its heartbeat moved since the last tick, the world's when it was probed this tick), firing
or not, acted or not, the last enforced episode. Firing emits `alert` and runs `do`; clearing
(`clear` threshold, hysteresis) emits `alert_cleared` and runs `clear_do`.

Actions run for real only in `enforce` mode and only those in ENFORCED (run 5: drains through
the rotation budget, type holds, world knobs and lane pauses); every other one is logged as
`would_do`. `notify` always runs. An enforced episode waits for its rule's cooldown,
`max_concurrent`, `max_per_hour` and, for `drain_worker`, a rotation slot (`max_rotating`),
and is tried again on later ticks while the rule fires. Past FLOCK_POLICY_MAX_ACTIONS_PER_HOUR
enforced episodes per swarm, rules drop to alert-only (`policy_budget_exhausted`).

State is in memory: a controller restart starts every rule over (alerts fire again after
`for`; drains still respect the rotation budget). Mutes are in the store.
"""

from __future__ import annotations

import json
import logging
import math
import operator
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from flock.controller.reconcile import (
    ROTATE_LOCK,
    rotatable,
    rotate_worker,
    rotation_room,
    swarm_stream,
)
from flock.controller.store import Store
from flock.models import Event, PolicyRule, SwarmSpec, parse_duration, world_tools

log = logging.getLogger("flock.policy")

MAX_ACTIONS_PER_HOUR = 20  # per swarm (FLOCK_POLICY_MAX_ACTIONS_PER_HOUR)
ENFORCED = frozenset({"drain_worker", "pause_type", "resume_type", "world_knob", "world_pause",
                      "world_resume"})
DIGEST_MAX = 4096
OPS: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
    "==": operator.eq, "!=": operator.ne}

# Without a `policies:` block (alert mode). World down is handled by the pause holds
# (`world.down_pause_s`), so it has no default rule here.
DEFAULT_RULES = [PolicyRule.model_validate(
    {"name": "worker-mem", "metric": "worker.nonreclaim_frac", "op": ">", "value": 0.9,
     "for": "2m", "clear": 0.8, "do": "drain_worker", "cooldown": "30m"})]


def rules_of(spec: SwarmSpec) -> tuple[str, list[PolicyRule]]:
    """(block mode, rules): the spec's, or the built-in ones in alert mode."""
    if spec.policies is None:
        return "alert", DEFAULT_RULES
    return spec.policies.mode, spec.policies.rules


def metric_values(metric: str, body: Mapping[str, Any]) -> dict[str, float | None]:
    """target -> value of `metric` in a snapshot body: worker ids for `worker.*`, else
    "swarm" (or "world")."""
    head, _, rest = metric.partition(".")
    if head == "worker":
        out = {}
        for wid, w in (body.get("workers") or {}).items():
            mem = w.get("mem") or {}
            if rest == "nonreclaim_frac":
                n, lim = mem.get("nonreclaim_mb"), mem.get("limit_mb")
                out[wid] = n / lim if n is not None and lim else None
            else:
                out[wid] = w.get(rest)
        return out
    if head == "world":
        w = body.get("world")
        if w is None:
            return {}
        v: Any = (float(bool(w.get("up"))) if rest == "up" else w.get("manifest_ms")
                  if rest == "manifest_ms" else (w.get("metrics") or {}).get(
                      rest.removeprefix("metrics.")))
        return {"world": v if isinstance(v, (int, float)) else None}
    name, _, key = rest.rpartition(".")
    node = (body.get(head) or {}).get(name)
    if node is None:
        return {}
    if head == "types":
        v = len(node.get("stuck") or []) if key == "stuck_n" else (node.get("by_state") or {}).get(key, 0)
    else:
        v = node.get(key)
    return {"swarm": v if isinstance(v, (int, float)) else None}


def _ok_clear(rule: PolicyRule, v: float) -> bool:
    """The firing rule clears: `op value` no longer holds, or with `clear` set, `op clear`."""
    return not OPS[rule.op](v, rule.value if rule.clear is None else rule.clear)


class Policies:
    def __init__(self, store: Store, *, secrets: Mapping[str, str],
                 transport: httpx.BaseTransport | None = None) -> None:
        self.store, self.transport = store, transport
        self.max_actions = int(secrets.get("FLOCK_POLICY_MAX_ACTIONS_PER_HOUR",
                                           MAX_ACTIONS_PER_HOUR))
        raw = secrets.get("FLOCK_MAX_ROTATING")
        self.max_rotating = int(raw) if raw else None
        self.loops: Any = None  # the LoopEngine: `pause` / `resume` holds (docs/loop.md)
        self.world_admin: Callable[..., Any] | None = None  # (swarm, method, path, body)
        self.state: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._acts: dict[tuple[str, str], deque] = {}  # (swarm, rule) -> enforced episode times
        self._budget: dict[str, deque] = {}  # swarm -> enforced episode times
        self._exhausted: dict[str, float] = {}  # swarm -> last policy_budget_exhausted
        self._hb: dict[tuple[str, str], Any] = {}  # (swarm, worker) -> heartbeat last seen
        self._probe: dict[str, Any] = {}  # swarm -> world probe time last seen
        self.recent: dict[str, deque] = {}  # swarm -> (ts, policy event) for the digest
        self._warned: set[tuple[str, str]] = set()  # (swarm, rule) with no value, warned

    # ------------------------------------------------------------------ mutes
    def mutes(self, swarm: str, now: float) -> dict[str, float | None]:
        """rule -> muted until (None: until unmuted); expired ones dropped."""
        m = self.store.get_control(swarm, "policy_mutes") or {}
        return {r: u for r, u in m.items() if u is None or u > now}

    def mute(self, swarm: str, rule: str, until: float | None) -> None:
        m = self.store.get_control(swarm, "policy_mutes") or {}
        self.store.put_control(swarm, "policy_mutes", {**m, rule: until})

    def unmute(self, swarm: str, rule: str) -> None:
        m = self.store.get_control(swarm, "policy_mutes") or {}
        m.pop(rule, None)
        self.store.put_control(swarm, "policy_mutes", m)

    def view(self, spec: SwarmSpec, now: float) -> dict[str, Any]:
        """`GET /swarms/{s}/policies`."""
        mode, rules = rules_of(spec)
        mutes = self.mutes(spec.swarm, now)
        used = [t for t in self._budget.get(spec.swarm, ()) if t > now - 3600]
        return {"swarm": spec.swarm, "mode": mode, "builtin": spec.policies is None,
                "actions_last_hour": len(used), "max_actions_per_hour": self.max_actions,
                "rules": [{**r.model_dump(mode="json", by_alias=True),
                           "effective_mode": r.mode or mode, "muted": r.name in mutes,
                           "muted_until": mutes.get(r.name),
                           "firing": sorted(t for (s, n, t), st in self.state.items()
                                            if s == spec.swarm and n == r.name and st["firing"])}
                          for r in rules]}

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, spec: SwarmSpec, body: Mapping[str, Any], now: float,
                 interval: float) -> list[tuple[str, dict[str, Any]]]:
        """One tick: update every rule, emit and act. Returns the events written."""
        swarm, out = spec.swarm, []
        mode, rules = rules_of(spec)
        mutes = self.mutes(swarm, now)
        fresh = {}
        for wid, w in (body.get("workers") or {}).items():
            fresh[wid] = w.get("hb") is not None and w.get("hb") != self._hb.get((swarm, wid))
        world = body.get("world") or {}
        fresh["world"] = world.get("ts") is not None and world.get("ts") != self._probe.get(swarm)
        names = {r.name for r in rules}
        for key in [k for k in self.state if k[0] == swarm and k[1] not in names]:
            del self.state[key]  # a rule removed by an apply
        holds = self._policy_holds(swarm)
        for reason in sorted(holds - names):  # its rule was removed: nothing would lift it
            out += self._lift(spec, reason, now)
        for rule in rules:
            healthy = False  # a fresh sample under the threshold, this tick
            need = max(1, math.ceil(parse_duration(rule.for_).total_seconds() / interval))
            values = metric_values(rule.metric, body)
            if not any(v is not None for v in values.values()):
                if (swarm, rule.name) not in self._warned:  # e.g. a tool no worker called yet
                    self._warned.add((swarm, rule.name))
                    log.warning("%s: policy %s: no value for %s", swarm, rule.name, rule.metric)
                    self._emit(swarm, now, "policy_warning", {
                        "rule": rule.name, "msg": f"no value for {rule.metric} yet"})
            else:
                self._warned.discard((swarm, rule.name))
            for key in [k for k in self.state if k[:2] == (swarm, rule.name)
                        and k[2] not in values]:
                st = self.state.pop(key)  # the worker is gone
                if st["firing"]:
                    out.append(self._emit(swarm, now, "alert_cleared", {
                        "rule": rule.name, "target": key[2], "value": None, "gone": True}))
            for target, v in values.items():
                st = self.state.setdefault((swarm, rule.name, target), {
                    "n": 0, "firing": False, "acted": False, "last_act": None})
                if v is None or not fresh.get(target, True):
                    continue  # no sample, or a stale one: neither counts nor resets
                muted = rule.name in mutes
                if not st["firing"]:
                    st["n"] = st["n"] + 1 if OPS[rule.op](v, rule.value) else 0
                    healthy |= st["n"] == 0
                    if st["n"] < need:
                        continue
                    st.update(firing=True, acted=False, enforced=False)
                    if not muted:
                        out.append(self._emit(swarm, now, "alert", {
                            "rule": rule.name, "target": target, "value": v}))
                elif _ok_clear(rule, v):
                    st.update(firing=False, n=0)
                    if not muted:
                        out.append(self._emit(swarm, now, "alert_cleared", {
                            "rule": rule.name, "target": target, "value": v}))
                    if st["acted"]:  # muted too: undo what the episode did
                        out += self._run(spec, rule, (rule.mode or mode) if st.get(
                            "enforced") else "alert", target, rule.clear_do, now,
                            clearing=True)
                    continue
                st["value"] = v
                if st["firing"] and not st["acted"] and not muted:
                    out += self._act(spec, rule, mode, target, st, now)
            # a hold of this rule with nothing firing (after a restart, a cleared episode whose
            # clear_do failed, ...): lift it
            if healthy and rule.name in holds and not any(
                    x["firing"] for k, x in self.state.items() if k[:2] == (swarm, rule.name)):
                out += self._lift(spec, rule.name, now)
        for wid, w in (body.get("workers") or {}).items():
            self._hb[(swarm, wid)] = w.get("hb")
        self._probe[swarm] = world.get("ts")
        return out

    def _policy_holds(self, swarm: str) -> set[str]:
        """Rule names that hold a type or agent now (automatic holds made by a policy)."""
        h = self.store.swarm_flags(swarm).get("holds") or {}
        return {e.get("reason") for k in ("types", "agents") for e in (h.get(k) or {}).values()
                if e.get("auto") and e.get("by") == "policy"}

    def _lift(self, spec: SwarmSpec, rule: str, now: float) -> list[tuple[str, dict[str, Any]]]:
        if getattr(self.loops, "resume", None) is None:
            return []
        self.loops.resume(spec.swarm, by="policy", auto_only=True, reason=rule)
        return [self._emit(spec.swarm, now, "policy_action", {
            "rule": rule, "target": "swarm", "action": "resume_type", "mode": "done",
            "orphan": True})]

    def _act(self, spec: SwarmSpec, rule: PolicyRule, mode: str, target: str,
             st: dict[str, Any], now: float) -> list[tuple[str, dict[str, Any]]]:
        """An episode's `do`, once: now, or on a later tick while a cap holds it back."""
        swarm = spec.swarm
        enforce = (rule.mode or mode) == "enforce" and any(
            next(iter(a)) in ENFORCED for a in rule.do)
        if enforce:
            cool = parse_duration(rule.cooldown).total_seconds()
            if st["last_act"] is not None and now - st["last_act"] < cool:
                return []
            if rule.max_concurrent is not None and sum(
                    1 for k, x in self.state.items()
                    if k[:2] == (swarm, rule.name) and x["firing"] and x.get("enforced")
            ) >= rule.max_concurrent:
                return []
            acts = self._acts.setdefault((swarm, rule.name), deque())
            while acts and acts[0] <= now - 3600:
                acts.popleft()
            if rule.max_per_hour is not None and len(acts) >= rule.max_per_hour:
                return []
            budget = self._budget.setdefault(swarm, deque())
            while budget and budget[0] <= now - 3600:
                budget.popleft()
            if len(budget) >= self.max_actions:
                enforce = False  # alert-only until the hour frees a slot
                out = []
                if swarm not in self._exhausted or now - self._exhausted[swarm] >= 3600:
                    self._exhausted[swarm] = now
                    out.append(self._emit(swarm, now, "policy_budget_exhausted", {
                        "rule": rule.name, "max_actions_per_hour": self.max_actions}))
                st["acted"] = True
                return out + self._run(spec, rule, "alert", target, rule.do, now)
            if any(next(iter(a)) == "drain_worker" for a in rule.do):
                with ROTATE_LOCK:  # the room check and the drain, against the reconciler's
                    if rotation_room(self.store, swarm, self.max_rotating, now) <= 0:
                        return []  # no rotation slot: wait for one
                    return self._enforce(spec, rule, mode, target, st, now, acts, budget)
            return self._enforce(spec, rule, mode, target, st, now, acts, budget)
        st["acted"] = True
        return self._run(spec, rule, rule.mode or mode, target, rule.do, now)

    def _enforce(self, spec, rule, mode, target, st, now, acts, budget):
        acts.append(now)
        budget.append(now)
        st["last_act"], st["enforced"], st["acted"] = now, True, True
        return self._run(spec, rule, rule.mode or mode, target, rule.do, now)

    def _run(self, spec: SwarmSpec, rule: PolicyRule, mode: str, target: str,
             actions: list[dict[str, Any]], now: float, clearing: bool = False
             ) -> list[tuple[str, dict[str, Any]]]:
        out = []
        for a in actions:
            (name, arg), = a.items()
            data: dict[str, Any] = {"rule": rule.name, "target": target, "action": name,
                                    **({"arg": arg} if arg is not None else {}),
                                    **({"clear": True} if clearing else {})}
            done = name == "notify" or (mode == "enforce" and name in ENFORCED)
            if done:
                try:
                    detail = self._do(spec, rule, name, arg, target, now)
                    if detail:
                        data.update(detail)
                except Exception as e:  # noqa: BLE001 - one failed action never stops the tick
                    log.warning("policy %s %s on %s: %s", rule.name, name, target, e)
                    data["error"] = f"{type(e).__name__}: {e}"[:300]
            data["mode"] = "done" if done and "error" not in data else "would_do"
            out.append(self._emit(spec.swarm, now, "policy_action", data))
        return out

    def _do(self, spec: SwarmSpec, rule: PolicyRule, name: str, arg: Any, target: str,
            now: float) -> dict[str, Any] | None:
        swarm = spec.swarm
        if name == "notify":
            st = self.state.get((swarm, rule.name, target)) or {}
            self.notify(spec, {"rule": rule.name, "target": target, "metric": rule.metric,
                               "value": st.get("value"), "firing": st.get("firing"),
                               "ts": datetime.fromtimestamp(now, UTC).isoformat()})
            return None
        if name == "drain_worker":
            w = self.store.get_worker(target)
            if w is None or w["swarm"] != swarm:
                raise ValueError(f"drain_worker needs a worker metric (target {target!r})")
            if not rotatable(w, now):
                return {"skipped": "not running or already draining"}
            rotate_worker(self.store, swarm, w, now, rule=rule.name)
            return None
        if name in ("pause_type", "resume_type"):
            types = arg if isinstance(arg, list) else [arg] if isinstance(arg, str) else [
                t for t, ts in spec.types.items() if world_tools(ts, spec.world)]
            fn = getattr(self.loops, "pause" if name == "pause_type" else "resume", None)
            if fn is None:
                raise RuntimeError("no hold primitive on this controller")
            # auto holds: a policy never lifts an operator's hold
            if name == "pause_type":
                fn(swarm, types=types, reason=rule.name, by="policy", auto=True)
            else:
                fn(swarm, types=types, by="policy", auto_only=True, reason=rule.name)
            return {"types": types}
        if self.world_admin is None:
            raise RuntimeError("no world admin route")
        if name == "world_knob":
            res = self.world_admin(swarm, "PATCH", "/admin/knobs", arg)
            for c in res.get("changes") or []:
                self._emit(swarm, now, "world_knob", {**c, "by": "policy", "reason": rule.name})
            return None
        lanes = arg.get("lanes") if isinstance(arg, dict) else arg
        verb = name.removeprefix("world_")
        self.world_admin(swarm, "POST", f"/admin/{verb}", {
            "lanes": lanes, **({"reason": f"policy {rule.name}"} if verb == "pause" else {})})
        self._emit(swarm, now, name, {"lanes": lanes, "by": "policy", "reason": rule.name})
        return None

    # ------------------------------------------------------------------ sinks
    def notify(self, spec: SwarmSpec, doc: dict[str, Any]) -> None:
        send(self.store, spec.policies.sinks if spec.policies else [], doc,
             json.dumps(doc, sort_keys=True), self.transport)

    def _emit(self, swarm: str, now: float, kind: str, data: dict[str, Any]
              ) -> tuple[str, dict[str, Any]]:
        log.info("%s: %s %s", swarm, kind, data)
        self.store.append_events(swarm, swarm_stream(swarm), [Event(
            type=kind, ts=datetime.fromtimestamp(now, UTC), agent="", data=data)])
        if kind != "alert_cleared":
            self.recent.setdefault(swarm, deque(maxlen=200)).append((now, kind, data))
        return kind, data


_SINKS = ThreadPoolExecutor(2, thread_name_prefix="sink")


def _post(sink: str, url: str, doc: dict[str, Any], transport: httpx.BaseTransport | None) -> None:
    try:
        with httpx.Client(transport=transport, timeout=5) as http:
            http.post(url, json=doc).raise_for_status()
    except Exception as e:  # noqa: BLE001 - a sink never stops the controller
        log.warning("sink %s: %s: %s", sink, type(e).__name__, e)


def send(store: Store, sinks: list[str], doc: dict[str, Any], line: str,
         transport: httpx.BaseTransport | None = None) -> None:
    """`file:<path>` appends `line` under the state dir (mirrored by the state sync);
    `webhook:<url>` POSTs `doc` as JSON from a worker thread (5 s timeout). A failing sink is logged and skipped."""
    for s in sinks:
        kind, _, where = s.partition(":")
        try:
            if kind == "file":
                path = store.data_dir / "reports" / where
                if Path(where).is_absolute() or ".." in Path(where).parts:
                    raise ValueError("file sinks stay under the state dir")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(line.rstrip("\n") + "\n")
            else:
                _SINKS.submit(_post, s, where, doc, transport)  # off the health tick
        except Exception as e:  # noqa: BLE001 - a sink never stops the controller
            log.warning("sink %s: %s: %s", s, type(e).__name__, e)


# --------------------------------------------------------------------------- digest


def digest(swarm: str, body: Mapping[str, Any], actions: list[tuple[float, str, dict]],
           now: float, every_s: float) -> str:
    """A short text digest of one snapshot body and the policy events since the last one,
    capped at DIGEST_MAX bytes."""
    stamp = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"flock {swarm} · {stamp}"]
    w = body.get("world")
    if w is not None:
        ms = w.get("manifest_ms")
        lines.append(f"world: {'up' if w.get('up') else 'DOWN'}"
                     + (f", manifest {ms:.0f} ms" if ms is not None else "")
                     + (f", slurm {w['slurm_state']}" if w.get("slurm_state") else ""))
        hourly = {k: v for k, v in (w.get("metrics") or {}).items()
                  if ("last_hour" in k or "per_hour" in k) and isinstance(v, (int, float))}
        if hourly:
            lines.append("world/h: " + ", ".join(f"{k} {v:g}" for k, v in sorted(hourly.items())))
    workers = body.get("workers") or {}
    etas = sorted((x["eta_limit_s"], wid) for wid, x in workers.items()
                  if x.get("eta_limit_s") is not None)
    restoring = sum(1 for x in workers.values() if x.get("restore"))
    lines.append(f"workers: {len(workers)}, restoring {restoring}"
                 + ("; nearest memory limit: " + ", ".join(
                     f"{wid} {eta / 3600:.1f}h" for eta, wid in etas[:5]) if etas else ""))
    for t, x in sorted((body.get("types") or {}).items()):
        states = " ".join(f"{k} {v}" for k, v in sorted((x.get("by_state") or {}).items()))
        stuck = x.get("stuck_n", len(x.get("stuck") or []))
        lines.append(f"{t}: {states}" + (f", stuck {stuck}" if stuck else ""))
    tok = (body.get("tokens") or {}).get("by_type") or {}
    if tok:
        lines.append(f"tokens this hour ({(body.get('tokens') or {}).get('hour')}): " + ", ".join(
            f"{t} in {u.get('input', 0):,} out {u.get('output', 0):,}"
            for t, u in sorted(tok.items())))
    for a in body.get("alerts") or []:
        lines.append(f"alert: {a.get('msg')}")
    acts = [(k, d) for ts, k, d in actions if ts > now - every_s]
    if acts:
        lines.append(f"policy events ({len(acts)}):")
        lines += [f"  {k} {d.get('rule')} {d.get('target', '')} {d.get('action', '')} "
                  f"{d.get('mode', '')}".rstrip() for k, d in acts[-20:]]
    text = "\n".join(lines) + "\n"
    raw = text.encode()
    return text if len(raw) <= DIGEST_MAX else raw[:DIGEST_MAX - 4].decode(errors="ignore") + "...\n"
