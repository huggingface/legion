"""Inference servers legion starts itself (docs/open-models.md "Servers legion starts"): an
`inference` block with `hf_jobs` names no URLs; the controller starts one group of servers per
model on it, waits until every replica answers `GET /v1/models`, pins their URLs into the spec
as `urls` would be (`resolved_model.inference.urls`), and stops them when the run ends. Past
startup everything is the `urls` path: the model check, the context window, preflight, Pi.

`Servers` keeps each swarm's groups in memory for the startup wait, pinning and status (a
restarted controller reattaches to the running jobs by their labels). Cleanup is one rule,
applied to the jobs themselves at every sweep: a server job of the swarm that the applied
spec does not use (its group, or a replica beyond the group's count) is cancelled once the
deadline in its `retire` label has passed. A job gets that deadline when it starts (the
launch or update has until then to apply it), shortened once its group is ready, and a new
one when an apply stops using it (the turns of draining workers). A failed group's jobs, and
every job of a stopped swarm, go at once. A cancel that fails is tried again at the next
sweep. A starter (legion.registry SERVERS, one module per block kind, e.g.
controller/hf_jobs_servers.py) does the backend's part.

    servers = Servers(os.environ, applied=lambda: {...})
    servers.start(spec)          # start, or reattach to the running jobs; returns at once
    servers.tick()               # stages, readiness, startup_timeout; the cleanup sweep
    servers.pin(spec)            # once ready: `urls` into each type's resolved block
    servers.retire_unused(spec, grace)  # after an apply: deadlines for what it stops using
    servers.stop(spec.swarm)     # cancel every server job of the swarm
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from legion.controller.inference import InferenceError
from legion.models import ServerStatus, SwarmSpec, TypeSpec, parse_duration
from legion.registry import SERVERS, load

log = logging.getLogger("legion.servers")

TERMINAL = frozenset({"COMPLETED", "CANCELED", "ERROR", "DELETED"})
READY_CHECK_S = 60.0  # a ready group's jobs are checked this often (between: none)
SWEEP_S = 30.0  # each swarm's server jobs are listed this often for the cleanup rule
LOG_LINES = 5  # a failure's message ends with the job's last log lines
# How long a started job waits for its launch or update to apply it: until its group is
# ready (at most startup_timeout), then this margin (preflight, the apply)
APPLY_MARGIN_S = 300.0
GROUP_LABEL, REPLICA_LABEL, RETIRE_LABEL = "inference", "replica", "retire"


@dataclass
class Group:
    """One model's servers on one block's settings: what a starter starts. `block` is the
    block's `hf_jobs` (a server block has `replicas` and `startup_timeout`; with types that
    share the group, the largest of each)."""

    key: str  # the settings' hash: its jobs' `inference` label
    kind: str  # the block key: hf_jobs
    model: str
    block: Any
    max_working: int | None  # the block's: the servers' max_num_seqs
    namespace: str | None  # None: the starter's default (the token's account), set at start
    labels: dict[str, str]  # what its jobs carry and are found by: {<label>: swarm, inference: key}
    types: list[str] = field(default_factory=list)


@dataclass
class Server:
    """One server job: its replica, URL, readiness and labels."""

    replica: int
    job_id: str | None = None
    url: str | None = None  # the OpenAI base URL (`.../v1`)
    stage: str | None = None
    message: str | None = None
    started_at: float | None = None
    ready_at: float | None = None  # first answered GET /v1/models
    group: str | None = None  # its `inference` label
    kind: str = ""  # its starter's (legion.registry SERVERS)
    namespace: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    new: bool = False  # this process started it (not found running)

    @property
    def ended(self) -> bool:
        return self.stage in TERMINAL

    @property
    def retire_at(self) -> float | None:
        """Its `retire` label: until then it is kept when the applied spec does not use it."""
        try:
            return float(self.labels[RETIRE_LABEL])
        except (KeyError, ValueError):
            return None


class StartError(Exception):
    """A start that failed part-way: `servers` are the group's jobs still running (the ones it
    found, and any it started but could not cancel again), for the caller to track."""

    def __init__(self, message: str, servers: list[Server]):
        super().__init__(message)
        self.servers = servers


class Starter(Protocol):
    def start(self, group: Group, retire_at: float) -> list[Server]:
        """The group's servers: the running ones (found by `group.labels`, any replica) as
        they are, and the missing ones of its first `replicas` started with `retire_at` as
        their deadline. StartError when starting one fails (the ones it started are cancelled
        again)."""

    def reattach(self, group: Group) -> list[Server]:
        """The group's running servers, started by an earlier controller; none started."""

    def jobs(self, labels: dict[str, str], namespace: str | None) -> list[Server]:
        """Every running server job carrying `labels` (and a group label: never a worker's)
        in `namespace` (None: the default)."""

    def check(self, group: Group, servers: list[Server]) -> None:
        """Refresh each server's stage, and set `ready_at` once it answers GET /v1/models."""

    def logs(self, group: Group, server: Server, n: int) -> list[str]:
        """The server's last `n` log lines (for a failure's message)."""

    def cancel(self, server: Server) -> None:
        """Cancel the server's job."""

    def retire(self, server: Server, at: float) -> None:
        """Set the job's deadline (its `retire` label) to `at`."""


def server_kind(inference: Any) -> str | None:
    """The block key of the servers legion starts for this block (`hf_jobs`), or None."""
    return next((k for k in SERVERS if getattr(inference, k, None) is not None), None)


def group_of(spec: SwarmSpec, ts: TypeSpec, label: str = "swarm") -> Group | None:
    """The group serving type `ts` on its own block (None: its block names no servers to
    start). Types with the same model and settings share one; the key covers what goes into
    the jobs, not their number or lifetime."""
    r = ts.resolved_model
    kind = server_kind(r.inference) if r is not None else None
    if kind is None:
        return None
    block = getattr(r.inference, kind)
    namespace = spec.namespace or None
    what = [kind, r.model_id, r.inference.max_working, namespace,
            block.model_dump(mode="json", exclude={"startup_timeout", "replicas",
                                                   "timeout"})]
    key = hashlib.sha256(json.dumps(what, sort_keys=True).encode()).hexdigest()[:12]
    return Group(key=key, kind=kind, model=r.model_id, block=block,
                 max_working=r.inference.max_working, namespace=namespace,
                 labels={label: spec.swarm, GROUP_LABEL: key})


def groups(spec: SwarmSpec, label: str = "swarm") -> list[Group]:
    """Each distinct group of the spec's types, with the types it serves, the most replicas
    any of them asks for and the longest startup_timeout."""
    out: dict[str, Group] = {}
    for name, ts in spec.types.items():
        g = group_of(spec, ts, label)
        if g is None:
            continue
        have = out.setdefault(g.key, g)
        have.types.append(name)
        if have is not g:
            startup = max(have.block.startup_timeout, g.block.startup_timeout,
                          key=lambda d: parse_duration(d).total_seconds())
            have.block = have.block.model_copy(update={
                "replicas": max(have.block.replicas, g.block.replicas),
                "startup_timeout": startup})
    return list(out.values())


def drain_grace(spec: SwarmSpec) -> float:
    """How long servers a spec no longer uses keep serving: the types' longest `turn_timeout`,
    the turns their agents' draining workers may still run on them."""
    return max((parse_duration(ts.turn_timeout).total_seconds() for ts in spec.types.values()),
               default=0.0)


@dataclass
class _Run:
    """A group's servers in this process: for the startup wait, pinning and status."""

    group: Group
    servers: list[Server]
    since: float  # the wait's start: `startup_timeout` counts from here
    state: str = "starting"  # starting, ready, failed, stopped
    error: str | None = None
    checked: float = 0.0


def _dt(t: float | None) -> datetime | None:
    return None if t is None else datetime.fromtimestamp(t, UTC)


def _startup_s(g: Group) -> float:
    return parse_duration(g.block.startup_timeout).total_seconds()


class Servers:
    """The servers of each swarm (or, for `legion serve`, each spec: `label="serve"`, no
    sweep). Thread-safe: the API's handlers and the ticker call it. `applied()` gives the
    applied spec of each swarm the controller runs (removing ones aside): the first `tick`
    reattaches to their running servers (until every lookup succeeded), and every sweep
    judges the jobs against them. `memory` (get, put) keeps where each swarm's servers ran
    across restarts."""

    def __init__(self, env: Mapping[str, str], starters: Mapping[str, Starter] | None = None,
                 label: str = "swarm", clock=time.time,
                 applied: Callable[[], dict[str, SwarmSpec]] | None = None,
                 memory: tuple[Callable[[str], Any], Callable[[str, Any], None]] | None = None,
                 sweep_every: float = SWEEP_S):
        self.env, self.label, self.clock, self.applied = env, label, clock, applied
        self.memory, self.sweep_every = memory, sweep_every
        self._starters: dict[str, Starter] = dict(starters or {})
        self.runs: dict[str, dict[str, _Run]] = {}
        self.lock = threading.RLock()
        self._reattached = False
        self._swept: dict[str, float] = {}
        self._stopped: set[str] = set()  # swarms whose every server job goes, now
        self._where: dict[str, set[tuple[str, str | None]]] = {}  # (kind, namespace) seen

    def starter(self, kind: str) -> Starter:
        if kind not in self._starters:
            self._starters[kind] = load(SERVERS[kind])(self.env)
        return self._starters[kind]

    # ------------------------------------------------------------------ where jobs live

    def _seen(self, swarm: str, g: Group) -> None:
        where = self._places(swarm)
        if (g.kind, g.namespace) not in where:
            where.add((g.kind, g.namespace))
            if self.memory is not None:
                try:
                    self.memory[1](swarm, sorted([k, n] for k, n in where))
                except Exception:
                    log.exception("servers of %s: remembering where they run failed", swarm)

    def _places(self, swarm: str) -> set[tuple[str, str | None]]:
        if swarm not in self._where:
            got = None
            if self.memory is not None:
                try:
                    got = self.memory[0](swarm)
                except Exception:
                    log.exception("servers of %s: reading where they ran failed", swarm)
            self._where[swarm] = {(k, n) for k, n in got or []}
        return self._where[swarm]

    def _specs(self) -> dict[str, SwarmSpec]:
        return self.applied() if self.applied is not None else {}

    # ------------------------------------------------------------------ lifecycle

    def start(self, spec: SwarmSpec) -> list[ServerStatus]:
        """Start the spec's groups that are not starting or ready (a failed or stopped one
        starts again), reusing running jobs; returns at once with their status. New jobs get
        the deadline its launch or update has to apply them; a reused job the applied spec does
        not use keeps its deadline, pushed back to that one if earlier. Nothing is cancelled
        here: an update may still be refused."""
        with self.lock:
            self._stopped.discard(spec.swarm)
            runs = self.runs.setdefault(spec.swarm, {})
            now, applied = self.clock(), self._specs().get(spec.swarm)
            for g in groups(spec, self.label):
                run, want = runs.get(g.key), g.block.replicas
                deadline = int(now + _startup_s(g) + APPLY_MARGIN_S)
                if run is not None and run.state in ("starting", "ready") and sum(
                        not s.ended and s.replica < want for s in run.servers) >= want:
                    run.group.types, run.group.block = g.types, g.block
                    self._hold(spec.swarm, applied, run.servers, deadline)
                    continue
                st = self.starter(g.kind)
                try:
                    servers = st.start(g, deadline)
                except StartError as e:
                    log.warning("servers %s for %s: start failed: %s", g.key, g.model, e)
                    servers, state, error = e.servers, "failed", f"start failed: {e}"
                except Exception as e:  # noqa: BLE001 - the backend refused: the status says why
                    log.warning("servers %s for %s: start failed: %s", g.key, g.model, e)
                    servers = [s for s in run.servers if not s.ended] if run else []
                    state, error = "failed", f"start failed: {type(e).__name__}: {e}"
                else:
                    state, error = "starting", None
                    log.info("servers %s for %s: %s", g.key, g.model,
                             ", ".join(f"{s.job_id} ({s.stage})" for s in servers))
                self._seen(spec.swarm, g)
                runs[g.key] = _Run(g, servers, now, state, error)
                if state == "starting":
                    self._hold(spec.swarm, applied, servers, deadline)
            return self.status(spec.swarm)

    def _hold(self, swarm: str, applied: SwarmSpec | None, servers: list[Server],
              deadline: float) -> None:
        """Push back to `deadline` the deadlines of these jobs the applied spec does not use
        (a reused job: an update that may yet be applied), as their labels are now."""
        ids = {s.job_id for s in servers if not s.ended}
        for s in self._jobs(swarm, applied) or [] if ids else []:
            if (s.job_id in ids and not self._wanted(swarm, applied, s)
                    and (s.retire_at or 0) < deadline):
                self._retire(s, deadline)

    def reattach(self, specs: Iterable[SwarmSpec]) -> bool:
        """After a restart: each applied spec's groups from their running jobs (none started).
        A group with no running job is shown stopped. False: a lookup failed; call again
        (`tick` does)."""
        ok, now = True, self.clock()
        for spec in specs:
            with self.lock:
                runs = self.runs.setdefault(spec.swarm, {})
                for g in groups(spec, self.label):
                    if g.key in runs:
                        continue
                    try:
                        found = self.starter(g.kind).reattach(g)
                    except Exception:
                        log.exception("servers %s for %s: reattach failed", g.key, g.model)
                        ok = False
                        continue
                    self._seen(spec.swarm, g)
                    runs[g.key] = _Run(g, found, now) if found else _Run(
                        g, [], now, "stopped", "no running job")
                    log.info("servers %s for %s: reattached %s", g.key, g.model,
                             ", ".join(s.job_id or "?" for s in found) or "nothing")
        return ok

    def tick(self) -> None:
        """Reattach a restarted controller's servers (until every lookup succeeded), check the
        starting groups (stages, readiness, startup_timeout) and, every READY_CHECK_S, the
        ready ones (a job that ended fails its group), then sweep each swarm every
        `sweep_every` s."""
        if not self._reattached and self.applied is not None:
            self._reattached = self.reattach(self._specs().values())
        now = self.clock()
        with self.lock:
            for runs in self.runs.values():
                for run in runs.values():
                    if run.state == "starting" or (
                            run.state == "ready" and now - run.checked >= READY_CHECK_S):
                        self._check(run, now)
            if self.applied is None:
                return  # `legion serve`: its servers stay until --stop
            specs = self._specs()
            for swarm in set(specs) | set(self.runs) | self._stopped:
                if now - self._swept.get(swarm, float("-inf")) >= self.sweep_every:
                    self._sweep(swarm, specs.get(swarm), now)

    def _check(self, run: _Run, now: float) -> None:
        g, st = run.group, self.starter(run.group.kind)
        need = [s for s in run.servers if s.replica < g.block.replicas]
        try:
            st.check(g, need)
        except Exception as e:  # noqa: BLE001 - the Hub or the network: the next tick tries again
            log.warning("servers %s for %s: check failed: %s", g.key, g.model, e)
            if run.state == "starting" and now - run.since > _startup_s(g):
                waiting = next((s for s in need if not s.ready_at), Server(replica=0))
                self._fail(run, f"not ready after startup_timeout {g.block.startup_timeout} "
                                f"(checking job {waiting.job_id} failed: {e})", waiting)
            return
        run.checked = now
        if ended := next((s for s in need if s.ended), None):
            why = f"job {ended.job_id} {ended.stage}" + (f": {ended.message}" if ended.message
                                                         else "")
            self._fail(run, why, ended)
        elif run.state == "starting" and need and all(s.ready_at for s in need):
            run.state = "ready"
            log.info("servers %s for %s: ready (%s)", g.key, g.model,
                     ", ".join(s.url or "?" for s in need))
            for s in need:  # what this start began: its launch or update now has APPLY_MARGIN_S
                if s.new and (s.retire_at or 0) > now + APPLY_MARGIN_S:
                    self._retire(s, int(now + APPLY_MARGIN_S))
        elif run.state == "starting" and now - run.since > _startup_s(g):
            waiting = next(s for s in need if not s.ready_at)
            self._fail(run, f"not ready after startup_timeout {g.block.startup_timeout}: job "
                            f"{waiting.job_id} {waiting.stage}", waiting)

    def _fail(self, run: _Run, why: str, server: Server) -> None:
        """The group failed: say why (with the job's last log lines) and cancel what runs (a
        cancel that fails is tried again at the next sweep: a failed group's jobs go)."""
        st = self.starter(run.group.kind)
        try:
            tail = [line.rstrip() for line in st.logs(run.group, server, LOG_LINES) if line.strip()]
        except Exception:  # noqa: BLE001 - the reason stands without the logs
            tail = []
        run.state, run.error = "failed", why + ("; last log lines: " + " | ".join(tail)
                                                if tail else "")
        log.warning("servers %s for %s: %s", run.group.key, run.group.model, run.error)
        for s in run.servers:
            self._cancel(s)

    def _cancel(self, s: Server) -> bool:
        if s.ended or s.job_id is None:
            return True
        try:
            self.starter(s.kind).cancel(s)
        except Exception:
            log.exception("servers: cancelling %s failed (the next sweep tries again)", s.job_id)
            return False
        s.stage = "CANCELED"
        return True

    def _retire(self, s: Server, at: int) -> None:
        try:
            self.starter(s.kind).retire(s, at)
        except Exception:
            log.exception("servers: setting %s's deadline failed", s.job_id)
            return
        for runs in self.runs.values():  # the tracked copy of the job sees its new deadline
            for run in runs.values():
                for x in run.servers:
                    if x.job_id == s.job_id:
                        x.labels = dict(s.labels)

    # ------------------------------------------------------------------ the cleanup rule

    def _wanted(self, swarm: str, applied: SwarmSpec | None, s: Server) -> bool:
        """The applied spec uses this job: its group, within the group's replicas, not failed."""
        if applied is None or swarm in self._stopped:
            return False
        g = next((g for g in groups(applied, self.label) if g.key == s.group), None)
        run = self.runs.get(swarm, {}).get(s.group or "")
        return (g is not None and s.replica < g.block.replicas
                and (run is None or run.state != "failed"))

    def _jobs(self, swarm: str, applied: SwarmSpec | None) -> list[Server] | None:
        """Every running server job of the swarm, wherever its servers ran; None when a
        lookup failed."""
        where = set(self._places(swarm))
        for g in groups(applied, self.label) if applied is not None else []:
            where.add((g.kind, g.namespace))
        out: dict[str, Server] = {}  # by job id: None and the account are one namespace
        for kind, namespace in sorted(where, key=str):
            try:
                out |= {s.job_id: s for s in self.starter(kind).jobs({self.label: swarm},
                                                                     namespace)}
            except Exception:
                log.exception("servers of %s: listing jobs failed", swarm)
                return None
        return list(out.values())

    def _sweep(self, swarm: str, applied: SwarmSpec | None, now: float) -> list[str]:
        """The rule: cancel each server job of the swarm the applied spec does not use, once
        its deadline has passed (a failed group's and a stopped swarm's at once). Drops the
        runs that no longer serve anything."""
        jobs = self._jobs(swarm, applied)
        if jobs is None:
            return []
        self._swept[swarm] = now
        runs, out = self.runs.get(swarm, {}), []
        for s in jobs:
            if self._wanted(swarm, applied, s):
                continue
            run = runs.get(s.group or "")
            now_ = swarm in self._stopped or (run is not None and run.state == "failed")
            if not now_ and s.retire_at is not None and now < s.retire_at:
                continue
            if self._cancel(s):
                out.append(s.job_id)
                for r in runs.values():
                    r.servers = [x for x in r.servers if x.job_id != s.job_id]
        if out:
            log.info("servers of %s: cancelled %s (not in the applied spec)", swarm,
                     ", ".join(out))
        if applied is not None:  # a retired group with nothing left running is history
            keep = {g.key for g in groups(applied, self.label)}
            for key, run in list(runs.items()):
                if key not in keep and run.state == "ready" and not any(
                        not s.ended for s in run.servers):
                    del runs[key]
        if swarm in self._stopped and not jobs:
            self._stopped.discard(swarm)
            self.runs.pop(swarm, None)
        return out

    def retire_unused(self, spec: SwarmSpec, grace: float) -> None:
        """After an apply: each server job of the swarm the spec no longer uses keeps serving
        the draining turns for `grace` s (its deadline, pushed back to now + grace)."""
        with self.lock:
            now = self.clock()
            for s in self._jobs(spec.swarm, spec) or []:
                if not self._wanted(spec.swarm, spec, s) and (s.retire_at or 0) < now + grace:
                    self._retire(s, int(now + grace))
                    log.info("servers of %s: %s no longer in the spec, cancelled in %.0f s",
                             spec.swarm, s.job_id, grace)
            keep = {g.key for g in groups(spec, self.label)}
            runs = self.runs.get(spec.swarm, {})
            for key, run in list(runs.items()):  # an obsolete failure is history
                if key not in keep and run.state in ("failed", "stopped") and not any(
                        not s.ended for s in run.servers):
                    del runs[key]

    def stop(self, swarm: str) -> list[str]:
        """Cancel every server job of `swarm` now; a cancel that fails is tried again at each
        sweep until none is left. Returns the job ids cancelled."""
        with self.lock:
            self._stopped.add(swarm)
            for run in self.runs.get(swarm, {}).values():
                run.state = "stopped" if run.state != "failed" else run.state
            out: list[str] = []
            for run in self.runs.get(swarm, {}).values():
                for s in run.servers:
                    if not s.ended and self._cancel(s):
                        out.append(s.job_id)
            out += [j for j in self._sweep(swarm, None, self.clock()) if j not in out]
            return out

    # ------------------------------------------------------------------ the spec

    def pin(self, spec: SwarmSpec, previous: SwarmSpec | None = None) -> None:
        """Write each ready group's URLs (its first `replicas`) into its types' resolved block
        (`urls`). A type whose group is not ready keeps the URLs `previous` pinned for the same
        block and model (a re-apply right after a restart); otherwise InferenceError says what
        to do."""
        with self.lock:
            runs = self.runs.get(spec.swarm, {})
            gs = {g.key: g for g in groups(spec, self.label)}
            for name, ts in spec.types.items():
                own = group_of(spec, ts, self.label)
                if own is None:
                    continue
                g = gs[own.key]
                r, run = ts.resolved_model, runs.get(g.key)
                old = previous.types.get(name) if previous is not None else None
                old = old.resolved_model if old is not None else None
                if run is not None and run.state == "ready":
                    urls = [s.url for s in sorted(run.servers, key=lambda s: s.replica)
                            if s.replica < g.block.replicas and not s.ended]
                elif (old is not None and old.inference.urls and old.model_id == r.model_id
                      and old.inference.unpinned() == r.inference.unpinned()
                      and (run is None or run.state == "starting")):
                    urls = list(old.inference.urls)
                else:
                    state = "not started" if run is None else run.state + (
                        f" ({run.error})" if run.error else "")
                    raise InferenceError(
                        f"types.{name}.model {r.model_id!r} on {r.provider_name}: its {g.kind} "
                        f"servers are {state}: `legion launch` starts them and waits (POST "
                        f"/swarms/{spec.swarm}/servers)")
                r.inference = r.inference.model_copy(update={"urls": urls})

    def status(self, swarm: str) -> list[ServerStatus]:
        with self.lock:
            out = []
            for run in self.runs.get(swarm, {}).values():
                g = run.group
                rows = sorted(run.servers, key=lambda s: s.replica) or [Server(replica=0)]
                for s in rows:
                    state = run.state
                    if state == "ready" and s.ended:
                        state = "failed"
                    out.append(ServerStatus(
                        model=g.model, kind=g.kind, group=g.key, types=g.types,
                        flavor=g.block.flavor, namespace=g.namespace or "", replica=s.replica,
                        job_id=s.job_id, url=s.url, stage=s.stage, message=s.message, state=state,
                        started_at=_dt(s.started_at), ready_at=_dt(s.ready_at),
                        error=run.error))
            return out
