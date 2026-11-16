"""Inference servers flock starts itself (docs/open-models.md "Servers flock starts"): an
`inference` block with `hf_jobs` names no URLs; the controller starts one group of servers per
model on it, waits until every replica answers `GET /v1/models`, pins their URLs into the spec
as `urls` would be (`resolved_model.inference.urls`), and stops them when the run ends. Past
startup everything is the `urls` path: the model check, the context window, preflight, Pi.

`Servers` keeps the groups of each swarm (in memory: a restarted controller reattaches to the
running jobs by their labels) and does the generic part: start, the readiness wait with its
`startup_timeout`, failures, pinning, status and stop. A starter (flock.registry SERVERS, one
module per block kind, e.g. controller/hf_jobs_servers.py) does the backend's part.

    servers = Servers(os.environ)
    servers.start(spec)          # start, or reattach to the running jobs; returns at once
    servers.tick()               # every few seconds: stages, readiness, startup_timeout
    servers.pin(spec)            # once ready: `urls` into each type's resolved block
    servers.prune(spec, grace)   # after an apply: retire what the spec no longer uses
    servers.stop(spec.swarm)     # cancel (also jobs labelled for the swarm it lost track of)
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

from flock.controller.inference import InferenceError
from flock.models import ServerStatus, SwarmSpec, TypeSpec, parse_duration
from flock.registry import SERVERS, load

log = logging.getLogger("flock.servers")

TERMINAL = frozenset({"COMPLETED", "CANCELED", "ERROR", "DELETED"})
READY_CHECK_S = 60.0  # a ready group's jobs are checked this often (between: none)
LOG_LINES = 5  # a failure's message ends with the job's last log lines


@dataclass
class Group:
    """One model's servers on one block's settings: what a starter starts. `block` is the
    block's `hf_jobs` (a server block has `replicas` and `startup_timeout`)."""

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
    """One replica: its job, URL and readiness."""

    replica: int
    job_id: str | None = None
    url: str | None = None  # the OpenAI base URL (`.../v1`)
    stage: str | None = None
    message: str | None = None
    started_at: float | None = None
    ready_at: float | None = None  # first answered GET /v1/models

    @property
    def ended(self) -> bool:
        return self.stage in TERMINAL


class StartError(Exception):
    """A start that failed part-way: the jobs it started were cancelled again; `servers` are
    the group's running ones it found, still running (the caller tracks them, to cancel)."""

    def __init__(self, message: str, servers: list[Server]):
        super().__init__(message)
        self.servers = servers


class Starter(Protocol):
    def start(self, group: Group) -> list[Server]:
        """The group's servers, one per replica: the running ones (found by `group.labels`)
        reused, the missing ones started. StartError when starting one fails."""

    def reattach(self, group: Group) -> list[Server]:
        """The group's running servers, started by an earlier controller; none started."""

    def check(self, group: Group, servers: list[Server]) -> None:
        """Refresh each server's stage, and set `ready_at` once it answers GET /v1/models."""

    def logs(self, group: Group, server: Server, n: int) -> list[str]:
        """The server's last `n` log lines (for a failure's message)."""

    def stop(self, group: Group, servers: list[Server]) -> None:
        """Cancel the servers that still run."""

    def sweep(self, labels: dict[str, str], namespace: str | None, keep: set[str]) -> list[str]:
        """Cancel every running job carrying `labels` whose group (its `inference` label) is
        not in `keep`, in `namespace` (None: the default); returns their ids."""


def server_kind(inference: Any) -> str | None:
    """The block key of the servers flock starts for this block (`hf_jobs`), or None."""
    return next((k for k in SERVERS if getattr(inference, k, None) is not None), None)


def group_of(spec: SwarmSpec, ts: TypeSpec, label: str = "swarm") -> Group | None:
    """The group serving type `ts` (None: its block names no servers to start). Types with the
    same model and settings share one; the key covers what goes into the jobs."""
    r = ts.resolved_model
    kind = server_kind(r.inference) if r is not None else None
    if kind is None:
        return None
    block = getattr(r.inference, kind)
    namespace = block.namespace or spec.namespace or None
    what = [kind, r.model_id, r.inference.max_working, namespace,
            block.model_dump(mode="json", exclude={"startup_timeout", "replicas", "namespace",
                                                   "timeout"})]
    key = hashlib.sha256(json.dumps(what, sort_keys=True).encode()).hexdigest()[:12]
    return Group(key=key, kind=kind, model=r.model_id, block=block,
                 max_working=r.inference.max_working, namespace=namespace,
                 labels={label: spec.swarm, "inference": key})


def groups(spec: SwarmSpec, label: str = "swarm") -> list[Group]:
    """Each distinct group of the spec's types, with the types it serves."""
    out: dict[str, Group] = {}
    for name, ts in spec.types.items():
        g = group_of(spec, ts, label)
        if g is not None:
            out.setdefault(g.key, g).types.append(name)
    return list(out.values())



def drain_grace(spec: SwarmSpec) -> float:
    """How long servers a spec no longer uses keep serving: the types' longest `turn_timeout`,
    the turns their agents' draining workers may still run on them."""
    return max((parse_duration(ts.turn_timeout).total_seconds() for ts in spec.types.values()),
               default=0.0)


@dataclass
class _Run:
    """A group's servers in this process."""

    group: Group
    servers: list[Server]
    since: float  # the wait's start: `startup_timeout` counts from here
    state: str = "starting"  # starting, ready, failed, stopped
    error: str | None = None
    checked: float = 0.0
    retire_at: float | None = None  # no longer in the spec: cancelled (and dropped) at this time


@dataclass
class _Sweep:
    """After a restart: the swarm's labelled jobs of groups its spec no longer has (a
    retirement the restart lost) are cancelled at `at`."""

    at: float
    keep: set[str]  # the spec's group keys
    where: set[tuple[str, str | None]]  # (kind, namespace) to look in


def _dt(t: float | None) -> datetime | None:
    return None if t is None else datetime.fromtimestamp(t, UTC)


class Servers:
    """The servers of each swarm (or, for `flock serve`, each spec: `label="serve"`).
    Thread-safe: the API's handlers and the ticker call it. `stored()` lists the specs a
    restarted controller holds: `tick` reattaches to their running servers until it has them
    all, and sweeps their obsolete groups' jobs once their drain grace is over."""

    def __init__(self, env: Mapping[str, str], starters: Mapping[str, Starter] | None = None,
                 label: str = "swarm", clock=time.time,
                 stored: Callable[[], list[SwarmSpec]] | None = None):
        self.env, self.label, self.clock, self.stored = env, label, clock, stored
        self._starters: dict[str, Starter] = dict(starters or {})
        self.runs: dict[str, dict[str, _Run]] = {}
        self.lock = threading.RLock()
        self._reattached = False
        self._sweeps: dict[str, _Sweep] = {}
        # every (kind, namespace) a swarm's servers ran in: where `stop` looks for its jobs
        self._where: dict[str, set[tuple[str, str | None]]] = {}

    def starter(self, kind: str) -> Starter:
        if kind not in self._starters:
            self._starters[kind] = load(SERVERS[kind])(self.env)
        return self._starters[kind]

    def _seen(self, swarm: str, g: Group) -> None:
        self._where.setdefault(swarm, set()).add((g.kind, g.namespace))

    # ------------------------------------------------------------------ lifecycle

    def start(self, spec: SwarmSpec) -> list[ServerStatus]:
        """Start the spec's groups that are not starting or ready (a failed or stopped one
        starts again), reusing running jobs; returns at once with their status. A start that
        fails keeps the group's running jobs tracked (`stop` cancels them)."""
        with self.lock:
            runs = self.runs.setdefault(spec.swarm, {})
            for g in groups(spec, self.label):
                run = runs.get(g.key)
                want = g.block.replicas
                if run is not None and run.state in ("starting", "ready") and sum(
                        not s.ended and s.replica < want for s in run.servers) >= want:
                    run.group.types, run.group.block = g.types, g.block  # replicas may have changed
                    run.retire_at = None  # in the spec again
                    continue
                kept = [s for s in run.servers if not s.ended] if run is not None else []
                try:
                    servers = self.starter(g.kind).start(g)
                except StartError as e:  # its own jobs rolled back; the running ones kept
                    log.warning("servers %s for %s: start failed: %s", g.key, g.model, e)
                    runs[g.key] = _Run(g, e.servers, self.clock(), "failed", f"start failed: {e}")
                    continue
                except Exception as e:  # noqa: BLE001 - the backend refused: the status says why
                    log.warning("servers %s for %s: start failed: %s", g.key, g.model, e)
                    runs[g.key] = _Run(g, kept, self.clock(), "failed",
                                       f"start failed: {type(e).__name__}: {e}")
                    continue
                finally:
                    self._seen(spec.swarm, g)
                runs[g.key] = _Run(g, servers, self.clock())
                log.info("servers %s for %s: %s", g.key, g.model,
                         ", ".join(f"{s.job_id} ({s.stage})" for s in servers))
            return self.status(spec.swarm)

    def reattach(self, specs: Iterable[SwarmSpec]) -> bool:
        """After a restart: each stored spec's groups from their running jobs (none started),
        replicas beyond the spec's retired after the drain grace, and a sweep of the jobs of
        groups the spec no longer has scheduled at the end of that grace. A group with no
        running job is shown stopped. False: a lookup failed; call again (`tick` does)."""
        ok, now = True, self.clock()
        for spec in specs:
            gs, grace = groups(spec, self.label), drain_grace(spec)
            with self.lock:
                runs = self.runs.setdefault(spec.swarm, {})
                for g in gs:
                    if g.key in runs:
                        continue
                    try:
                        found = self.starter(g.kind).reattach(g)
                    except Exception:
                        log.exception("servers %s for %s: reattach failed", g.key, g.model)
                        ok = False
                        continue
                    self._seen(spec.swarm, g)
                    want = g.block.replicas
                    servers = [s for s in found if s.replica < want]
                    runs[g.key] = _Run(g, servers, now) if servers else _Run(
                        g, [], now, "stopped", "no running job")
                    self._retire_servers(spec.swarm, g, [s for s in found if s.replica >= want],
                                         now + grace, "starting")
                    log.info("servers %s for %s: reattached %s", g.key, g.model,
                             ", ".join(s.job_id or "?" for s in found) or "nothing")
                if gs and spec.swarm not in self._sweeps:
                    self._sweeps[spec.swarm] = _Sweep(now + grace, {g.key for g in gs},
                                                      {(g.kind, g.namespace) for g in gs})
        return ok

    def tick(self) -> None:
        """Reattach a restarted controller's servers (until it has them all), run due sweeps,
        cancel retired groups whose grace is over, and check the starting groups (stages,
        readiness, startup_timeout) and, every READY_CHECK_S, the ready ones: a job that ended
        fails its group."""
        if not self._reattached and self.stored is not None:
            self._reattached = self.reattach(self.stored())
        now = self.clock()
        with self.lock:
            for swarm, sweep in list(self._sweeps.items()):
                if now >= sweep.at and self._sweep(swarm, sweep.where, sweep.keep) is not None:
                    del self._sweeps[swarm]
            for runs in self.runs.values():
                for key, run in list(runs.items()):
                    if run.retire_at is not None and now >= run.retire_at:
                        if self._cancel(run):
                            del runs[key]
                    elif run.state == "starting" or (
                            run.state == "ready" and now - run.checked >= READY_CHECK_S):
                        self._check(run, now)

    def _sweep(self, swarm: str, where: Iterable[tuple[str, str | None]], keep: set[str]
               ) -> list[str] | None:
        """Cancel the swarm's labelled jobs whose group is not in `keep`: their ids, or None
        when a lookup failed."""
        out: list[str] = []
        for kind, namespace in where:
            try:
                out += self.starter(kind).sweep({self.label: swarm}, namespace, keep)
            except Exception:
                log.exception("servers of %s: sweep failed", swarm)
                return None
        if out:
            log.info("servers of %s: cancelled %s (groups no longer in the spec)", swarm,
                     ", ".join(out))
        return out

    def _check(self, run: _Run, now: float) -> None:
        g, st = run.group, self.starter(run.group.kind)
        try:
            st.check(g, run.servers)
        except Exception as e:  # noqa: BLE001 - the Hub or the network: the next tick tries again
            log.warning("servers %s for %s: check failed: %s", g.key, g.model, e)
            if run.state == "starting" and now - run.since > self._startup_s(g):
                waiting = next((s for s in run.servers if not s.ready_at), Server(replica=0))
                self._fail(run, f"not ready after startup_timeout {g.block.startup_timeout} "
                                f"(checking job {waiting.job_id} failed: {e})", waiting)
            return
        run.checked = now
        if ended := next((s for s in run.servers if s.ended), None):
            why = f"job {ended.job_id} {ended.stage}" + (f": {ended.message}" if ended.message
                                                         else "")
            self._fail(run, why, ended)
        elif run.state == "starting" and all(s.ready_at for s in run.servers):
            run.state = "ready"
            log.info("servers %s for %s: ready (%s)", g.key, g.model,
                     ", ".join(s.url or "?" for s in run.servers))
        elif run.state == "starting" and now - run.since > self._startup_s(g):
            waiting = next(s for s in run.servers if not s.ready_at)
            self._fail(run, f"not ready after startup_timeout {g.block.startup_timeout}: job "
                            f"{waiting.job_id} {waiting.stage}", waiting)

    @staticmethod
    def _startup_s(g: Group) -> float:
        return parse_duration(g.block.startup_timeout).total_seconds()

    def _fail(self, run: _Run, why: str, server: Server) -> None:
        """The group failed: say why (with the job's last log lines) and cancel what runs."""
        st = self.starter(run.group.kind)
        try:
            tail = [line.rstrip() for line in st.logs(run.group, server, LOG_LINES) if line.strip()]
        except Exception:  # noqa: BLE001 - the reason stands without the logs
            tail = []
        run.state, run.error = "failed", why + ("; last log lines: " + " | ".join(tail)
                                                if tail else "")
        log.warning("servers %s for %s: %s", run.group.key, run.group.model, run.error)
        self._cancel(run, state="failed")

    def _cancel(self, run: _Run, state: str = "stopped") -> list[str] | None:
        """Cancel the run's servers that still run: their ids, or None when cancelling failed
        (the run keeps them, for the next try)."""
        live = [s for s in run.servers if not s.ended]
        if live:
            try:
                self.starter(run.group.kind).stop(run.group, live)
            except Exception:
                log.exception("servers %s: cancel failed", run.group.key)
                return None
            log.info("servers %s for %s: stopped %s", run.group.key, run.group.model,
                     ", ".join(s.job_id or "?" for s in live))
        if run.state != "failed":
            run.state = state
        run.retire_at = None
        return [s.job_id for s in live if s.job_id]

    def _retire_servers(self, swarm: str, g: Group, servers: list[Server], at: float,
                        state: str) -> None:
        """Replicas a group no longer needs: each its own run, cancelled at `at`."""
        for s in servers:
            self.runs[swarm][f"{g.key}/r{s.replica}"] = _Run(g, [s], self.clock(), state,
                                                            retire_at=at)

    def stop(self, swarm: str) -> list[str]:
        """Cancel every server of `swarm`: the ones tracked, then any job labelled for it (one
        this controller lost track of); returns the job ids."""
        out: list[str] = []
        with self.lock:
            for run in self.runs.get(swarm, {}).values():
                out += self._cancel(run) or []
            self._sweeps.pop(swarm, None)
            out += [j for j in self._sweep(swarm, self._where.get(swarm, ()), set()) or []
                    if j not in out]
        return out

    def prune(self, spec: SwarmSpec, grace: float = 0.0) -> list[str]:
        """After an apply: the swarm's groups the spec no longer uses, and the replicas beyond
        its `replicas`, are cancelled after `grace` s (the turns their agents' draining
        workers still run on them; 0: now). Obsolete groups with nothing running (failed,
        stopped) are dropped. Returns the job ids cancelled now."""
        keep = {g.key: g for g in groups(spec, self.label)}
        now, out = self.clock(), []
        with self.lock:
            runs = self.runs.get(spec.swarm, {})
            for key, run in list(runs.items()):
                g = keep.get(key)
                if g is not None:
                    excess = [s for s in run.servers if s.replica >= g.block.replicas]
                    if excess:
                        run.servers = [s for s in run.servers if s.replica < g.block.replicas]
                        self._retire_servers(spec.swarm, run.group, excess, now + grace,
                                             run.state)
                elif not any(not s.ended for s in run.servers):
                    del runs[key]  # nothing runs: an obsolete failure is history
                elif run.retire_at is None:
                    run.retire_at = now + grace
                    log.info("servers %s for %s: no longer in the spec, cancelled in %.0f s",
                             key, run.group.model, grace)
            for key, run in list(runs.items()):  # a grace of 0: now
                if (run.retire_at is not None and run.retire_at <= now
                        and (done := self._cancel(run)) is not None):
                    out += done
                    del runs[key]
        return out

    def drop_unused(self, swarm: str, applied: SwarmSpec | None) -> list[str]:
        """A launch or update that did not apply: cancel now the groups it started that the
        applied spec does not use (all of the swarm's when none is applied); the ones retiring
        keep their grace. Returns the job ids."""
        if applied is None:
            return self.stop(swarm)
        keep = {g.key for g in groups(applied, self.label)}
        out: list[str] = []
        with self.lock:
            runs = self.runs.get(swarm, {})
            for key, run in list(runs.items()):
                if (key not in keep and run.retire_at is None
                        and (done := self._cancel(run)) is not None):
                    out += done
                    del runs[key]
        return out

    # ------------------------------------------------------------------ the spec

    def pin(self, spec: SwarmSpec, previous: SwarmSpec | None = None) -> None:
        """Write each ready group's URLs (its first `replicas`) into its types' resolved block
        (`urls`). A type whose group is not ready keeps the URLs `previous` pinned for the same
        block and model (a re-apply right after a restart); otherwise InferenceError says what
        to do."""
        with self.lock:
            runs = self.runs.get(spec.swarm, {})
            for name, ts in spec.types.items():
                g = group_of(spec, ts, self.label)
                if g is None:
                    continue
                r, run = ts.resolved_model, runs.get(g.key)
                old = previous.types.get(name) if previous is not None else None
                old = old.resolved_model if old is not None else None
                if run is not None and run.state == "ready":
                    urls = [s.url for s in sorted(run.servers, key=lambda s: s.replica)
                            if s.replica < g.block.replicas]
                elif (old is not None and old.inference.urls and old.model_id == r.model_id
                      and old.inference.unpinned() == r.inference.unpinned()
                      and (run is None or run.state == "starting")):
                    urls = list(old.inference.urls)
                else:
                    state = "not started" if run is None else run.state + (
                        f" ({run.error})" if run.error else "")
                    raise InferenceError(
                        f"types.{name}.model {r.model_id!r} on {r.provider_name}: its {g.kind} "
                        f"servers are {state}: `flock launch` starts them and waits (POST "
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
