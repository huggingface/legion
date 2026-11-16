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
    servers.stop(spec.swarm)     # cancel
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


class Starter(Protocol):
    def start(self, group: Group) -> list[Server]:
        """The group's servers, one per replica: the running ones (found by `group.labels`)
        reused, the missing ones started."""

    def reattach(self, group: Group) -> list[Server]:
        """The group's running servers, started by an earlier controller; none started."""

    def check(self, group: Group, servers: list[Server]) -> None:
        """Refresh each server's stage, and set `ready_at` once it answers GET /v1/models."""

    def logs(self, group: Group, server: Server, n: int) -> list[str]:
        """The server's last `n` log lines (for a failure's message)."""

    def stop(self, group: Group, servers: list[Server]) -> None:
        """Cancel the servers that still run."""


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


@dataclass
class _Run:
    """A group's servers in this process."""

    group: Group
    servers: list[Server]
    since: float  # the wait's start: `startup_timeout` counts from here
    state: str = "starting"  # starting, ready, failed, stopped
    error: str | None = None
    checked: float = 0.0


def _dt(t: float | None) -> datetime | None:
    return None if t is None else datetime.fromtimestamp(t, UTC)


class Servers:
    """The servers of each swarm (or, for `flock serve`, each spec: `label="serve"`).
    Thread-safe: the API's handlers and the ticker call it. `stored()` lists the specs a
    restarted controller holds: the first `tick` reattaches to their running servers."""

    def __init__(self, env: Mapping[str, str], starters: Mapping[str, Starter] | None = None,
                 label: str = "swarm", clock=time.time,
                 stored: Callable[[], list[SwarmSpec]] | None = None):
        self.env, self.label, self.clock, self.stored = env, label, clock, stored
        self._starters: dict[str, Starter] = dict(starters or {})
        self.runs: dict[str, dict[str, _Run]] = {}
        self.lock = threading.RLock()
        self._reattached = False

    def starter(self, kind: str) -> Starter:
        if kind not in self._starters:
            self._starters[kind] = load(SERVERS[kind])(self.env)
        return self._starters[kind]

    # ------------------------------------------------------------------ lifecycle

    def start(self, spec: SwarmSpec) -> list[ServerStatus]:
        """Start the spec's groups that are not starting or ready (a failed or stopped one
        starts again), reusing running jobs; returns at once with their status."""
        with self.lock:
            runs = self.runs.setdefault(spec.swarm, {})
            for g in groups(spec, self.label):
                run = runs.get(g.key)
                live = run is not None and run.state in ("starting", "ready") and sum(
                    not s.ended for s in run.servers) >= g.block.replicas
                if live:
                    run.group.types = g.types  # the types it serves may have changed
                    continue
                try:
                    servers = self.starter(g.kind).start(g)
                except Exception as e:  # noqa: BLE001 - the backend refused: the status says why
                    log.warning("servers %s for %s: start failed: %s", g.key, g.model, e)
                    runs[g.key] = _Run(g, [], self.clock(), "failed",
                                       f"start failed: {type(e).__name__}: {e}")
                    continue
                runs[g.key] = _Run(g, servers, self.clock())
                log.info("servers %s for %s: %s", g.key, g.model,
                         ", ".join(f"{s.job_id} ({s.stage})" for s in servers))
            return self.status(spec.swarm)

    def reattach(self, specs: Iterable[SwarmSpec]) -> None:
        """After a restart: each stored spec's groups from their running jobs (none started).
        A group with no running job is shown stopped."""
        for spec in specs:
            for g in groups(spec, self.label):
                with self.lock:
                    runs = self.runs.setdefault(spec.swarm, {})
                    if g.key in runs:
                        continue
                    try:
                        servers = self.starter(g.kind).reattach(g)
                    except Exception:
                        log.exception("servers %s for %s: reattach failed", g.key, g.model)
                        continue
                    runs[g.key] = _Run(g, servers, self.clock()) if servers else _Run(
                        g, [], self.clock(), "stopped", "no running job")
                    log.info("servers %s for %s: reattached %s", g.key, g.model,
                             ", ".join(s.job_id or "?" for s in servers) or "nothing")

    def tick(self) -> None:
        """Check the starting groups (stages, readiness, startup_timeout) and, every
        READY_CHECK_S, the ready ones: a job that ended fails its group."""
        if not self._reattached and self.stored is not None:
            self.reattach(self.stored())
        self._reattached = True
        now = self.clock()
        with self.lock:
            for runs in self.runs.values():
                for run in runs.values():
                    if run.state == "starting" or (
                            run.state == "ready" and now - run.checked >= READY_CHECK_S):
                        self._check(run, now)

    def _check(self, run: _Run, now: float) -> None:
        g, st = run.group, self.starter(run.group.kind)
        try:
            st.check(g, run.servers)
        except Exception as e:  # noqa: BLE001 - the Hub or the network: the next tick tries again
            log.warning("servers %s for %s: check failed: %s", g.key, g.model, e)
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
        try:
            st.stop(run.group, run.servers)
        except Exception:
            log.exception("servers %s: cancel failed", run.group.key)

    def stop(self, swarm: str, keep: Iterable[str] = ()) -> list[str]:
        """Cancel every server of `swarm` (the groups in `keep` aside); returns the job ids."""
        keep, out = set(keep), []
        with self.lock:
            for key, run in self.runs.get(swarm, {}).items():
                live = [s for s in run.servers if not s.ended]
                if key in keep or not live:
                    if key not in keep and run.state in ("starting", "ready"):
                        run.state = "stopped"
                    continue
                try:
                    self.starter(run.group.kind).stop(run.group, live)
                except Exception:
                    log.exception("servers %s: cancel failed", key)
                    continue
                out += [s.job_id for s in live if s.job_id]
                if run.state != "failed":
                    run.state = "stopped"
                log.info("servers %s for %s: stopped %s", key, run.group.model,
                         ", ".join(s.job_id or "?" for s in live))
        return out

    def prune(self, spec: SwarmSpec) -> list[str]:
        """After an apply: stop the swarm's groups the spec no longer uses."""
        return self.stop(spec.swarm, keep=[g.key for g in groups(spec, self.label)])

    # ------------------------------------------------------------------ the spec

    def pin(self, spec: SwarmSpec, previous: SwarmSpec | None = None) -> None:
        """Write each ready group's URLs into its types' resolved block (`urls`). A type whose
        group is not ready keeps the URLs `previous` pinned for the same block and model (a
        re-apply right after a restart); otherwise InferenceError says what to do."""
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
                    urls = [s.url for s in sorted(run.servers, key=lambda s: s.replica)]
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
