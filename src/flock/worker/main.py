"""flock-worker: one HF Job hosting up to `agents_per_worker` agents.

Tasks: command long-poll, heartbeat (random initial phase), events flusher, periodic sync,
deadline watcher. Drain (`drain_window` before DEADLINE, or a drain command): stop starting
harness processes, let in-flight turns finish (bounded by turn_timeout), final sync per
agent, flush events, last heartbeat, exit 0. Controller errors are retried with backoff
(429 and 5xx inside the client first, honouring Retry-After: a Space edge throttles a 1k swarm),
except 404/409 on our own worker routes: the controller says this slot has no live job (we
are a stale process and a replacement may already own our agents), so we kill the harness
processes and exit 0 without syncing, to avoid overwriting the replacement's bucket state.

Preemption (SIGUSR1, and SIGTERM outside compute local; docs/slurm.md): the machine is being
taken away. A cluster's Spot watcher cancels the job about 30 s after the signal, so the whole
handler is bounded by FLOCK_DRAIN_BUDGET_S (default 20 s): stop admitting turns, interrupt every
running turn at once (`turn_interrupted`), sync every agent to the bucket in parallel, write
`<swarm>/_workers/<worker>/preempted.json` in the bucket, flush events, send a last heartbeat
with `preempted: true`, exit PREEMPT_EXIT (75). The controller relaunches the slot at once (a
planned restart, no crash backoff) and the new job restores every agent from the bucket.

Memory relief ladder (docs/slurm.md "Memory"), every MEMORY_SAMPLE s on the cgroup's
nonreclaimable memory against its limit (`MemThresholds`, FLOCK_MEM_*): `wants_rotation` in the
heartbeat, then a drain bounded by a deadline (exit 0), then `preempt("memory")` (exit 75). It
acts on the whole worker: no per-agent reading picks a Trial to restart.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import random
import re
import signal
import sys
import threading
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

from flock.home import homeless_defaults
from flock.models import (
    HEARTBEAT_INTERVAL,
    AssignCommand,
    Command,
    CommandBatch,
    DrainCommand,
    EventBatch,
    LoopStartCommand,
    LoopStopCommand,
    PromptCommand,
    RestoreProgress,
    StopCommand,
    TypeSpec,
    UnassignCommand,
    WorkerHeartbeat,
    WorkerKnobs,
    WorkerMem,
    parse_duration,
    type_spec_rev,
)
from flock.registry import HARNESSES, load
from flock.secrets_file import load as load_secrets
from flock.secrets_file import secret_env
from flock.worker import sync as sync_mod
from flock.worker.adapters.base import HarnessAdapter
from flock.worker.agents import AgentRuntime
from flock.worker.cgroup import TMPFS_TYPES, CgroupMem, MemThresholds, fs_type
from flock.worker.events import Outbox, parse_iso, utcnow
from flock.worker.harbor import HarborHost
from flock.worker.sync import (
    BucketStats,
    BucketSync,
    HubBucketStore,
    count_hub_http,
    restore_batch,
    restore_threads,
    workdir_excludes,
)
from flock.worker.tools import ToolsManager, tools_root
from flock.world.protocol import fill_host

log = logging.getLogger("flock.worker")

DEFAULT_DRAIN_WINDOW = timedelta(minutes=10)
PREEMPT_EXIT = 75  # EX_TEMPFAIL: the job ended because its machine was taken away
DEFAULT_DRAIN_BUDGET = 20.0  # s, FLOCK_DRAIN_BUDGET_S: the whole preemption handler
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: str, env: Mapping[str, str]) -> str:
    """Expand `${NAME}` from `env`. A job-mode controller hands workers
    `http://${HF_NETWORK_GROUP_PREFIX}controller:7860`: HF Jobs sets that prefix only inside
    network-group members, and only the worker can read it (docs/deploy-job.md)."""

    def sub(m: re.Match) -> str:
        if m.group(1) not in env:
            raise ValueError(
                f"{value!r} needs ${{{m.group(1)}}}, which is not set: is this worker in the "
                "controller's network group?"
            )
        return env[m.group(1)]

    return _VAR.sub(sub, value)


@dataclass
class WorkerConfig:
    url: str
    token: str
    worker_id: str
    swarm: str
    type: str
    deadline: datetime
    bucket: str | None
    work_root: Path
    poll_wait: int = 25
    poll_min_interval: float = 1.0  # guard against a controller that answers empty polls at once
    heartbeat_interval: float = HEARTBEAT_INTERVAL.total_seconds()
    flush_interval: float = 3.0
    sync_interval: float = 180.0
    drain_window: timedelta = DEFAULT_DRAIN_WINDOW  # from the job env; assigns may update it
    incarnation: str | None = None  # WORKER_INCARNATION: binds our polls to this job
    # compute local (docs/local.md): the controller's pid; the worker exits once it is gone
    parent_pid: int | None = None
    drain_budget: float = DEFAULT_DRAIN_BUDGET  # s, the preemption handler (SIGUSR1)
    # Slurm (docs/slurm.md): FLOCK_URL is a template with ${CTL_HOST}, read from this file at
    # start and again when the controller stops answering (it may have moved host)
    url_template: str | None = None
    host_file: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> WorkerConfig:
        host_file = env.get("FLOCK_CTL_HOST_FILE") or None
        template = env["FLOCK_URL"] if host_file and "${CTL_HOST}" in env["FLOCK_URL"] else None
        url = env["FLOCK_URL"]
        if template is not None:
            url = fill_host(url, "${CTL_HOST}", env.get("CTL_HOST") or read_host_file(host_file))
        return cls(
            url=expand_env(url, env).rstrip("/"),
            url_template=template,
            host_file=host_file,
            drain_budget=float(env.get("FLOCK_DRAIN_BUDGET_S") or DEFAULT_DRAIN_BUDGET),
            token=env["FLOCK_TOKEN"],
            worker_id=env["WORKER_ID"],
            swarm=env["SWARM"],
            type=env["TYPE"],
            deadline=parse_iso(env["DEADLINE"]),
            bucket=env.get("BUCKET_ID") or None,
            work_root=Path(env.get("WORK_ROOT", "/work/agents")),
            sync_interval=float(env.get("SYNC_INTERVAL", 180)),
            flush_interval=float(env.get("FLUSH_INTERVAL", 3.0)),
            heartbeat_interval=float(env.get("HEARTBEAT_INTERVAL", HEARTBEAT_INTERVAL.total_seconds())),
            drain_window=parse_duration(env.get("DRAIN_WINDOW") or DEFAULT_DRAIN_WINDOW),
            incarnation=env.get("WORKER_INCARNATION") or None,
            parent_pid=int(env["FLOCK_LOCAL_PARENT_PID"]) if env.get("FLOCK_LOCAL_PARENT_PID")
            else None,
        )


def read_host_file(path: str) -> str:
    """The controller's `host` or `host:port` from its host file (first word; the controller
    job writes it)."""
    try:
        words = Path(path).read_text().split()
    except OSError as e:
        raise ValueError(f"controller host file {path}: {e}") from None
    if not words:
        raise ValueError(f"controller host file {path} is empty")
    return words[0]


RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
UNSENT_STATUSES = frozenset({429, 503})  # not processed: safe to resend a non-idempotent call
RETRY_BASE = 1.0  # s, first retry delay before jitter
RETRY_CAP = 30.0  # s, longest single wait, Retry-After included
MAX_CONCURRENCY = 3  # requests in flight per worker: one poll, one heartbeat, one event batch


def _retry_after(r: httpx.Response) -> float | None:
    """Retry-After as seconds (delta-seconds or an HTTP date), if the response has one."""
    raw = r.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, (when - utcnow()).total_seconds()) if when.tzinfo else None


def _retry_delay(attempt: int, retry_after: float | None) -> float:
    """Exponential backoff with full jitter, never shorter than the server's Retry-After."""
    backoff = random.uniform(0, min(RETRY_CAP, RETRY_BASE * 2 ** attempt))
    return min(RETRY_CAP, max(backoff, retry_after or 0.0))


class ControllerClient:
    """The worker's three controller routes. 429, 5xx and transport errors are retried here
    with backoff and jitter, honouring Retry-After, waiting up to `budget` s per call; at
    most MAX_CONCURRENCY requests are in flight (waits do not hold a slot). `stats` counts
    requests, throttled (429) responses and retries for the worker's log."""

    def __init__(self, url: str, token: str, *, transport: httpx.AsyncBaseTransport | None = None,
                 incarnation: str | None = None, max_concurrency: int = MAX_CONCURRENCY,
                 sleep: Callable[[float], Awaitable[object]] | None = None,
                 resolve_url: Callable[[], str] | None = None) -> None:
        self.http = httpx.AsyncClient(base_url=url, headers={"Authorization": f"Bearer {token}"},
                                      timeout=httpx.Timeout(20, read=60), transport=transport)
        self.params = {"incarnation": incarnation} if incarnation else {}
        self._slots = asyncio.Semaphore(max(1, max_concurrency))
        self._sleep = sleep or asyncio.sleep
        self.stats: Counter[str] = Counter()
        # re-reads where the controller is (a host file on Slurm) after a transport error, and
        # before every heartbeat (Worker.send_heartbeat), so a move is seen within one interval
        self.resolve_url = resolve_url

    def refresh_url(self) -> None:
        if self.resolve_url is None:
            return
        try:
            url = self.resolve_url().rstrip("/")
        except (OSError, ValueError) as e:
            log.warning("controller address: %s", e)
            return
        if url and url != str(self.http.base_url).rstrip("/"):
            log.warning("controller moved: %s -> %s", self.http.base_url, url)
            self.http.base_url = url
            self.stats["moved"] += 1

    async def _send(self, method: str, path: str, *, budget: float, retry_sent: bool = True,
                    **kw) -> httpx.Response:
        """`budget`: seconds this call may spend waiting between attempts. `retry_sent=False`:
        retry only what never reached the app (no connection, 429, 503); a request that may
        have been processed (read error or timeout, other 5xx) is not sent again."""
        waited = 0.0
        attempt = 0
        while True:
            r: httpx.Response | None = None
            err: httpx.TransportError | None = None
            async with self._slots:
                self.stats["requests"] += 1
                try:
                    r = await self.http.request(method, path, **kw)
                except httpx.TransportError as e:
                    err = e
            if err is not None:
                self.refresh_url()
            if r is not None and r.status_code not in (RETRY_STATUSES if retry_sent
                                                       else UNSENT_STATUSES):
                r.raise_for_status()
                return r
            if err is not None and not retry_sent and not isinstance(
                    err, httpx.ConnectError | httpx.ConnectTimeout):
                raise err
            if r is not None and r.status_code == 429:
                self.stats["throttled"] += 1
            delay = _retry_delay(attempt, _retry_after(r) if r is not None else None)
            if waited + delay > budget:
                if r is not None:
                    r.raise_for_status()
                raise err  # type: ignore[misc]
            attempt += 1
            self.stats["retries"] += 1
            waited += delay
            await self._sleep(delay)

    async def commands(self, worker: str, wait: int, *, budget: float = 30.0) -> CommandBatch:
        r = await self._send("GET", f"/workers/{worker}/commands", budget=budget,
                             params={"wait": wait, **self.params},
                             timeout=httpx.Timeout(20, read=wait + 20))
        return CommandBatch.model_validate(r.json() if r.content else {"commands": []})

    async def heartbeat(self, hb: WorkerHeartbeat, *, budget: float = 20.0) -> dict:
        """The controller's answer: {"ok": true}, and a changed type spec (docs/live.md)."""
        r = await self._send("POST", f"/workers/{hb.worker}/heartbeat", budget=budget,
                             content=hb.model_dump_json(), params=self.params,
                             headers={"content-type": "application/json"})
        try:
            body = r.json() if r.content else {}
        except ValueError:
            body = {}
        return body if isinstance(body, dict) else {}

    async def tools_bundle(self, swarm: str, version: str, *, budget: float = 60.0) -> bytes:
        r = await self._send("GET", f"/swarms/{swarm}/tools/{version}", budget=budget,
                             timeout=httpx.Timeout(20, read=120))
        return r.content

    async def events(self, batch: EventBatch, *, budget: float = 30.0) -> None:
        await self._send("POST", f"/workers/{batch.worker}/events", budget=budget,
                         content=batch.model_dump_json(), params=self.params,
                         headers={"content-type": "application/json"})

    async def forward(self, method: str, path: str, token: str, *, params: dict | None = None,
                      body: object = None, budget: float = 60.0) -> httpx.Response:
        """An agent's own call (controller_call): sent with the agent's token
        instead of the worker's, retried and re-resolved like every route here; the answer is
        returned whatever its status. Only a GET is retried after the request may have reached
        the controller, so a prompt is never delivered twice.
        503 counts as never processed: the Space edge's answer, and the controller's only for
        event posts, before any work (never for these calls), so resending it is safe."""
        try:
            return await self._send(method, path, budget=budget, retry_sent=method == "GET",
                                    params=params, json=body,
                                    headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPStatusError as e:
            return e.response

    async def aclose(self) -> None:
        await self.http.aclose()


_SEG = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
# What an agent may call through its worker (controller_call): the routes an agent token may use
# (flock.controller.spawn.agent_may, which still checks the subtree on the controller).
AGENT_ROUTES = tuple((m, re.compile(p)) for m, p in (
    ("GET", r"/whoami"),
    ("GET", rf"/agents/{_SEG}"),
    ("GET", rf"/agents/{_SEG}/events"),
    ("POST", rf"/agents/{_SEG}/spawn"),
    ("POST", rf"/agents/{_SEG}/prompt"),
    ("POST", rf"/agents/{_SEG}/children/prompts"),
    ("DELETE", rf"/agents/{_SEG}"),
    ("GET", rf"/swarms/{_SEG}/agents"),
))


@dataclass
class Result:
    """A controller_call's answer: `output` is `{status, body}` or `{error}` as JSON."""

    output: str
    error: bool
    duration_s: float = 0.0


def agent_route_allowed(method: str, path: str) -> bool:
    return any(m == method and p.fullmatch(path) for m, p in AGENT_ROUTES)


MEMORY_INTERVAL = 600.0  # s, FLOCK_MEMORY_LOG_S: heap trim and the memory lines in the log
MEMORY_SAMPLE = 15.0  # s, cgroup sample for the heartbeat
ARENAS = 4  # glibc malloc arenas unless MALLOC_ARENA_MAX is set (see limit_malloc_arenas)


def _glibc():
    import ctypes

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return libc if hasattr(libc, "malloc_trim") else None
    except OSError:
        return None


def rss_mb() -> float | None:
    """This process's resident set (VmRSS), in MiB; None off Linux."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def trim_heap() -> None:
    """Give memory freed in the C heap back to the system (glibc `malloc_trim`, every arena).
    Large transient buffers (JSON of message entries, bucket listings in sync threads) leave
    freed pages in the heap that the allocator otherwise keeps for the life of the process."""
    if (libc := _glibc()) is not None:
        libc.malloc_trim(0)


def limit_malloc_arenas(env: Mapping[str, str] = os.environ, arenas: int = ARENAS) -> bool:
    """Cap glibc's malloc arenas (default 8 per core) before threads start: each executor
    thread (bucket syncs, file I/O) would otherwise keep its own arena's high-water mark. An
    explicit MALLOC_ARENA_MAX wins (glibc reads it itself)."""
    if env.get("MALLOC_ARENA_MAX") or (libc := _glibc()) is None:
        return False
    return libc.mallopt(-8, arenas) == 1  # M_ARENA_MAX


def knobs_rev(cfg: WorkerConfig, mem: Mapping[str, float] | None = None) -> str:
    """A short hash of the settings this worker runs with (shown next to `spec_rev`)."""
    knobs = [cfg.drain_budget, cfg.sync_interval, workdir_excludes(), restore_batch(),
             restore_threads(), sorted((mem or {}).items())]
    return hashlib.sha256(json.dumps(knobs).encode()).hexdigest()[:12]


# `worker.memory` keys of the type spec -> the env names the memory watchdog reads
MEM_KNOB_ENV = {"rotate": "FLOCK_MEM_ROTATE", "drain": "FLOCK_MEM_DRAIN",
                "preempt": "FLOCK_MEM_PREEMPT"}


def default_adapter_factory(cfg: WorkerConfig, harbor: HarborHost | None = None
                            ) -> Callable[[str, TypeSpec], HarnessAdapter]:
    def make(name: str, spec: TypeSpec) -> HarnessAdapter:
        kw = {"host": harbor} if spec.harness.value == "harbor" else {}
        return load(HARNESSES[spec.harness].adapter)(cfg.work_root / name, name, spec, **kw)

    return make


STALE_STATUSES = (404, 409)


def _stale(e: Exception) -> bool:
    return isinstance(e, httpx.HTTPStatusError) and e.response.status_code in STALE_STATUSES


def _backoff(attempt: int) -> float:
    return min(60.0, 2 ** min(attempt, 6)) * (0.5 + random.random() / 2)


class Worker:
    def __init__(
        self,
        cfg: WorkerConfig,
        client: ControllerClient,
        *,
        adapter_factory: Callable[[str, TypeSpec], HarnessAdapter] | None = None,
        sync: BucketSync | None = None,
        tools: ToolsManager | None = None,
        cgroup: CgroupMem | None = None,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.tools = tools or ToolsManager(tools_root(cfg.work_root), fetch=self._fetch_tools)
        # harness harbor: what its agents' Trials share on this worker (flock.worker.harbor)
        self.harbor = HarborHost(cfg.work_root / "_harbor" / "_worker", emit=self._resource_event,
                                 folders=self.tools.folder, controller=self.controller_call)
        self.adapter_factory = adapter_factory or default_adapter_factory(cfg, self.harbor)
        self.sync = sync
        self.outbox = Outbox()
        self.agents: dict[str, AgentRuntime] = {}
        self.draining = False
        self.drain_window = cfg.drain_window
        self._drained = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()
        self._wake_heartbeat = asyncio.Event()
        self._unassigning: dict[str, asyncio.Task] = {}
        self.stale = False
        self._restoring = 0  # setups (restores) in flight; a "wave" ends when it drops to 0
        self._wave: tuple[float, int, dict] | None = None  # (start, agents, stats snapshot)
        self._throttled_logged = 0
        self._terminating = False
        self.preempted = False  # the preemption handler started (no new turns)
        self.preempt_synced = False  # it finished: the heartbeat may say `preempted` now
        self._drain_done = False  # drain() finished: every agent synced (or named in unsynced)
        self.drained = False  # the last heartbeat says so (set once nothing else will be sent)
        self._interrupted: set[str] = set()  # agents whose turn a preemption cut
        self._deadline = 0.0  # loop time by which a preempted worker has exited
        self.spec_rev: str | None = None  # the type spec revision we have (docs/live.md)
        self.cgroup = cgroup or CgroupMem()
        self.mem: WorkerMem | None = None  # the last cgroup sample (memory_loop)
        self.wants_rotation = False  # heartbeat: the controller may drain us (rotation budget)
        self._rung = 0  # highest ladder rung logged in this episode (0: below `rotate`)
        self._oom_kills: int | None = None
        self._env_cfg = dataclasses.replace(cfg)  # env/defaults, under the spec's `worker:`
        self.mem_knobs: dict[str, float] = {}  # FLOCK_MEM_* name -> the spec's value
        self.knobs_rev = knobs_rev(cfg)

    def apply_knobs(self, knobs: WorkerKnobs | None) -> None:
        """The type's `worker:` block (assign and live spec): each value set wins over the
        worker's env, which wins over the default. Takes effect at once (sync and drain read
        `cfg` when they run; the sync excludes are updated here)."""
        k, env, cfg = knobs or WorkerKnobs(), self._env_cfg, self.cfg
        pick = (lambda v, e: e if v is None else v)
        cfg.drain_budget = pick(k.drain_budget_s, env.drain_budget)
        cfg.sync_interval = pick(k.sync_interval_s, env.sync_interval)
        sync_mod.SPEC_KNOBS = {n: v for n in ("sync_exclude", "restore_batch", "restore_threads")
                               if (v := getattr(k, n)) is not None}  # one rebind: no empty window
        self.mem_knobs = {MEM_KNOB_ENV[n]: v for n, v in
                          (k.memory.model_dump() if k.memory else {}).items() if v is not None}
        rev = knobs_rev(cfg, self.mem_knobs)
        if rev != self.knobs_rev:
            log.info("worker knobs %s: %s", rev, k.model_dump(exclude_none=True) or "env/defaults")
            self.knobs_rev = rev

    def mem_knob(self, name: str, default: float) -> float:
        """A memory threshold by its env name (MEM_KNOB_ENV): the spec's `worker.memory`, else
        the env variable, else `default`."""
        if name in self.mem_knobs:
            return self.mem_knobs[name]
        try:
            return float(os.environ[name])
        except (KeyError, ValueError):
            return default

    async def _fetch_tools(self, version: str) -> bytes:
        return await self.client.tools_bundle(self.cfg.swarm, version)

    async def controller_call(self, agent: str, entry: str, args: dict) -> Result:
        """`controller:request {method, path, params?, json?}` from the agent's `spawn.*` tools
        (the Harbor spawn resource, docs/spawn.md): an allowlisted route, sent with the agent's
        own token by this worker's client, which follows the controller when it moves. The
        result is `{status, body}`; a call that still fails after retries emits
        `controller_call_failed {path, error}` on the agent's stream."""
        t0 = time.monotonic()
        method, path = str(args.get("method") or "").upper(), str(args.get("path") or "")
        rt = self.agents.get(agent)
        token = rt.flock_env.get("FLOCK_AGENT_TOKEN") if rt is not None else None

        def fail(error: str) -> Result:
            return Result(json.dumps({"error": error}), True, time.monotonic() - t0)

        if entry != "controller:request" or not agent_route_allowed(method, path):
            return fail(f"controller: {method} {path} is not allowed for an agent")
        if rt is None or not token:
            return fail("controller: this agent has no flock token on this worker")
        params, body = args.get("params"), args.get("json")
        try:
            r = await self.client.forward(method, path, token,
                                          params=params if isinstance(params, dict) else None,
                                          body=body)
        except httpx.HTTPError as e:
            error = f"{type(e).__name__}: {e}"
            rt.emit("controller_call_failed", {"path": path, "error": error})
            return fail(f"controller unreachable: {error}")
        if r.status_code >= 500:
            rt.emit("controller_call_failed", {"path": path, "error": f"HTTP {r.status_code}"})
        try:
            out = r.json() if r.content else None
        except ValueError:
            out = r.text
        return Result(json.dumps({"status": r.status_code, "body": out}), r.status_code >= 400,
                      time.monotonic() - t0)

    def _resource_event(self, agent: str, event: dict) -> None:
        """A harbor resource's event for one of this worker's agents (a board mention): the
        agent's `resource` event, which the controller turns into `<resource>.<kind>` wakes."""
        if (rt := self.agents.get(agent)) is not None:
            rt.emit("resource", event)

    def working_count(self) -> int:
        return sum(1 for a in self.agents.values() if a.live)

    # ------------------------------------------------------------------ commands
    async def dispatch(self, cmd: Command) -> None:
        if isinstance(cmd, AssignCommand):
            if pending := self._unassigning.get(cmd.agent):  # re-assign right after unassign
                await asyncio.shield(pending)
            self.apply_knobs(cmd.spec.worker)
            if cmd.agent in self.agents:
                self.agents[cmd.agent].set_live(spec=cmd.spec)  # at its next turn boundary
                self._set_lineage(self.agents[cmd.agent], cmd)
                return
            if self.draining:
                log.warning("assign %s ignored while draining", cmd.agent)
                return
            rt = AgentRuntime(cmd.agent, cmd.type, cmd.spec, swarm=self.cfg.swarm,
                              work_root=self.cfg.work_root,
                              adapter=self.adapter_factory(cmd.agent, cmd.spec), outbox=self.outbox,
                              sync=self.sync, working_count=self.working_count)
            self._set_lineage(rt, cmd)
            self.agents[cmd.agent] = rt
            self.spec_rev = type_spec_rev(cmd.spec)
            self.drain_window = parse_duration(cmd.spec.drain_window)
            self._spawn(self._setup(rt, cmd))
        elif isinstance(cmd, PromptCommand):
            rt = self.agents.get(cmd.agent)
            if rt is None:
                log.warning("prompt %s for unknown agent %s dropped", cmd.prompt_id, cmd.agent)
                return
            await rt.prompt(cmd.text, prompt_id=cmd.prompt_id, timeout=cmd.timeout, loop=cmd.loop)
        elif isinstance(cmd, LoopStartCommand | LoopStopCommand):
            rt = self.agents.get(cmd.agent)
            if rt is None:
                log.warning("%s for unknown agent %s dropped", cmd.kind, cmd.agent)
            elif isinstance(cmd, LoopStartCommand):
                rt.start_loop(cmd)
            else:
                rt.stop_loop(cmd.reason)
        elif isinstance(cmd, StopCommand):
            if rt := self.agents.get(cmd.agent):
                self._spawn(rt.stop())
        elif isinstance(cmd, UnassignCommand):
            rt = self.agents.get(cmd.agent)
            if rt is not None and cmd.agent not in self._unassigning:
                self._unassigning[cmd.agent] = self._spawn(self._unassign(rt))
        elif isinstance(cmd, DrainCommand):
            self._spawn(self.drain("command"))

    def _set_lineage(self, rt: AgentRuntime, cmd: AssignCommand) -> None:
        """What the agent's harness needs to call the controller as itself (docs/spawn.md):
        its own scoped token, never the worker's FLOCK_TOKEN."""
        rt.parent = cmd.parent
        rt.flock_env = {
            "FLOCK_URL": self.cfg.url,
            "FLOCK_AGENT": cmd.agent,
            "FLOCK_PARENT": cmd.parent or "",
            "FLOCK_AGENT_TOKEN": cmd.agent_token or "",
            # Masks the worker's token in the environment the adapters inherit (the harness env
            # allow-list also drops it).
            "FLOCK_TOKEN": "",
        }

    async def _setup(self, rt: AgentRuntime, cmd: AssignCommand) -> None:
        """rt.setup, timed as part of a restore wave (all setups that overlap in time)."""
        loop = asyncio.get_running_loop()
        if self._wave is None:
            snap = self.sync.stats.snapshot() if self.sync else {}
            self._wave = (loop.time(), 0, snap)
        start, n, snap = self._wave
        self._wave = (start, n + 1, snap)
        self._restoring += 1
        try:
            await rt.setup(restore=cmd.restore, resume_prompt=cmd.resume_prompt)
        finally:
            self._restoring -= 1
            if self._restoring == 0 and self._wave is not None:
                start, n, snap = self._wave
                self._wave = None
                calls = self.sync.stats.since(snap) if self.sync else {}
                log.info("restore wave: %d agents set up in %.1f s; bucket %s", n,
                         loop.time() - start, calls)

    async def _unassign(self, rt: AgentRuntime) -> None:
        try:
            await rt.unassign()
            if rt.spec.ephemeral:  # docs/spawn.md "Ephemeral types": nothing stays behind
                rt.purge()
        finally:
            self.agents.pop(rt.name, None)
            self._unassigning.pop(rt.name, None)

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    # ------------------------------------------------------------------ loops
    async def poll_loop(self) -> None:
        attempt = 0
        loop = asyncio.get_running_loop()
        while not self._drained.is_set():
            t0 = loop.time()
            try:
                batch = await self.client.commands(self.cfg.worker_id, self.cfg.poll_wait)
                attempt = 0
            except (httpx.HTTPError, ValueError) as e:
                if _stale(e):
                    self._spawn(self.abandon(f"commands: {e}"))
                    return
                attempt += 1
                log.warning("command poll failed (%s); retrying", e)
                await asyncio.sleep(_backoff(attempt))
                continue
            for cmd in batch.commands:
                try:
                    await self.dispatch(cmd)
                except Exception:
                    log.exception("command %s failed", cmd.kind)
            # at most one poll per poll_min_interval, also when commands keep coming: prompts
            # that arrive meanwhile ride in the next batch (fewer requests through the edge)
            await asyncio.sleep(max(0.0, self.cfg.poll_min_interval - (loop.time() - t0)))

    def heartbeat_payload(self) -> WorkerHeartbeat:
        return WorkerHeartbeat(worker=self.cfg.worker_id, deadline=self.cfg.deadline,
                               draining=self.draining, spec_rev=self.spec_rev,
                               preempted=self.preempt_synced, drained=self.drained,
                               interrupted=sorted(self._interrupted),
                               agents=[a.heartbeat() for a in self.agents.values()],
                               mem=self.mem, restore=self.restore_progress(),
                               knobs_rev=self.knobs_rev,
                               wants_rotation=self.wants_rotation,
                               unsynced={a.name: a.unsynced for a in self.agents.values()
                                         if a.unsynced})

    def restore_progress(self) -> RestoreProgress | None:
        """Restores in flight, summed over agents (BucketSync.restore_progress)."""
        prog = list(getattr(self.sync, "restore_progress", {}).values())
        if not prog:
            return None
        return RestoreProgress(agents=len(prog), files_done=sum(p[0] for p in prog),
                               files_total=sum(p[1] for p in prog),
                               started=min(p[2] for p in prog))

    async def send_heartbeat(self) -> bool:
        if (refresh := getattr(self.client, "refresh_url", None)) is not None:
            refresh()  # a controller move is picked up here even before any call fails
        try:  # retried inside the client for most of an interval: the controller declares a
            # worker dead only after worker_dead_after without any heartbeat, poll or event post
            resp = await self.client.heartbeat(self.heartbeat_payload(),
                                               budget=max(1.0, self.cfg.heartbeat_interval * 2 / 3))
            if isinstance(resp, dict) and not self.draining:
                self._live_from(resp)
            return True
        except httpx.HTTPError as e:
            log.warning("heartbeat failed: %s", e)
            if _stale(e) and not self.stale:
                self._spawn(self.abandon(f"heartbeat: {e}"))
            return False

    def _live_from(self, resp: dict) -> None:
        """A changed type spec (docs/live.md): every agent takes it at its next turn boundary."""
        raw = resp.get("type_spec")
        if isinstance(raw, dict):
            try:
                spec = TypeSpec.model_validate(raw)
            except ValueError as e:
                log.warning("heartbeat: unreadable type spec: %s", e)
            else:
                self.spec_rev = resp.get("spec_rev") or type_spec_rev(spec)
                self.apply_knobs(spec.worker)
                for rt in self.agents.values():
                    rt.set_live(spec)

    async def heartbeat_loop(self) -> None:
        await asyncio.sleep(random.random() * self.cfg.heartbeat_interval)
        while True:
            await self.send_heartbeat()
            self._wake_heartbeat.clear()
            try:  # +-10% jitter keeps 40+ workers from lining up on the edge
                await asyncio.wait_for(self._wake_heartbeat.wait(),
                                       self.cfg.heartbeat_interval * random.uniform(0.9, 1.1))
            except TimeoutError:
                pass
            stats = getattr(self.client, "stats", None) or {}
            if (n := stats.get("throttled", 0)) > self._throttled_logged:
                log.warning("controller throttled %d requests so far (429); %d retries",
                            n, stats.get("retries", 0))
                self._throttled_logged = n

    async def flush_events(self) -> bool:
        while len(self.outbox):
            batch = self.outbox.take()
            try:
                await self.client.events(EventBatch(worker=self.cfg.worker_id, events=batch))
            except httpx.HTTPError as e:
                self.outbox.put_back(batch)
                log.warning("event flush failed: %s", e)
                return False
        return True

    async def flush_loop(self) -> None:
        attempt = 0
        while True:
            await asyncio.sleep(self.cfg.flush_interval if attempt == 0 else _backoff(attempt))
            attempt = 0 if await self.flush_events() else attempt + 1

    async def sync_loop(self) -> None:
        if self.sync is None:
            return
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(self.cfg.sync_interval)
            # an agent still in setup() (restoring) is never pushed: it would mirror a
            # half-restored workdir over its bucket copy (flock.worker.sync, "Push safety")
            due = [rt for rt in list(self.agents.values())
                   if rt.setup_done and (rt.live or rt.dirty)]
            if not due:
                continue
            t0, snap = loop.time(), self.sync.stats.snapshot()
            for rt in due:
                await rt.sync_now("periodic")
            log.info("sync cycle: %d of %d agents in %.1f s; bucket %s", len(due), len(self.agents),
                     loop.time() - t0, self.sync.stats.since(snap))

    def memory_report(self) -> dict:
        """What the worker holds that can grow, for the periodic memory line."""
        return {"outbox_events": len(self.outbox), "outbox_bytes": self.outbox.bytes,
                "outbox_dropped": self.outbox.dropped, "agents": len(self.agents),
                "live": self.working_count(), "tasks": len(asyncio.all_tasks())}

    def memory_start(self) -> None:
        """Once: where the memory numbers come from, and a warning for a work root in RAM."""
        cg = self.cgroup
        log.info("memory: cgroup v%d %s, limit %s MB%s", cg.version, cg.path, cg.limit_mb,
                 "" if cg.armed else " (no limit found: memory is reported, nothing is armed)")
        # shortcut: no per-agent cgroups; under v2 they need the worker to move itself into a
        # leaf first, and their oom_kills would roll up into this cgroup's memory.events
        log.info("memory: memory.oom.group=%s", cg.oom_group())
        if cg.oom_group() == "1":
            log.warning("memory.oom.group is 1: one OOM kill ends every process in the job; "
                        "only the relief ladder helps")
        if (kind := fs_type(self.cfg.work_root)) in TMPFS_TYPES:
            log.warning("work root %s is on %s: every workdir file counts as memory",
                        self.cfg.work_root, kind)

    async def memory_loop(self) -> None:
        """Every MEMORY_SAMPLE s: sample the cgroup for the heartbeat. Every MEMORY_INTERVAL
        (FLOCK_MEMORY_LOG_S, 0 turns it off): trim the C heap, log the RSS before and after
        with the cgroup sample and the sizes of the worker's bounded queues."""
        interval = float(os.environ.get("FLOCK_MEMORY_LOG_S") or MEMORY_INTERVAL)
        loop = asyncio.get_running_loop()
        self.memory_start()
        next_log = loop.time() + interval
        while True:
            try:
                self.mem = WorkerMem(**await asyncio.to_thread(self.cgroup.sample))
                await self.relieve()
                if interval > 0 and loop.time() >= next_log:
                    next_log = loop.time() + interval
                    before = rss_mb()
                    await asyncio.to_thread(trim_heap)
                    after = rss_mb()
                    if before is not None and after is not None:
                        log.info("memory: rss %.0f MiB (%.0f MiB before heap trim); cgroup %s; %s",
                                 after, before, self.mem.model_dump(exclude_none=True),
                                 self.memory_report())
            except Exception:  # one bad reading must not end the loop
                log.exception("memory sample failed")
                self.mem = None  # no stale reading in later heartbeats
            await asyncio.sleep(MEMORY_SAMPLE)

    async def relieve(self) -> None:
        """One step of the memory relief ladder on the last sample (see the module doc). Each
        rung is logged once per episode (until memory falls below `rotate`). `worker_lifetime`
        stays the backstop."""
        th, mem = MemThresholds.read(self.mem_knob), self.mem
        if not self.cgroup.armed or mem is None or not mem.limit_mb or mem.nonreclaim_mb is None:
            return
        frac = mem.nonreclaim_mb / mem.limit_mb
        oom = None not in (mem.oom_kills, self._oom_kills) and mem.oom_kills > self._oom_kills
        if mem.oom_kills is not None:
            self._oom_kills = mem.oom_kills
        rung = (3 if oom or frac >= th.preempt else 2 if frac >= th.drain
                else 1 if frac >= th.rotate else 0)
        if rung > self._rung:
            what = ("", "rotation wanted",
                    f"draining (turns cut after {th.drain_deadline_s:.0f} s)",
                    "oom_kill: preempting" if oom else "preempting")[rung]
            log.warning("memory: nonreclaim %.0f of %.0f MB (%.0f%%): %s", mem.nonreclaim_mb,
                        mem.limit_mb, frac * 100, what)
        self._rung = max(self._rung, rung) if rung else 0
        self.wants_rotation = rung >= 1
        if rung == 3:
            self._spawn(self.preempt("memory"))
        elif rung == 2 and not self.draining:
            self._spawn(self.drain("memory", deadline=th.drain_deadline_s))

    async def deadline_loop(self) -> None:
        while True:
            left = (self.cfg.deadline - self.drain_window - utcnow()).total_seconds()
            if left <= 0:
                break
            await asyncio.sleep(min(left, 30))
        await self.drain("deadline")

    async def parent_loop(self, every: float = 2.0) -> None:
        """compute local: a worker whose controller process died (kill -9, a crash) would poll
        a dead URL until its deadline. Exit like a stale slot instead: stop, no sync."""
        if self.cfg.parent_pid is None:
            return
        while os.getppid() == self.cfg.parent_pid:
            await asyncio.sleep(every)
        await self.abandon(f"controller process {self.cfg.parent_pid} is gone")

    async def terminate(self, reason: str) -> None:
        """compute local: SIGTERM from the local run at stop. Interrupt every turn now (the
        adapters kill their harness and tool processes), then exit through the normal path:
        flush events, last heartbeat. No sync (a local worker has no bucket)."""
        if self._terminating or self.stale:
            return
        log.info("terminating (%s)", reason)
        self._terminating = self.draining = True
        for rt in self.agents.values():
            rt.allow_starts = False
        await self._stop_all("terminated")
        self._drained.set()

    async def _stop_all(self, reason: str) -> None:
        """Stop every agent and cancel what outlives it (harbor Trials, unverified, also one
        still starting: stop waits for its start), all in parallel: no container or sandbox
        outlives this worker."""
        agents = list(self.agents.values())
        await asyncio.gather(*(rt.stop(reason) for rt in agents),
                             *(rt.close(0) for rt in agents), return_exceptions=True)

    async def preempt(self, reason: str) -> None:
        """The machine goes away in seconds (docs/slurm.md): everything below fits in
        `cfg.drain_budget`. Turns are interrupted, not finished; each agent's session and
        workdir go to the bucket in parallel; then the preempted marker. Events, the last
        heartbeat (`preempted: true`) and exit code PREEMPT_EXIT follow in `run`."""
        if self.preempted or self.stale:
            return
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        self._deadline = t0 + self.cfg.drain_budget
        log.warning("preempted (%s): interrupting %d agents, syncing within %.0f s", reason,
                    len(self.agents), self.cfg.drain_budget)
        # no early heartbeat: `preempted` tells the controller to relaunch, which must wait
        # until the bucket has this worker's last state
        self.preempted = self._terminating = self.draining = True
        if self.sync:
            self.sync.stop_rebase = True  # a workdir base build must not eat the budget
        agents = list(self.agents.values())
        for rt in agents:
            rt.allow_starts = False
        # syncs run in threads: one per agent, not the default pool's few
        from concurrent.futures import ThreadPoolExecutor

        loop.set_default_executor(ThreadPoolExecutor(max_workers=max(8, len(agents) + 4),
                                                     thread_name_prefix="preempt"))
        # leave time for the events and the last heartbeat (a few seconds)
        budget = max(1.0, self.cfg.drain_budget * 0.8)

        async def one(rt: AgentRuntime) -> None:
            if rt.turn_id is not None or rt.awaiting:
                self._interrupted.add(rt.name)
            # the Trial must not outlive the machine: cancelled now, unverified (also one still
            # starting: stop waits for its start); Harbor pulls its sessions as it stops it
            closing = asyncio.ensure_future(rt.close(0))
            await rt.stop("preempted")
            await asyncio.wait({closing}, timeout=max(0.5, (t0 + budget - loop.time()) / 2))
            await rt.sync_now("preempt")
            await closing

        snap = self.sync.stats.snapshot() if self.sync else {}
        try:
            await asyncio.wait_for(asyncio.gather(*(one(rt) for rt in agents),
                                                  return_exceptions=True), budget)
        except TimeoutError:
            log.warning("preempt: sync did not finish in %.0f s; exiting anyway", budget)
        await self._mark_preempted(reason, max(0.5, self._deadline - loop.time() - 2))
        log.info("preempt: %d agents interrupted and synced in %.1f s; bucket %s", len(agents),
                 loop.time() - t0, self.sync.stats.since(snap) if self.sync else {})
        self.preempt_synced = True
        self._drained.set()

    async def _mark_preempted(self, reason: str, timeout: float) -> None:
        """`<swarm>/_workers/<worker>/preempted.json` in the bucket: who, when, why."""
        if self.sync is None:
            return
        import tempfile

        body = json.dumps({"worker": self.cfg.worker_id, "reason": reason,
                           "at": utcnow().isoformat(), "agents": sorted(self.agents)})
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(body)
        path = Path(f.name)
        dest = f"{self.cfg.swarm}/_workers/{self.cfg.worker_id}/preempted.json"
        try:
            await asyncio.wait_for(asyncio.to_thread(self.sync.store.upload, [(path, dest)]),
                                   timeout)
        except Exception as e:  # noqa: BLE001 - the heartbeat flag still tells the controller
            log.warning("preempt marker not written: %s", type(e).__name__)
        finally:
            path.unlink(missing_ok=True)

    async def drain(self, reason: str, deadline: float | None = None) -> None:
        """Finish the turns in flight, sync, exit 0. With `deadline` (s), turns still running
        then are interrupted; they resume after the restart (heartbeat `interrupted`)."""
        if self.draining:
            return
        log.info("draining (%s)", reason)
        self.draining = True
        if self.sync:
            self.sync.stop_rebase = True
        self._wake_heartbeat.set()
        loop = asyncio.get_running_loop()
        t0, snap = loop.time(), (self.sync.stats.snapshot() if self.sync else {})
        agents = list(self.agents.values())
        cut = await asyncio.gather(*(rt.drain(deadline) for rt in agents), return_exceptions=True)
        self._interrupted.update(rt.name for rt, c in zip(agents, cut, strict=True) if c is True)
        for rt, c in zip(agents, cut, strict=True):
            if isinstance(c, BaseException):  # the drain itself failed: its sync is not known
                rt.unsynced = f"drain failed: {type(c).__name__}: {c}"[:300]
        unsynced = [rt.name for rt in agents if rt.unsynced]
        log.log(logging.WARNING if unsynced else logging.INFO,
                "drain: %d agents finished in %.1f s, %s; bucket %s", len(self.agents),
                loop.time() - t0, f"NOT synced: {', '.join(unsynced)}" if unsynced else "all synced",
                self.sync.stats.since(snap) if self.sync else {})
        self._drain_done = True
        self._drained.set()

    async def abandon(self, why: str) -> None:
        """Stale slot: kill harness processes, no sync, exit 0."""
        if self.stale:
            return
        log.warning("controller no longer knows this worker (%s); exiting", why)
        self.stale = self.draining = True
        for rt in self.agents.values():
            rt.sync = None
            rt.allow_starts = False
        await self._stop_all("abandoned")
        self._drained.set()

    async def run(self) -> int:
        loops = [self.poll_loop(), self.heartbeat_loop(), self.flush_loop(), self.sync_loop(),
                 self.deadline_loop(), self.parent_loop(), self.memory_loop()]
        tasks = [asyncio.create_task(c) for c in loops]
        loop = asyncio.get_running_loop()
        # compute local: SIGTERM is the local run stopping (terminate). Elsewhere SIGTERM and
        # SIGUSR1 mean the machine goes away (a Spot notice, a time limit, a cancel): preempt.
        local = self.cfg.parent_pid is not None
        handlers = {
            signal.SIGUSR1: lambda: self._spawn(self.preempt("SIGUSR1")),
            signal.SIGTERM: (lambda: self._spawn(self.terminate("SIGTERM"))) if local
            else (lambda: self._spawn(self.preempt("SIGTERM"))),
        }
        for sig, fn in handlers.items():
            loop.add_signal_handler(sig, fn)
        try:
            await self._drained.wait()
        finally:
            for sig in handlers:
                loop.remove_signal_handler(sig)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if self.preempted:  # what is left of the budget, for events and the last heartbeat
            left = max(1.0, self._deadline - loop.time())
            try:
                await asyncio.wait_for(self._final_reports(), left)
            except TimeoutError:
                log.warning("preempt: events/heartbeat cut at the budget")
        else:
            # before `drained`: that heartbeat lets the controller cancel the job at once.
            # Not in a preemption's budget: the job ends anyway
            await self.harbor.close()
            # the loops are stopped: the heartbeat after the events flush is the last one
            self.drained = self._drain_done
            await self._final_reports()
        await self.client.aclose()
        log.info("drained; exiting%s", " (preempted)" if self.preempted else "")
        return PREEMPT_EXIT if self.preempted else 0

    async def _final_reports(self) -> None:
        for attempt in range(5):
            if self.stale or await self.flush_events():
                break
            await asyncio.sleep(0.2 if self.preempted else _backoff(attempt))
        if not self.stale:
            await self.send_heartbeat()


def quiet_libraries() -> None:
    """One line per poll/heartbeat and tqdm bars per upload would drown the job log. The env var
    (read by huggingface_hub and its xet uploads) works where disable_progress_bars() does not."""
    for name in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def run_and_exit(coro, *, exit: Callable[[int], object] = os._exit) -> None:
    """Run the worker to completion, then exit at once, without asyncio.run's teardown.

    By the time Worker.run returns, the worker has synced every agent, flushed its events and
    sent its last heartbeat, so nothing in the process is needed any more. asyncio.run would
    then cancel leftover tasks and join the default executor, and the interpreter would join
    non-daemon threads; any of them blocking (a hung bucket or SDK call) keeps the job RUNNING
    after a drain. Log what is still alive, then os._exit."""
    loop = asyncio.new_event_loop()
    code = loop.run_until_complete(coro)
    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
    alive = [t.name for t in threading.enumerate()
             if t is not threading.main_thread() and t.is_alive()]
    if pending or alive:
        log.info("exiting with %d pending tasks and %d live threads: %s", len(pending), len(alive),
                 sorted({(t.get_coro().__qualname__ if t.get_coro() else "?") for t in pending})[:10]
                 + sorted(set(alive))[:10])
    logging.shutdown()
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    exit(code)


def main() -> None:
    limit_malloc_arenas()  # before any thread starts
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    quiet_libraries()
    log.info("flock-worker on Python %s (%s)", sys.version.split()[0], sys.executable)
    # FLOCK_WORKER_SECRETS_FILE: credentials from a 0600 file, not the exec env (docs/slurm.md)
    if names := load_secrets():
        log.info("secrets: %d read from FLOCK_WORKER_SECRETS_FILE (%s)", len(names),
                 ", ".join(names))
    cfg = WorkerConfig.from_env(secret_env())
    cfg.work_root.mkdir(parents=True, exist_ok=True)
    # a container with no (writable) HOME: HOME, caches and TMPDIR next to the work root
    # (/work/agents -> /work/home, /work/hf, /work/tmp, ...), never /tmp (docs/slurm.md)
    if set_ := homeless_defaults(cfg.work_root.parent):
        log.info("no usable HOME: %s", set_)
    sync = None
    if cfg.bucket:
        stats = BucketStats()
        count_hub_http(stats)
        sync = BucketSync(HubBucketStore(cfg.bucket, token=secret_env().get("HF_TOKEN") or None,
                                          stats=stats), cfg.swarm)
    resolve = None
    if cfg.url_template and cfg.host_file:
        template, host_file = cfg.url_template, cfg.host_file

        def resolve() -> str:
            return fill_host(template, "${CTL_HOST}", read_host_file(host_file))

    client = ControllerClient(cfg.url, cfg.token, incarnation=cfg.incarnation,
                              resolve_url=resolve)
    worker = Worker(cfg, client, sync=sync)
    run_and_exit(worker.run())


if __name__ == "__main__":
    main()
