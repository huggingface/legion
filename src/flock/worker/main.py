"""flock-worker: one HF Job hosting up to `agents_per_worker` agents.

Tasks: command long-poll, heartbeat (random initial phase), events flusher, periodic sync,
deadline watcher. Drain (`drain_window` before DEADLINE, or a drain command): stop starting
harness processes, let in-flight turns finish (bounded by turn_timeout), final sync per
agent, flush events, last heartbeat, exit 0. Controller errors are retried with backoff,
except 404/409 on our own worker routes: the controller says this slot has no live job (we
are a stale process and a replacement may already own our agents), so we kill the harness
processes and exit 0 without syncing, to avoid overwriting the replacement's bucket state.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from flock.models import (
    HEARTBEAT_INTERVAL,
    AssignCommand,
    Command,
    CommandBatch,
    DrainCommand,
    EventBatch,
    PromptCommand,
    StopCommand,
    TypeSpec,
    UnassignCommand,
    WorkerHeartbeat,
    parse_duration,
)
from flock.worker.adapters.base import HarnessAdapter
from flock.worker.agents import AgentRuntime
from flock.worker.events import Outbox, parse_iso, utcnow
from flock.worker.identity import Identities
from flock.worker.sync import BucketStats, BucketSync, HubBucketStore, count_hub_http

log = logging.getLogger("flock.worker")

DEFAULT_DRAIN_WINDOW = timedelta(minutes=10)


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

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> WorkerConfig:
        return cls(
            url=env["FLOCK_URL"].rstrip("/"),
            token=env["FLOCK_TOKEN"],
            worker_id=env["WORKER_ID"],
            swarm=env["SWARM"],
            type=env["TYPE"],
            deadline=parse_iso(env["DEADLINE"]),
            bucket=env.get("BUCKET") or None,
            work_root=Path(env.get("WORK_ROOT", "/work/agents")),
            sync_interval=float(env.get("SYNC_INTERVAL", 180)),
            heartbeat_interval=float(env.get("HEARTBEAT_INTERVAL", HEARTBEAT_INTERVAL.total_seconds())),
            drain_window=parse_duration(env.get("DRAIN_WINDOW") or DEFAULT_DRAIN_WINDOW),
            incarnation=env.get("WORKER_INCARNATION") or None,
        )


class ControllerClient:
    def __init__(self, url: str, token: str, *, transport: httpx.AsyncBaseTransport | None = None,
                 incarnation: str | None = None) -> None:
        self.http = httpx.AsyncClient(base_url=url, headers={"Authorization": f"Bearer {token}"},
                                      timeout=httpx.Timeout(20, read=60), transport=transport)
        self.params = {"incarnation": incarnation} if incarnation else {}

    async def commands(self, worker: str, wait: int) -> CommandBatch:
        r = await self.http.get(f"/workers/{worker}/commands", params={"wait": wait, **self.params},
                                timeout=httpx.Timeout(20, read=wait + 20))
        r.raise_for_status()
        return CommandBatch.model_validate(r.json() if r.content else {"commands": []})

    async def heartbeat(self, hb: WorkerHeartbeat) -> None:
        r = await self.http.post(f"/workers/{hb.worker}/heartbeat", content=hb.model_dump_json(),
                                 params=self.params, headers={"content-type": "application/json"})
        r.raise_for_status()

    async def events(self, batch: EventBatch) -> None:
        r = await self.http.post(f"/workers/{batch.worker}/events", content=batch.model_dump_json(),
                                 headers={"content-type": "application/json"})
        r.raise_for_status()

    async def aclose(self) -> None:
        await self.http.aclose()


def default_adapter_factory(cfg: WorkerConfig) -> Callable[[str, TypeSpec], HarnessAdapter]:
    def make(name: str, spec: TypeSpec) -> HarnessAdapter:
        workdir = cfg.work_root / name
        if spec.harness.value == "claude":
            from flock.worker.adapters.claude import ClaudeAdapter

            return ClaudeAdapter(workdir)
        if spec.harness.value == "pi":
            from flock.controller.prices import type_price_table
            from flock.worker.adapters.pi import PiAdapter, pi_model_config

            return PiAdapter(workdir, agent=name, model_config=pi_model_config(spec),
                             prices=type_price_table(spec))
        if spec.harness.value == "llm":
            from flock.controller.prices import type_price_table
            from flock.worker.adapters.llm import LLMAdapter

            return LLMAdapter(workdir, agent=name, prices=type_price_table(spec))
        from flock.worker.adapters.codex import CodexAdapter

        return CodexAdapter(workdir)

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
        identities: Identities | None = None,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.adapter_factory = adapter_factory or default_adapter_factory(cfg)
        self.sync = sync
        self.identities = identities or Identities()
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

    def working_count(self) -> int:
        return sum(1 for a in self.agents.values() if a.live)

    # ------------------------------------------------------------------ commands
    async def dispatch(self, cmd: Command) -> None:
        if isinstance(cmd, AssignCommand):
            if pending := self._unassigning.get(cmd.agent):  # re-assign right after unassign
                await asyncio.shield(pending)
            if cmd.agent in self.agents:
                self.agents[cmd.agent].spec = cmd.spec
                self._set_lineage(self.agents[cmd.agent], cmd)
                return
            if self.draining:
                log.warning("assign %s ignored while draining", cmd.agent)
                return
            rt = AgentRuntime(cmd.agent, cmd.type, cmd.spec, swarm=self.cfg.swarm,
                              work_root=self.cfg.work_root,
                              adapter=self.adapter_factory(cmd.agent, cmd.spec), outbox=self.outbox,
                              sync=self.sync, identities=self.identities,
                              working_count=self.working_count)
            self._set_lineage(rt, cmd)
            self.agents[cmd.agent] = rt
            self.drain_window = parse_duration(cmd.spec.drain_window)
            self._spawn(self._setup(rt, cmd))
        elif isinstance(cmd, PromptCommand):
            rt = self.agents.get(cmd.agent)
            if rt is None:
                log.warning("prompt %s for unknown agent %s dropped", cmd.prompt_id, cmd.agent)
                return
            await rt.prompt(cmd.text, prompt_id=cmd.prompt_id, timeout=cmd.timeout)
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
            # Masks the worker's token in the environment the adapters still inherit; the
            # harness env allow-list (PR #2) makes this redundant.
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
            if not batch.commands:
                await asyncio.sleep(max(0.0, self.cfg.poll_min_interval - (loop.time() - t0)))

    def heartbeat_payload(self) -> WorkerHeartbeat:
        return WorkerHeartbeat(worker=self.cfg.worker_id, deadline=self.cfg.deadline,
                               draining=self.draining,
                               agents=[a.heartbeat() for a in self.agents.values()])

    async def send_heartbeat(self) -> bool:
        try:
            await self.client.heartbeat(self.heartbeat_payload())
            return True
        except httpx.HTTPError as e:
            log.warning("heartbeat failed: %s", e)
            if _stale(e) and not self.stale:
                self._spawn(self.abandon(f"heartbeat: {e}"))
            return False

    async def heartbeat_loop(self) -> None:
        await asyncio.sleep(random.random() * self.cfg.heartbeat_interval)
        while True:
            await self.send_heartbeat()
            self._wake_heartbeat.clear()
            try:
                await asyncio.wait_for(self._wake_heartbeat.wait(), self.cfg.heartbeat_interval)
            except TimeoutError:
                pass

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
            due = [rt for rt in list(self.agents.values()) if rt.live or rt.dirty]
            if not due:
                continue
            t0, snap = loop.time(), self.sync.stats.snapshot()
            for rt in due:
                await rt.sync_now("periodic")
            log.info("sync cycle: %d of %d agents in %.1f s; bucket %s", len(due), len(self.agents),
                     loop.time() - t0, self.sync.stats.since(snap))

    async def deadline_loop(self) -> None:
        while True:
            left = (self.cfg.deadline - self.drain_window - utcnow()).total_seconds()
            if left <= 0:
                break
            await asyncio.sleep(min(left, 30))
        await self.drain("deadline")

    async def drain(self, reason: str) -> None:
        if self.draining:
            return
        log.info("draining (%s)", reason)
        self.draining = True
        self._wake_heartbeat.set()
        loop = asyncio.get_running_loop()
        t0, snap = loop.time(), (self.sync.stats.snapshot() if self.sync else {})
        await asyncio.gather(*(rt.drain() for rt in list(self.agents.values())), return_exceptions=True)
        log.info("drain: %d agents finished and synced in %.1f s; bucket %s", len(self.agents),
                 loop.time() - t0, self.sync.stats.since(snap) if self.sync else {})
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
        await asyncio.gather(*(rt.stop("abandoned") for rt in list(self.agents.values())),
                             return_exceptions=True)
        self._drained.set()

    async def run(self) -> int:
        loops = [self.poll_loop(), self.heartbeat_loop(), self.flush_loop(), self.sync_loop(),
                 self.deadline_loop()]
        tasks = [asyncio.create_task(c) for c in loops]
        try:
            await self._drained.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for attempt in range(5):
            if self.stale or await self.flush_events():
                break
            await asyncio.sleep(_backoff(attempt))
        if not self.stale:
            await self.send_heartbeat()
        await self.client.aclose()
        log.info("drained; exiting")
        return 0


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
    after a drain (1k run, check 5). Log what is still alive, then os._exit."""
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
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    quiet_libraries()
    cfg = WorkerConfig.from_env()
    cfg.work_root.mkdir(parents=True, exist_ok=True)
    sync = None
    if cfg.bucket:
        stats = BucketStats()
        count_hub_http(stats)
        sync = BucketSync(HubBucketStore(cfg.bucket, stats=stats), cfg.swarm)
    client = ControllerClient(cfg.url, cfg.token, incarnation=cfg.incarnation)
    worker = Worker(cfg, client, sync=sync)
    run_and_exit(worker.run())


if __name__ == "__main__":
    main()
