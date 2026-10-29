"""Controller entry point: `flock-controller`.

Two deployments (docs/deploy-job.md):
- `space` (default): an HF Docker Space; workers reach it through the Spaces edge at its
  public URL. Fine for small swarms.
- `job` (FLOCK_MODE=job): an HF Job in a network group under the alias `controller`. Workers
  join the same group and reach it directly at `http://${HF_NETWORK_GROUP_PREFIX}controller:7860`
  (the worker expands the variable). Humans use the jobs proxy: the API on 7860 (HF token
  gated) and a read-only status page on 7861 (public). The SQLite state is synced to a bucket.
"""

import asyncio
import contextlib
import logging
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass

from starlette.responses import PlainTextResponse

from flock.controller.api import Notifier, create_app
from flock.controller.jobs import FakeJobsClient, HubJobsClient, JobsClient
from flock.controller.reconcile import Settings, run_reconciler
from flock.controller.state import DEFAULT_INTERVAL, StateSync, state_sync_from_env
from flock.controller.store import Store

log = logging.getLogger("flock.controller")
STATS_LOG_INTERVAL = 60.0
DEFAULT_ALIAS = "controller"
DEFAULT_STATUS_PORT = 7861
STATUS_ROUTES = frozenset({"/", "/healthz"})
GRACEFUL_SHUTDOWN = 5.0  # s for open requests (parked long-polls) once listeners are closed
FINAL_PUSH_TIMEOUT = 30.0  # s; FLOCK_FINAL_PUSH_TIMEOUT overrides


@dataclass
class Deployment:
    mode: str  # "space" (also local runs) or "job"
    port: int
    worker_url: str  # FLOCK_URL handed to workers
    public_url: str  # where humans and the CLI reach the API
    status_port: int | None = None  # read-only status app (`/`, `/healthz`), job mode
    status_url: str | None = None
    network_group: str | None = None
    namespace: str | None = None  # job mode: the only namespace specs may launch workers in


def _space_url(env: Mapping[str, str], port: int) -> str:
    if env.get("FLOCK_URL"):
        return env["FLOCK_URL"]
    if env.get("SPACE_HOST"):  # set by HF Spaces
        return f"https://{env['SPACE_HOST']}"
    return f"http://localhost:{port}"


def jobs_proxy_url(job_id: str, port: int) -> str:
    """The HF Jobs proxy URL of an exposed port (format from the Jobs docs)."""
    return f"https://{job_id}--{port}.hf.jobs"


def deployment(env: Mapping[str, str] = os.environ) -> Deployment:
    port = int(env.get("PORT", "7860"))
    mode = env.get("FLOCK_MODE", "space")
    if mode != "job":
        if mode != "space":
            raise SystemExit(f"FLOCK_MODE must be 'space' or 'job', not {mode!r}")
        status = int(env["FLOCK_STATUS_PORT"]) if env.get("FLOCK_STATUS_PORT") else None
        url = _space_url(env, port)
        # FLOCK_WORKER_URL: what workers are told, when it differs from the public URL, e.g.
        # `http://${CTL_HOST}:7860` on Slurm (each worker expands it from its controller host
        # file, so a controller that moves host does not relaunch the fleet; docs/slurm.md)
        worker = env.get("FLOCK_WORKER_URL") or url
        return Deployment("space", port, worker, url, status_port=status)
    group = env.get("FLOCK_NETWORK_GROUP", "")
    if not group:
        raise SystemExit("FLOCK_MODE=job needs FLOCK_NETWORK_GROUP (the workers' network group)")
    alias = env.get("FLOCK_NETWORK_ALIAS", DEFAULT_ALIAS)
    status_port = int(env.get("FLOCK_STATUS_PORT", DEFAULT_STATUS_PORT))
    # JOB_ID is set by HF Jobs in every job container; the proxy URL follows from it, so the
    # controller needs no API call and no second launch step to know its own address.
    job_id = env.get("JOB_ID", "")
    public = env.get("FLOCK_PUBLIC_URL") or (
        jobs_proxy_url(job_id, port) if job_id else f"http://localhost:{port}"
    )
    status_url = jobs_proxy_url(job_id, status_port) if job_id else None
    # Literal `${HF_NETWORK_GROUP_PREFIX}`: the worker expands it from its own environment
    # (flock.worker.main.expand_env); HF Jobs sets it in every group member.
    worker_url = env.get("FLOCK_WORKER_URL") or f"http://${{HF_NETWORK_GROUP_PREFIX}}{alias}:{port}"
    return Deployment(
        "job", port, worker_url, public, status_port=status_port, status_url=status_url,
        network_group=group, namespace=env.get("FLOCK_NAMESPACE") or None,
    )


def parse_labels(raw: str) -> dict[str, str]:
    """'a=1,b=2' -> {'a': '1', 'b': '2'} (FLOCK_JOB_LABELS)."""
    out = {}
    for part in raw.split(","):
        if part.strip():
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
    return out


def status_only(app):
    """ASGI app that serves only `GET|HEAD /` and `/healthz` of `app` (no auth needed for
    those), 404 for everything else. For the public status port in job mode."""

    async def asgi(scope, receive, send):
        if scope["type"] == "lifespan":  # the main server owns the app's lifespan
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        ok = scope["type"] == "http" and scope.get("method") in ("GET", "HEAD")
        if ok and scope.get("path") in STATUS_ROUTES:
            await app(scope, receive, send)
            return
        await PlainTextResponse("not found (status port: / and /healthz only)", 404)(
            scope, receive, send
        )

    return asgi


@dataclass
class Built:
    app: object
    store: Store
    deployment: Deployment
    jobs: JobsClient
    state: StateSync | None
    settings: Settings
    halt: object = None  # () -> None: cancel the reconcile and sync loops (shutdown)
    final_push_timeout: float = FINAL_PUSH_TIMEOUT


WORLD_FOLLOW_INTERVAL = 15.0  # s between reads of FLOCK_WORLD_HOST_FILE
COMPUTE_KINDS = (None, "hf-jobs", "local", "slurm", "external")


def build(env: Mapping[str, str] = os.environ) -> Built:
    token = env.get("FLOCK_TOKEN", "")
    if not token:
        raise SystemExit("FLOCK_TOKEN must be set")
    kind = env.get("FLOCK_COMPUTE") or None
    if kind not in COMPUTE_KINDS:
        # never fall back to HF Jobs: a typo on a cluster would launch paid jobs
        raise SystemExit(f"FLOCK_COMPUTE={kind!r} is not one of "
                         f"{', '.join(k for k in COMPUTE_KINDS if k)} (unset: hf-jobs)")
    dep = deployment(env)
    data_dir = env.get("FLOCK_DATA", "./data")
    if env is os.environ:  # a container with no usable HOME: hub caches under the data dir
        from pathlib import Path

        from flock.home import cache_defaults

        cache_defaults(Path(data_dir).resolve() / "cache")
    state = state_sync_from_env(env, data_dir)
    if dep.mode == "job" and state is None:
        log.warning("job mode without FLOCK_STATE_URL: controller state is lost on restart")
    if state is not None and env.get("FLOCK_STATE_RESTORE", "1") != "0":
        state.restore()
    store = Store(data_dir)
    if state is not None and (fixed := store.repair_events()):
        log.info("state: trimmed %d event streams to the restored index", fixed)
    jobs: JobsClient
    compute: str | None = None  # the spec `compute.kind` this controller accepts
    if kind == "local":  # docs/local.md: workers as local subprocesses
        if dep.mode == "job":
            raise SystemExit("FLOCK_COMPUTE=local cannot run with FLOCK_MODE=job")
        from flock.controller.local import LocalJobsClient

        jobs, compute = LocalJobsClient(root=data_dir), "local"
    elif kind in ("slurm", "external"):  # docs/slurm.md
        from flock.controller.slurm import ExternalJobsClient, SlurmJobsClient

        compute = kind
        jobs = SlurmJobsClient(env=env) if compute == "slurm" else ExternalJobsClient(env=env)
    elif env.get("FLOCK_FAKE_JOBS") == "1":
        jobs = FakeJobsClient(network_group=dep.network_group)
    else:
        jobs = HubJobsClient(token=env.get("HF_TOKEN"), network_group=dep.network_group)
        compute = "hf-jobs"
    log.info("compute %s: %s", compute or ("fake" if env.get("FLOCK_FAKE_JOBS") == "1"
                                           else "hf-jobs"), type(jobs).__name__)
    settings = Settings(
        public_url=dep.worker_url,
        token=token,
        job_labels=parse_labels(env.get("FLOCK_JOB_LABELS", "")),
    )
    interval = float(env.get("FLOCK_RECONCILE_INTERVAL", "5"))
    sync_every = float(env.get("FLOCK_STATE_SYNC_INTERVAL", DEFAULT_INTERVAL))

    def stats() -> dict:
        calls = getattr(jobs, "calls", None)
        out: dict = {"mode": dep.mode, "public_url": dep.public_url}
        if dep.status_url:
            out["status_url"] = dep.status_url
        if calls is not None:
            out["hub_calls"] = dict(calls)
        if state is not None:
            out["state_pushes"] = state.pushes
        return out

    async def log_stats() -> None:
        while True:
            await asyncio.sleep(STATS_LOG_INTERVAL)
            log.info("hub calls: %s", stats().get("hub_calls", {}))

    async def follow_worlds() -> None:
        """FLOCK_WORLD_HOST_FILE (docs/slurm.md): a world job requeued on another node writes
        its new host there; specs with ${WORLD_HOST} in world.url get the new endpoint, and
        workers get it with their next heartbeat (same credential: no relaunch)."""
        from flock.world.host import follow_world_host

        while True:
            await asyncio.sleep(WORLD_FOLLOW_INTERVAL)
            for sw in store.list_swarms():
                try:
                    new = await asyncio.to_thread(follow_world_host, sw["spec"], token, env,
                                                    env=env)
                except Exception:
                    log.exception("world host check failed")
                    continue
                if new is not None:
                    store.update_swarm_spec(new)

    async def sync_state() -> None:
        while True:
            await asyncio.sleep(sync_every)
            try:
                await asyncio.to_thread(state.push, store)
            except Exception:
                log.exception("state push failed")

    loops: list[asyncio.Task] = []  # cancelled by `halt` on shutdown, before the final push

    async def background(notifier: Notifier) -> None:
        loops.append(asyncio.current_task())
        tasks = [asyncio.create_task(log_stats())]
        if env.get("FLOCK_WORLD_HOST_FILE"):
            tasks.append(asyncio.create_task(follow_worlds()))
        if state is not None:
            tasks.append(asyncio.create_task(sync_state()))
        loops.extend(tasks)
        try:
            await run_reconciler(
                store, jobs, settings, interval, wake=lambda t: notifier.wait("reconcile:", t)
            )
        finally:
            for t in tasks:
                t.cancel()

    def halt() -> None:
        for t in loops:
            t.cancel()

    app = create_app(store, token, background=background, stats=stats, namespace=dep.namespace,
                     compute=compute)
    timeout = float(env.get("FLOCK_FINAL_PUSH_TIMEOUT", FINAL_PUSH_TIMEOUT))
    return Built(app, store, dep, jobs, state, settings, halt, timeout)


def build_app():
    b = build()
    return b.app, b.deployment.port


async def final_push(b: Built) -> bool:
    """Last state push before exit, bounded by `b.final_push_timeout`. Always logs one line:
    `final push: ok (N files)`, `final push: failed (...)` or `final push: off`."""
    if b.state is None:
        log.info("final push: off (no FLOCK_STATE_URL)")
        return True
    try:
        n = await asyncio.wait_for(
            asyncio.to_thread(b.state.push, b.store, force=True, final=True), b.final_push_timeout
        )
    except TimeoutError:
        log.error("final push: failed (timeout after %g s)", b.final_push_timeout)
        return False
    except Exception as exc:  # noqa: BLE001 - any failure is logged, the exit goes on
        log.error("final push: failed (%s: %s)", type(exc).__name__, str(exc)[:200])
        return False
    log.info("final push: ok (%d files)", n)
    return True


STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)


async def serve(b: Built, host: str = "0.0.0.0") -> None:
    """Run the API (and the status port, if any) until SIGTERM/SIGINT. Shutdown, in order:
    close the listeners and stop the reconcile and sync loops; push the state (bounded by
    FLOCK_FINAL_PUSH_TIMEOUT) and log the result; give open requests GRACEFUL_SHUTDOWN s; exit.
    The push runs before the drain, so a short platform grace period still gets it."""
    import uvicorn

    class Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):  # signals are handled once, below, for both servers
            yield

    dep = b.deployment
    graceful = {"timeout_graceful_shutdown": GRACEFUL_SHUTDOWN}
    servers = [Server(uvicorn.Config(b.app, host=host, port=dep.port, log_level="info",
                                     **graceful))]
    if dep.status_port:
        cfg = uvicorn.Config(status_only(b.app), host=host, port=dep.status_port,
                             lifespan="off", log_level="warning", **graceful)
        servers.append(Server(cfg))

    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()

    def on_signal(sig: signal.Signals) -> None:
        if not stopping.is_set():
            log.info("%s: shutting down", sig.name)
        stopping.set()

    # SIGUSR1: a Spot notice on Slurm (the watcher signals every process of the job); the
    # same once-only shutdown as SIGTERM (docs/slurm.md)
    for sig in STOP_SIGNALS:
        loop.add_signal_handler(sig, on_signal, sig)
    log.info("flock controller (%s mode): API %s, workers use %s%s", dep.mode, dep.public_url,
             dep.worker_url, f", status {dep.status_url}" if dep.status_url else "")

    async def run(s) -> None:
        try:
            await s.serve()
        finally:
            stopping.set()  # one server ending (a bind error) ends both

    tasks = [asyncio.create_task(run(s)) for s in servers]
    try:
        await stopping.wait()
        for s in servers:
            s.should_exit = True
        if b.halt is not None:
            b.halt()
        await final_push(b)
        _, pending = await asyncio.wait(tasks, timeout=GRACEFUL_SHUTDOWN + 5)
        for t in pending:
            t.cancel()
    finally:
        for sig in STOP_SIGNALS:
            loop.remove_signal_handler(sig)
    log.info("flock controller: exit")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(serve(build()))


if __name__ == "__main__":
    main()
