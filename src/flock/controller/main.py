"""Controller entry point: `flock-controller`.

Two deployments (docs/deploy-job.md):
- `space` (default): an HF Docker Space; workers reach it through the Spaces edge at its
  public URL. Fine for small swarms.
- `job` (FLOCK_MODE=job): an HF Job in a network group under the alias `controller`. Workers
  join the same group and reach it directly at `http://${HF_NETWORK_GROUP_PREFIX}controller:7860`
  (the worker expands the variable). Humans use the jobs proxy: the API on 7860 (HF token
  gated) and a read-only status page on 7861 (public). The SQLite state is synced to a bucket.
"""

from __future__ import annotations

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
        return Deployment("space", port, url, url, status_port=status)
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


def build(env: Mapping[str, str] = os.environ) -> Built:
    token = env.get("FLOCK_TOKEN", "")
    if not token:
        raise SystemExit("FLOCK_TOKEN must be set")
    dep = deployment(env)
    data_dir = env.get("FLOCK_DATA", "./data")
    state = state_sync_from_env(env, data_dir)
    if dep.mode == "job" and state is None:
        log.warning("job mode without FLOCK_STATE_URL: controller state is lost on restart")
    if state is not None and env.get("FLOCK_STATE_RESTORE", "1") != "0":
        state.restore()
    store = Store(data_dir)
    if state is not None and (fixed := store.repair_events()):
        log.info("state: trimmed %d event streams to the restored index", fixed)
    jobs: JobsClient
    if env.get("FLOCK_FAKE_JOBS") == "1":
        jobs = FakeJobsClient(network_group=dep.network_group)
    else:
        jobs = HubJobsClient(token=env.get("HF_TOKEN"), network_group=dep.network_group)
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

    async def sync_state() -> None:
        while True:
            await asyncio.sleep(sync_every)
            try:
                await asyncio.to_thread(state.push, store)
            except Exception:
                log.exception("state push failed")

    async def background(notifier: Notifier) -> None:
        tasks = [asyncio.create_task(log_stats())]
        if state is not None:
            tasks.append(asyncio.create_task(sync_state()))
        try:
            await run_reconciler(
                store, jobs, settings, interval, wake=lambda t: notifier.wait("reconcile:", t)
            )
        finally:
            for t in tasks:
                t.cancel()

    app = create_app(store, token, background=background, stats=stats, namespace=dep.namespace)
    return Built(app, store, dep, jobs, state, settings)


def build_app():
    b = build()
    return b.app, b.deployment.port


async def serve(b: Built, host: str = "0.0.0.0") -> None:
    """Run the API (and the status port, if any) until SIGTERM/SIGINT, then push state."""
    import uvicorn

    class Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):  # signals are handled once, below, for both servers
            yield

    dep = b.deployment
    servers = [Server(uvicorn.Config(b.app, host=host, port=dep.port, log_level="info"))]
    if dep.status_port:
        cfg = uvicorn.Config(status_only(b.app), host=host, port=dep.status_port,
                             lifespan="off", log_level="warning")
        servers.append(Server(cfg))

    def stop() -> None:
        for s in servers:
            s.should_exit = True

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop)
    log.info("flock controller (%s mode): API %s, workers use %s%s", dep.mode, dep.public_url,
             dep.worker_url, f", status {dep.status_url}" if dep.status_url else "")

    async def run(s) -> None:
        try:
            await s.serve()
        finally:
            stop()  # one server ending (a bind error, a signal) ends both

    await asyncio.gather(*(run(s) for s in servers))
    if b.state is not None:
        log.info("state: final push before exit")
        try:
            await asyncio.to_thread(b.state.push, b.store, force=True)
        except Exception:
            log.exception("final state push failed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(serve(build()))


if __name__ == "__main__":
    main()
