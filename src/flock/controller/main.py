"""Controller entry point: `flock-controller`."""

from __future__ import annotations

import asyncio
import logging
import os

from flock.controller.api import Notifier, create_app
from flock.controller.jobs import FakeJobsClient, HubJobsClient, JobsClient
from flock.controller.reconcile import Settings, run_reconciler
from flock.controller.store import Store

log = logging.getLogger("flock.controller")
STATS_LOG_INTERVAL = 60.0


def _public_url(port: int) -> str:
    if os.environ.get("FLOCK_URL"):
        return os.environ["FLOCK_URL"]
    if os.environ.get("SPACE_HOST"):  # set by HF Spaces
        return f"https://{os.environ['SPACE_HOST']}"
    return f"http://localhost:{port}"


def parse_labels(raw: str) -> dict[str, str]:
    """'a=1,b=2' -> {'a': '1', 'b': '2'} (FLOCK_JOB_LABELS)."""
    out = {}
    for part in raw.split(","):
        if part.strip():
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
    return out


def build_app():
    port = int(os.environ.get("PORT", "7860"))
    token = os.environ.get("FLOCK_TOKEN", "")
    if not token:
        raise SystemExit("FLOCK_TOKEN must be set")
    store = Store(os.environ.get("FLOCK_DATA", "./data"))
    jobs: JobsClient
    if os.environ.get("FLOCK_FAKE_JOBS") == "1":
        jobs = FakeJobsClient()
    else:
        jobs = HubJobsClient(token=os.environ.get("HF_TOKEN"))
    settings = Settings(
        public_url=_public_url(port),
        token=token,
        job_labels=parse_labels(os.environ.get("FLOCK_JOB_LABELS", "")),
    )
    interval = float(os.environ.get("FLOCK_RECONCILE_INTERVAL", "5"))

    def stats() -> dict:
        calls = getattr(jobs, "calls", None)
        return {"hub_calls": dict(calls)} if calls is not None else {}

    async def log_stats() -> None:
        while True:
            await asyncio.sleep(STATS_LOG_INTERVAL)
            log.info("hub calls: %s", stats().get("hub_calls", {}))

    async def background(notifier: Notifier) -> None:
        stats_task = asyncio.create_task(log_stats())
        try:
            await run_reconciler(
                store, jobs, settings, interval, wake=lambda t: notifier.wait("reconcile:", t)
            )
        finally:
            stats_task.cancel()

    return create_app(store, token, background=background, stats=stats), port


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app, port = build_app()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
