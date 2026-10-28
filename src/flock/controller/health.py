"""Controller health (docs/health.md): world and provider probes, per-swarm snapshots, token
usage per hour and hourly per-agent rollups for the run viewer.

Every SAMPLE_S the controller probes each swarm's world (`GET /manifest` timed, `GET /metrics`
when the manifest lists `metrics-v1`, the Slurm job state when `world.slurm_job_name` is set)
and the providers with `metrics: true` (vLLM Prometheus text), then builds one snapshot per
swarm from the store (heartbeats, telemetry), the swarm stream and in-memory counters fed by
`post_events`. Event files are never re-read.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import httpx

from flock.controller.reconcile import swarm_stream
from flock.controller.store import Store
from flock.models import Event, SwarmSpec
from flock.world.host import world_token
from flock.world.protocol import TOKEN_HEADER

log = logging.getLogger("flock.health")

SAMPLE_S = 30.0  # FLOCK_HEALTH_INTERVAL overrides
PROBE_TIMEOUT_S = 5.0
DOWN_AFTER = 2  # consecutive failed manifest probes before `world_down`
VLLM_METRICS = {"num_requests_waiting": "waiting", "num_requests_running": "running",
                "gpu_cache_usage_perc": "kv", "kv_cache_usage_perc": "kv"}  # vLLM v0 / v1 names


_PROM = re.compile(r"^([A-Za-z_:][\w:]*)(?:\{[^}]*\})?\s+(\S+)")


def parse_vllm(text: str) -> dict[str, float]:
    """`waiting`, `running` (summed over label sets) and `kv` (max) from vLLM's /metrics."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _PROM.match(line)
        key = VLLM_METRICS.get(m.group(1).rsplit(":", 1)[-1]) if m else None
        if key is None:
            continue
        try:
            v = float(m.group(2))
        except ValueError:
            continue
        out[key] = max(out.get(key, 0.0), v) if key == "kv" else out.get(key, 0.0) + v
    return out


class Health:
    def __init__(self, store: Store, *, token: str = "", secrets: Mapping[str, str] = os.environ,
                 jobs: Any = None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.store, self.token, self.secrets, self.jobs = store, token, secrets, jobs
        self.transport = transport
        self.world: dict[str, dict[str, Any]] = {}  # swarm -> {up, manifest_ms, slurm_state, metrics}
        self.providers: dict[str, dict[str, dict[str, float]]] = {}  # swarm -> provider -> queue
        self._fails: dict[str, int] = {}  # swarm -> consecutive failed manifest probes

    async def run(self, interval: float = SAMPLE_S) -> None:
        """Probe, then snapshot, every `interval` s, until cancelled."""
        while True:
            now = time.time()
            await self.probe(now)
            try:
                await asyncio.to_thread(self.tick, now)
            except Exception:
                log.exception("health snapshot failed")
            await asyncio.sleep(interval)

    def tick(self, now: float) -> None:
        pass

    # ------------------------------------------------------------------ probes
    async def probe(self, now: float) -> None:
        async with httpx.AsyncClient(transport=self.transport, timeout=PROBE_TIMEOUT_S) as http:
            for sw in self.store.list_swarms():
                spec: SwarmSpec = sw["spec"]
                try:
                    await self._probe_world(http, spec, now)
                    await self._probe_providers(http, spec)
                except Exception:  # a probe never stops the loop
                    log.exception("health probe of %s failed", spec.swarm)

    async def _probe_world(self, http: httpx.AsyncClient, spec: SwarmSpec, now: float) -> None:
        w = spec.world
        if w is None or not w.endpoint:
            self.world.pop(spec.swarm, None)
            return
        token = world_token(spec, self.token, self.secrets)
        headers = {TOKEN_HEADER: token} if token else {}
        state: dict[str, Any] = {"up": False, "manifest_ms": None, "slurm_state": None,
                                 "metrics": None}
        t0 = time.monotonic()
        error = None
        try:
            r = await http.get(f"{w.endpoint}/manifest", headers=headers)
            state["manifest_ms"] = round((time.monotonic() - t0) * 1000, 1)
            state["up"] = r.status_code == 200
            if not state["up"]:
                error = f"manifest answered {r.status_code}"
            elif "metrics-v1" in (r.json().get("capabilities") or []):
                m = await http.get(f"{w.endpoint}/metrics", headers=headers)
                if m.status_code == 200:
                    state["metrics"] = m.json()  # stored as-is
        except (httpx.HTTPError, ValueError) as e:
            error = f"{type(e).__name__}: {e}"[:200]
        if w.slurm_job_name and hasattr(self.jobs, "state_by_name"):
            st = await asyncio.to_thread(self.jobs.state_by_name, w.slurm_job_name)
            state["slurm_state"] = st or "absent"
        self.world[spec.swarm] = state
        fails = 0 if state["up"] else self._fails.get(spec.swarm, 0) + 1
        prev, self._fails[spec.swarm] = self._fails.get(spec.swarm, 0), fails
        if fails == DOWN_AFTER:
            self._event(spec.swarm, now, "world_down", {
                "endpoint": w.endpoint, "error": error, "slurm_state": state["slurm_state"]})
        elif fails == 0 and prev >= DOWN_AFTER:
            self._event(spec.swarm, now, "world_up", {
                "endpoint": w.endpoint, "manifest_ms": state["manifest_ms"]})

    async def _probe_providers(self, http: httpx.AsyncClient, spec: SwarmSpec) -> None:
        out: dict[str, dict[str, float]] = {}
        for name, p in spec.providers.items():
            if not p.metrics:
                continue
            q: dict[str, float] = {}
            for base in p.base_urls or ([p.base_url] if p.base_url else []):
                root = base.rstrip("/").removesuffix("/v1")
                try:
                    r = await http.get(f"{root}/metrics")
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    log.debug("provider %s metrics at %s: %s", name, root, e)
                    continue
                for k, v in parse_vllm(r.text).items():
                    q[k] = max(q.get(k, 0.0), v) if k == "kv" else q.get(k, 0.0) + v
            out[name] = q
        self.providers[spec.swarm] = out

    def _event(self, swarm: str, now: float, kind: str, data: dict[str, Any]) -> None:
        log.warning("%s: %s %s", swarm, kind, data)
        self.store.append_events(swarm, swarm_stream(swarm), [Event(
            type=kind, ts=datetime.fromtimestamp(now, UTC), agent="", data=data)])
