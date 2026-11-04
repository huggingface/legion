"""Python client for the controller API: the one way the `flock` CLI talks to a controller.

Config: FLOCK_URL and FLOCK_TOKEN (sent as X-Flock-Token; a `*.hf.jobs` URL or
FLOCK_PROXY_AUTH=hf also sends HF_TOKEN as the bearer the jobs proxy wants). Without FLOCK_URL,
a local run `flock launch` started in this directory (`.flock/current`, docs/local.md) is used.
Inside an agent the worker sets FLOCK_AGENT_TOKEN (that agent's scoped token, which wins over
FLOCK_TOKEN), FLOCK_AGENT and FLOCK_PARENT; when the agent's pin file (FLOCK_TOOLS_PIN) names a
worker gateway, calls go through it instead. See docs/spawn.md for what an agent may call.

Methods take and return plain data (dicts, `SwarmSpec`); a refused or failed call raises
`ClientError`.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from flock.models import AGENT_NAME_RE, SwarmSpec


class ClientError(Exception):
    """A controller call that failed: `status` (None without an HTTP answer, e.g. no HF token
    for the jobs proxy or the worker gateway failing) and the reply `body`; str() is the line
    the CLI prints."""

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status, self.body = status, body

    @property
    def detail(self) -> str:
        """The reply's `detail` when it is a string, else the whole body."""
        try:
            d = json.loads(self.body).get("detail")
        except (ValueError, AttributeError):
            d = None
        return d if isinstance(d, str) else self.body


def proxy_auth(url: str, env: Mapping[str, str] = os.environ) -> bool:
    """True when `url` is behind the HF Jobs proxy (a job-mode controller seen from outside its
    network group): the proxy wants `Authorization: Bearer <HF token>` (docs/deploy-job.md)."""
    if env.get("FLOCK_PROXY_AUTH") == "hf":
        return True
    return (httpx.URL(url).host or "").endswith(".hf.jobs")


def auth_headers(url: str, env: Mapping[str, str] = os.environ) -> dict[str, str]:
    """`X-Flock-Token` always. `Authorization` carries the HF token behind the jobs proxy, and
    the flock token otherwise (what Space-mode controllers before X-Flock-Token expect)."""
    token = env.get("FLOCK_AGENT_TOKEN") or env.get("FLOCK_TOKEN", "")
    headers = {"X-Flock-Token": token}
    if proxy_auth(url, env):
        hf = env.get("HF_TOKEN")
        if not hf:
            from huggingface_hub import get_token

            hf = get_token()
        if not hf:
            raise ClientError(f"{url} is behind the HF Jobs proxy: set HF_TOKEN (or `hf auth login`)")
        headers["Authorization"] = f"Bearer {hf}"
    else:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def unreachable(url: str, env: Mapping[str, str] = os.environ) -> str:
    """The line for a controller that does not answer: with a run dir (`--run`,
    FLOCK_RUN_DIR) also what its hosts/controller.host says and when it was written (the
    controller's last start, not a heartbeat)."""
    line = f"controller unreachable at {url}"
    host = Path(env.get("FLOCK_RUN_DIR") or "") / "hosts" / "controller.host"
    if env.get("FLOCK_RUN_DIR") and host.is_file():
        t = datetime.fromtimestamp(host.stat().st_mtime, UTC)
        line += (f"; hosts/controller.host says {host.read_text(encoding='utf-8').strip()}, "
                 f"written at {t:%Y-%m-%d %H:%M:%S}Z")
    return line


CURRENT_FILE = Path(".flock") / "current"  # written by a local run (docs/local.md)


def local_run(env: Mapping[str, str] = os.environ) -> dict | None:
    """The live local run to talk to when FLOCK_URL is unset: `$FLOCK_CURRENT` or
    `.flock/current` in the working directory, if its process is still alive."""
    if env.get("FLOCK_URL"):
        return None
    from flock.current import read_current

    return read_current(Path(env.get("FLOCK_CURRENT") or CURRENT_FILE))


def connect(env: Mapping[str, str] = os.environ) -> httpx.Client:
    """The HTTP client for one call. Tests replace it to point at an in-process app."""
    if (run := local_run(env)) is not None:
        env = {**env, "FLOCK_URL": run["url"], "FLOCK_TOKEN": run["token"]}
        env.pop("FLOCK_AGENT_TOKEN", None)
    url = env.get("FLOCK_URL", "http://localhost:7860")
    return httpx.Client(base_url=url, headers=auth_headers(url, env), timeout=60)


RETRY_ATTEMPTS = 5  # requests in all, per call
RETRY_BASE = 1.0  # s, first backoff before jitter
RETRY_CAP = 30.0  # s, longest single wait, Retry-After included
_sleep = time.sleep  # tests replace it


def _retry_after(r: httpx.Response) -> float | None:
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
    return max(0.0, when.timestamp() - time.time()) if when.tzinfo else None


def _retryable(method: str, r: httpx.Response | None) -> bool:
    """429 (the Space edge throttling 1,000 agents) and 502/503/504 never reached the app or
    were not processed: retry any method. A 500 or a dropped connection may have been
    processed, so only reads retry on those."""
    if r is None:
        return method == "GET"
    return r.status_code in (429, 502, 503, 504) or (r.status_code == 500 and method == "GET")


def _request(c: httpx.Client, method: str, path: str, **kw) -> httpx.Response:
    """One controller request, retried up to RETRY_ATTEMPTS times with exponential backoff and
    full jitter, never sooner than Retry-After."""
    for attempt in range(RETRY_ATTEMPTS):
        last = attempt == RETRY_ATTEMPTS - 1
        try:
            r = c.request(method, path, **kw)
        except httpx.TransportError:
            if last or not _retryable(method, None):
                raise
            r = None
        if r is not None and (last or not _retryable(method, r)):
            return r
        backoff = random.uniform(0, min(RETRY_CAP, RETRY_BASE * 2 ** attempt))
        wait = max(backoff, (_retry_after(r) or 0.0) if r is not None else 0.0)
        _sleep(min(RETRY_CAP, wait))
    raise AssertionError("unreachable")


def gateway(env: Mapping[str, str] = os.environ) -> tuple[str, str, str] | None:
    """(address, key, agent) of the worker gateway named in the agent's pin file
    (FLOCK_TOOLS_PIN), or None outside an agent or on a worker without one."""
    from flock.tools.toolset import PIN_ENV, read_pin

    if not (pin := env.get(PIN_ENV)):
        return None
    try:
        ts, agent, _ = read_pin(pin)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return (ts.gateway, ts.gateway_key or "", agent) if ts.gateway else None


def _call_gateway(gw: tuple[str, str, str], method: str, path: str, timeout: float | None,
                  **kw) -> Any:
    """One controller call through the worker (`controller:request`): the worker sends it with
    this agent's token to wherever the controller is now, retrying there."""
    import asyncio

    from flock.tools.toolset import gateway_call

    address, key, agent = gw
    args = {"method": method, "path": path, "params": kw.get("params"), "json": kw.get("json")}
    r = asyncio.run(gateway_call(address, key, agent, "controller:request", args,
                                 timeout=(timeout or 60) + 120))
    try:
        out = json.loads(r.output)
    except ValueError:
        out = {}
    if not isinstance(out, dict) or "status" not in out:
        err = out.get("error") if isinstance(out, dict) else None
        raise ClientError(f"error: {err or r.output}")
    if out["status"] >= 400:
        body = out["body"] if isinstance(out["body"], str) else json.dumps(out["body"])
        raise ClientError(f"error {out['status']}: {body}", out["status"], body)
    return out["body"]


class Client:
    """The controller API. Base URL and auth come from `env` (default the process
    environment) at each call, as described in the module docstring."""

    def __init__(self, env: Mapping[str, str] | None = None):
        self.env = os.environ if env is None else env
        self.env_files: list[str] = []  # a run dir's cluster.env, run.env (`for_run`)

    @classmethod
    def for_run(cls, run_dir: str | Path | None = None, *, need_host: bool = True,
                env: Mapping[str, str] = os.environ) -> Client:
        """The client for RUN_DIR (`flock init`; default `$FLOCK_RUN_DIR`, else plain `env`):
        its controller (hosts/controller.host, else its local run) and token, with its
        cluster.env and run.env in `env_files` (the first --env files). An exported FLOCK_URL
        beats `$FLOCK_RUN_DIR` but not an explicit `run_dir`. ClientError when `need_host` and
        the run has no controller yet."""
        from flock.rundir import run_context

        d = run_dir or env.get("FLOCK_RUN_DIR")
        if not d:
            return cls(env)
        ctx, files = run_context(d)
        if run_dir is None and env.get("FLOCK_URL"):
            api = cls(env)  # an exported FLOCK_URL (with its token) beats FLOCK_RUN_DIR
        else:
            if need_host and "FLOCK_URL" not in ctx and "FLOCK_CURRENT" not in ctx:
                raise ClientError(f"no controller host file in {d}/hosts; launch first")
            # never this run's token with another controller's URL
            api = cls({**{k: v for k, v in env.items() if k != "FLOCK_URL"}, **ctx})
        api.env_files = files
        return api

    def _call(self, method: str, path: str, *, timeout: float | None = None, **kw) -> Any:
        if (gw := gateway(self.env)) is not None:  # inside an agent: its worker (docs/spawn.md)
            return _call_gateway(gw, method, path, timeout, **kw)
        if timeout is not None:  # per request, without the TestClient's `timeout` deprecation
            kw["extensions"] = {"timeout": httpx.Timeout(timeout).as_dict()}
        r = self._send(method, path, **kw)
        if r.status_code >= 400:
            raise ClientError(f"error {r.status_code}: {r.text}", r.status_code, r.text)
        return r.json()

    def _found(self, path: str) -> dict | None:
        """GET `path` straight to the controller: the reply, or None on any error."""
        r = self._send("GET", path)
        return r.json() if r.status_code == 200 else None

    def _send(self, method: str, path: str, **kw) -> httpx.Response:
        """One request (retried); no answer at all raises `ClientError(unreachable(...))`."""
        with connect(self.env) as c:
            try:
                return _request(c, method, path, **kw)
            except httpx.TransportError:
                raise ClientError(unreachable(str(c.base_url).rstrip("/"), self.env)) from None

    @property
    def as_agent(self) -> bool:
        """Running inside an agent (its scoped token or its worker's gateway): names are
        agents, never swarms."""
        return bool(self.env.get("FLOCK_AGENT_TOKEN") or gateway(self.env))

    def is_swarm(self, name: str) -> bool:
        """A swarm, unless it names an agent: a numeric agent name (`...-NNNN`) that is not a
        swarm, or any other name (`naming: words`, e.g. `amber-otter`) that is an agent and
        not a swarm. A name without a hyphen is no agent's: no lookup."""
        if "-" not in name:
            return True
        if self._found(f"/swarms/{name}") is not None:
            return True
        return not AGENT_NAME_RE.match(name) and self._found(f"/agents/{name}") is None

    # -- swarms ---------------------------------------------------------------------------------

    def swarms(self) -> list[dict]:
        """Every swarm's status."""
        return self._call("GET", "/swarms")

    def swarm(self, swarm: str) -> dict:
        """{"spec": stored spec, "status": status}; ClientError 404 when there is none."""
        return self._call("GET", f"/swarms/{swarm}")

    def create(self, spec: SwarmSpec, *, source: dict[str, str] | None = None) -> dict:
        """Create the swarm (refused with 412 when it exists: If-None-Match); `source` is the
        X-Flock-Source header. Returns its status."""
        return self._call("PUT", f"/swarms/{spec.swarm}", params={"replicas": "spec"},
                          json=spec.model_dump(mode="json"),
                          headers={**(source or {}), "If-None-Match": "*"})

    def preflight(self, spec: SwarmSpec) -> dict:
        """Probe the spec's provider-backed models; nothing is stored."""
        return self._call("POST", f"/swarms/{spec.swarm}/preflight",
                          json=spec.model_dump(mode="json"), timeout=300)

    def upload_tools(self, swarm: str, data: bytes) -> dict:
        """Upload a packed tools bundle: {"version", "tools"}."""
        return self._call("POST", f"/swarms/{swarm}/tools", content=data,
                          headers={"content-type": "application/gzip"}, timeout=120)

    def tool_bundles(self, swarm: str) -> list[dict]:
        """The swarm's uploaded tools bundles."""
        return self._call("GET", f"/swarms/{swarm}/tools").get("bundles", [])

    def update(self, spec: SwarmSpec, *, check: bool = False,
               source: dict[str, str] | None = None) -> dict:
        """Apply `spec` to the running swarm (replicas as in `spec`). `check`: dry run, the
        reply carries the manifest it would store and the deletes."""
        return self._call("PUT", f"/swarms/{spec.swarm}", json=spec.model_dump(mode="json"),
                          params={"replicas": "spec", **({"dry_run": 1} if check else {})},
                          headers=source)

    def resume(self, swarm: str, *, types: list[str] | None = None,
               agents: list[str] | None = None) -> dict:
        """Lift holds (all, or the named ones) and the release: workers relaunch."""
        return self._call("POST", f"/swarms/{swarm}/resume",
                          json={"types": types or None, "agents": agents or None, "release": True})

    def stop(self, swarm: str, *, types: list[str] | None = None, agents: list[str] | None = None,
             interrupt: bool = False, reason: str = "", release: bool = False) -> dict:
        """Hold the swarm (or the named types/agents); `interrupt` cuts running turns,
        `release` drains and frees every worker. Returns what is held."""
        return self._call("POST", f"/swarms/{swarm}/pause", json={
            "types": types or None, "agents": agents or None, "interrupt": interrupt,
            "reason": reason, "release": release})

    def stop_loop(self, swarm: str, type: str, *, final: bool = True) -> dict:
        """Stop a type's loop, after a final round when `final`."""
        return self._call("POST", f"/swarms/{swarm}/loop",
                          json={"type": type, "action": "stop", "final": final})

    def delete(self, swarm: str) -> dict:
        """Drain the swarm and drop it from the controller (the bucket keeps its copies)."""
        return self._call("DELETE", f"/swarms/{swarm}")

    def health(self, swarm: str) -> dict | None:
        """The newest health snapshot, or None before the first."""
        return self._found(f"/swarms/{swarm}/health")

    def workers(self, swarm: str) -> list[dict]:
        return self._call("GET", "/workers", params={"swarm": swarm})

    def loops(self, swarm: str) -> list[dict]:
        """Each type's loop with its per-agent rows (`per_agent`)."""
        return self._call("GET", f"/swarms/{swarm}/loop")

    # -- agents ---------------------------------------------------------------------------------

    def agents(self, swarm: str, *, type: str | None = None, state: str | None = None,
               parent: str | None = None) -> list[dict]:
        """Every agent of the swarm (all pages), optionally filtered."""
        params = {k: v for k, v in (("type", type), ("state", state), ("parent", parent)) if v}
        out, cursor = [], None
        while True:
            page = self._call("GET", f"/swarms/{swarm}/agents",
                              params={**params, "cursor": cursor} if cursor else params)
            out += page["agents"]
            if not (cursor := page["cursor"]):
                return out

    def agent(self, name: str) -> dict:
        return self._call("GET", f"/agents/{name}")

    def prompt(self, name: str, text: str, *, timeout: str | None = None,
               reply_to: str | None = None) -> dict:
        return self._call("POST", f"/agents/{name}/prompt",
                          json={"timeout": timeout, "reply_to": reply_to, "text": text})

    def prompt_type(self, swarm: str, type: str, text: str, *, state: str | None = None,
                    skip_pending: bool | None = None, timeout: str | None = None,
                    reply_to: str | None = None) -> dict:
        """Prompt every agent of a type (one request); returns the counts."""
        return self._call("POST", f"/swarms/{swarm}/prompts",
                          json=_bulk(text, type, state, skip_pending, timeout, reply_to))

    def prompt_children(self, parent: str, text: str, *, type: str | None = None,
                        state: str | None = None, skip_pending: bool | None = None,
                        timeout: str | None = None, reply_to: str | None = None) -> dict:
        """Prompt the direct children of `parent` (one request); returns the counts."""
        return self._call("POST", f"/agents/{parent}/children/prompts",
                          json=_bulk(text, type, state, skip_pending, timeout, reply_to))

    def events(self, name: str, *, swarm: bool, after: int = 0, follow: bool = False,
               timeout: float | None = None) -> Iterator[dict]:
        """Pages of an agent's (or, `swarm`, a swarm's) event stream after cursor `after`:
        {"events": [...], "next": cursor}. One page, or with `follow` long-polls (25 s each)
        forever or until `timeout` seconds have passed."""
        end = None if timeout is None else time.monotonic() + timeout
        route = f"/swarms/{name}/events" if swarm else f"/agents/{name}/events"
        while True:
            page = self._call("GET", route, params={"after": after, "wait": 25 if follow else 0})
            yield page
            after = page["next"]
            if not follow or (end is not None and time.monotonic() >= end):
                return

    def spawn(self, parent: str, type: str, *, prompt: str | None = None,
              key: str | None = None) -> dict:
        """Spawn a child agent of `type` (inside an agent); `key` makes a repeat idempotent."""
        return self._call("POST", f"/agents/{parent}/spawn",
                          json={"type": type, "prompt": prompt, "key": key})

    def remove(self, name: str) -> dict:
        """Remove a spawned agent and everything it spawned."""
        return self._call("DELETE", f"/agents/{name}")

    # -- the world's admin API (docs/world.md) --------------------------------------------------

    def world_knobs(self, swarm: str) -> dict:
        return self._call("GET", f"/swarms/{swarm}/world/knobs")

    def set_world_knobs(self, swarm: str, values: dict[str, Any], *, reason: str = "",
                        source: dict[str, str] | None = None) -> dict:
        return self._call("PATCH", f"/swarms/{swarm}/world/knobs", json=values,
                          params={"reason": reason}, headers=source)


def _bulk(text, type, state, skip_pending, timeout, reply_to) -> dict:
    body = {"timeout": timeout, "reply_to": reply_to, "type": type, "state": state, "text": text}
    if skip_pending is not None:
        body["skip_pending"] = skip_pending
    return body
