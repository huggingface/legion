"""The worker side of a world (docs/world.md): its manifest tools as built-in tools, calls
forwarded with the agent's identity and the shared secret, the hooks, and the World section of
the agent's instructions.

- The controller sends the swarm's `world` block (endpoint and manifest, fetched at apply) with
  every assign and heartbeat response, plus this type's selection; `configure` takes it. A turn
  that starts after a re-apply sees the new manifest (`toolset_for` is read at turn start).
- World tools are `ToolDef`s with `builtin="world:<name>"`, next to the folder and comms tools:
  the `llm` harness calls them in process (`call`), `flock-tool` through the worker's local
  gateway (the comms hub's listener, which routes `world:*` here).
- The secret arrives as the job secret `FLOCK_WORLD_TOKEN`; it stays in the worker (removed from
  the harness env and from every tool call's env). It is sent only to the configuration it
  belongs to: the world block names it (`credential_origin`, `credential_id` =
  HMAC(secret, origin)); when a re-apply binds the world to another secret or endpoint, calls
  under the new configuration are refused here and `needs_restart()` tells the worker to drain,
  so the controller relaunches it with the new secret.
- A turn is pinned: `toolset_for` snapshots endpoint, manifest, selection and whether the secret
  belongs to them, and every call of that turn (in process, or through the gateway with the
  snapshot id in the pinned tool's `builtin`, `world:<name>@<id>`) uses that snapshot.
- Hooks the manifest lists are POSTed to `/hooks` in the background: `agent_start` at an agent's
  first prompt on this worker, `turn` at each turn end, `agent_done` at unassign. Best effort.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from flock.models import ToolCounters, WorldSpec
from flock.secrets_file import secret_env
from flock.tools.loader import ToolDef
from flock.tools.runner import DEFAULT_TIMEOUT_S, RESULT_MAX_BYTES, Result, check_args
from flock.tools.toolset import Toolset
from flock.world.protocol import (
    AGENT_HEADER,
    REQUEST_HEADER,
    SWARM_HEADER,
    TOKEN_ENV,
    TOKEN_HEADER,
    TURN_HEADER,
    TYPE_HEADER,
    WorldManifest,
    credential_id,
    origin_matches,
)
from flock.world.workspace import capture

log = logging.getLogger(__name__)

BUILTIN_MODULE = "flock:world"
SECTION_START = "<!-- flock:world -->"
SECTION_END = "<!-- /flock:world -->"
RETRIES = 1  # extra attempts after a 429 / 503, for read-only tools only
RETRY_CAP_S = 10.0
HOOK_TIMEOUT_S = 5.0
HOLDOFF_CAP_S = 600.0
DOWN_HOLD_S = 60.0  # retry_after_s while the controller reports the world down (heartbeats refresh it)
HOLDOFF_DEFAULT_S = 30.0  # a 429 that names no wait
SNAPSHOTS_KEPT = 32
# world error codes that mean "not now" rather than "wrong" (counted as `refused`); any error
# with a `retry_after_s` and any HTTP 429 counts too
REFUSED_CODES = frozenset({"open_budget", "hub_busy", "busy", "paused", "overloaded", "parked"})


def _err(msg: str, t0: float, **extra: Any) -> Result:
    return Result(json.dumps({"error": msg, **extra}, ensure_ascii=False), True,
                  time.monotonic() - t0)


def _retry_after(r: httpx.Response, default: float = 1.0, cap: float = RETRY_CAP_S) -> float:
    try:
        return max(0.0, min(float(r.headers.get("retry-after", default)), cap))
    except ValueError:
        return default


def _parked(r: httpx.Response) -> dict | None:
    """The body of a parked world's 503 (docs/world.md, "Parked mode"), else None."""
    if r.status_code != 503:
        return None
    try:
        body = r.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) and body.get("parked") is True else None


def _loopback(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Snapshot:
    """One world configuration as a turn sees it, for the whole turn."""

    id: str
    endpoint: str
    manifest: WorldManifest
    names: tuple[str, ...]
    send_token: bool  # the worker's secret belongs to this configuration
    refused: str | None  # why calls under this configuration are refused (credential)

    def get(self, name: str):
        return self.manifest.get(name) if name in self.names else None


def _down_message(since: object, raw: dict) -> str | None:
    """The refusal for `world.down_since` (ISO time) in a heartbeat response, else None."""
    if not isinstance(since, str):
        return None
    try:
        hhmm = datetime.fromisoformat(since).astimezone(UTC).strftime("%H:%M UTC")
    except ValueError:
        hhmm = since
    name = (raw.get("manifest") or {}).get("name") or "?"
    return f"world {name} down since {hhmm}; do local work or end your turn"


class WorldHub:
    def __init__(self, *, swarm: str, type_name: str = "", env: Mapping[str, str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 runtime: Callable[[str], Any] | None = None,
                 timeout: float = DEFAULT_TIMEOUT_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.swarm, self.type = swarm, type_name
        self._env = env if env is not None else secret_env()
        self._transport = transport
        self._runtime = runtime or (lambda name: None)
        self.timeout = timeout
        self._clock = clock
        # holdoffs: (until, world message, code); per world after a 429, per (agent, tool) after a hint
        self._world_hold: tuple[float, str, str] | None = None
        self._tool_hold: dict[tuple[str, str], tuple[float, str, str]] = {}
        self._down: str | None = None  # the controller's `world.down_since`, as the refusal
        self.spec: WorldSpec | None = None
        self.names: list[str] = []
        self._client: httpx.AsyncClient | None = None
        self._started: set[str] = set()  # agents whose agent_start hook was sent
        self._snaps: OrderedDict[str, Snapshot] = OrderedDict()
        self._shown: dict[str, str] = {}  # agent -> world version its instructions show
        self._acked: dict[str, str] = {}  # agent -> world version of its last finished turn
        self._tasks: set[asyncio.Task] = set()
        self.counters: dict[str, ToolCounters] = {}  # per tool, since this process started

    # ------------------------------------------------------------------ configuration
    @property
    def active(self) -> bool:
        s = self.spec
        return s is not None and s.endpoint is not None and s.manifest is not None and bool(self.names)

    @property
    def version(self) -> str | None:
        return self.spec.manifest.version if self.active else None  # type: ignore[union-attr]

    def configure(self, spec: WorldSpec | None, names: list[str] | None) -> None:
        """The swarm's world block (from an assign or a heartbeat response) and this worker
        type's world tools. Applies from the next turn boundary."""
        self.spec = spec
        self.names = list(names or []) if spec is not None else []

    def configure_raw(self, raw: object) -> None:
        """The `world` block of a heartbeat response."""
        if not isinstance(raw, dict):
            self._down = None
            self.configure(None, [])
            return
        raw = dict(raw)
        names = raw.pop("tools", None)
        self._down = _down_message(raw.pop("down_since", None), raw)
        try:
            spec = WorldSpec.model_validate(raw)
        except ValueError as e:
            log.warning("heartbeat: unreadable world block: %s", e)
            return
        self.configure(spec, names if isinstance(names, list) else [])

    # ------------------------------------------------------------------ credential
    def _credential(self, s: WorldSpec) -> tuple[bool, str | None]:
        """(send the secret, why calls are refused) for configuration `s`."""
        token = self._env.get(TOKEN_ENV) or ""
        if s.credential_id is None:
            return False, None  # a world without a secret: none is sent
        name = s.manifest.name if s.manifest else "?"
        restart = "this worker restarts to get the new one; try again shortly"
        if not token:
            return False, f"this worker has no secret for world {name}; {restart}"
        origin = s.credential_origin or ""
        bound = (origin.startswith("local:") and _loopback(s.endpoint or "")) or \
            origin_matches(origin, s.endpoint or "")
        if not bound or not hmac.compare_digest(credential_id(token, origin), s.credential_id):
            return False, f"world {name} now uses another secret or endpoint; {restart}"
        return True, None

    def needs_restart(self) -> bool:
        """This worker's types use the world, but its secret is not the current one."""
        return self.active and self._credential(self.spec)[1] is not None  # type: ignore[arg-type]

    def snapshot(self) -> Snapshot | None:
        """The current configuration, kept by id (the last SNAPSHOTS_KEPT) for pinned turns."""
        if not self.active:
            return None
        s = self.spec
        assert s is not None and s.endpoint is not None and s.manifest is not None
        send, refused = self._credential(s)
        key = json.dumps([s.endpoint, s.manifest.model_dump(mode="json"), self.names,
                          s.credential_id, send, refused], sort_keys=True)
        sid = hashlib.sha256(key.encode()).hexdigest()[:12]
        if sid not in self._snaps:
            self._snaps[sid] = Snapshot(sid, s.endpoint, s.manifest, tuple(self.names), send,
                                        refused)
            while len(self._snaps) > SNAPSHOTS_KEPT:
                self._snaps.popitem(last=False)
        self._snaps.move_to_end(sid)
        return self._snaps[sid]

    # ------------------------------------------------------------------ tools
    def tooldefs(self, snap: Snapshot | None = None) -> tuple[ToolDef, ...]:
        snap = snap or self.snapshot()
        if snap is None:
            return ()
        return tuple(ToolDef(name=t.name, description=t.description, parameters=t.parameters,
                             module=BUILTIN_MODULE, function=t.name,
                             builtin=f"world:{t.name}@{snap.id}")
                     for n in snap.names if (t := snap.manifest.get(n)) is not None)

    def toolset_for(self, agent: str, base: Toolset, gateway: Any = None) -> Toolset:
        """`base` (folder and comms tools of this turn) plus the agent's world tools, pinned to
        the current configuration. Built-in calls are routed by prefix: `world:*` here, the rest
        to `base`'s own handler. `gateway` (the comms hub) serves them to `flock-tool`."""
        snap = self.snapshot()
        defs = self.tooldefs(snap)
        if snap is None or not defs:
            return base
        have = set(base.names())
        prev = base.builtin_call

        async def route(entry: str, args: dict[str, Any]) -> Result:
            if entry.startswith("world:"):
                name = entry.split(":", 1)[1].split("@", 1)[0]
                return await self._call(snap, agent, name, args)
            if prev is not None:
                return await prev(entry, args)
            return _err(f"{entry} is not available here", time.monotonic())

        gw, key = base.gateway, base.gateway_key
        if gw is None and gateway is not None and gateway.address is not None:
            gw, key = gateway.address, gateway.key(agent)
        return replace(base, tools=base.tools + tuple(d for d in defs if d.name not in have),
                       gateway=gw, gateway_key=key, builtin_call=route)

    def deny_env(self) -> list[str]:
        """Variables neither the harness nor a tool call ever gets: the world's secret."""
        if self.spec is None:
            return []
        return sorted({TOKEN_ENV, *([self.spec.token_env] if self.spec.token_env else [])})

    def section(self, agent: str) -> str:
        """The World section of the agent's instructions: the world, its version, its tools,
        and a note when the world changed since the agent's last turn."""
        if not self.active:
            return ""
        m = self.spec.manifest  # type: ignore[union-attr]
        names = ", ".join(f"`{n}`" for n in self.names if m.get(n) is not None)
        lines = [SECTION_START, f"## World: {m.name} (version {m.version})", ""]
        if m.description.strip():
            lines += [m.description.strip(), ""]
        lines.append(f"The swarm shares this world. Its tools: {names}. Every call is made as "
                     f"you ({agent}); an error result says what to fix.")
        self._shown[agent] = m.version
        was = self._acked.get(agent)
        if was is not None and was != m.version:
            lines += ["", (f"Note: the world changed since your last turn (version {was} -> "
                           f"{m.version}); the tools above are current.")]
        lines.append(SECTION_END)
        return "\n".join(lines)

    # ------------------------------------------------------------------ lifecycle hooks
    async def on_prompt(self, agent: str, harness: str, gateway: Any = None) -> None:
        """An agent's harness is about to start: open the gateway for a CLI harness and send
        `agent_start` once per agent on this worker. Never raises."""
        if not self.active:
            return
        if harness != "llm" and gateway is not None:
            try:
                await gateway.start_gateway()
            except OSError as e:
                log.warning("world: cannot open the tool gateway: %s", e)
        if agent not in self._started:
            self._started.add(agent)
            self._hook("agent_start", agent)

    def on_turn(self, agent: str, turn: str | None, data: dict[str, Any]) -> None:
        if not self.active:
            return
        if agent in self._shown:
            self._acked[agent] = self._shown[agent]
        self._hook("turn", agent, data, turn)

    def on_done(self, agent: str, reason: str) -> None:
        if self.active:
            self._hook("agent_done", agent, {"reason": reason})
        self._started.discard(agent)
        self._acked.pop(agent, None)
        self._shown.pop(agent, None)

    def _hook(self, event: str, agent: str, data: dict[str, Any] | None = None,
              turn: str | None = None) -> None:
        s = self.spec
        if s is None or s.manifest is None or event not in s.manifest.hooks or s.endpoint is None:
            return
        send, refused = self._credential(s)
        if refused is not None:
            return  # never our secret to another world; the worker is about to restart
        body = {"event": event, "agent": agent, "type": self.type, "swarm": self.swarm,
                "turn": turn or "", "data": data or {}}
        try:
            t = asyncio.get_running_loop().create_task(self._post_hook(s.endpoint, body, send))
        except RuntimeError:
            return
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _post_hook(self, endpoint: str, body: dict[str, Any], send: bool) -> None:
        try:
            r = await self._http().post(f"{endpoint}/hooks", json=body,
                                        headers=self._auth() if send else {},
                                        timeout=HOOK_TIMEOUT_S)
            if r.status_code >= 400:
                log.info("world hook %s for %s: HTTP %d", body["event"], body["agent"], r.status_code)
        except httpx.HTTPError as e:
            log.info("world hook %s for %s failed: %s", body["event"], body["agent"], e)

    # ------------------------------------------------------------------ calls
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self._transport, timeout=self.timeout)
        return self._client

    def _auth(self) -> dict[str, str]:
        token = self._env.get(TOKEN_ENV)
        return {TOKEN_HEADER: token} if token else {}

    async def call(self, agent: str, entry: str, args: dict[str, Any]) -> Result:
        """One world tool call by `agent` (the gateway's route): `world:<name>@<snapshot id>`
        runs under that pinned configuration, `world:<name>` under the current one. Never
        raises: errors are results the model reads."""
        name, _, sid = entry.split(":", 1)[-1].partition("@")
        snap = self._snaps.get(sid) if sid else self.snapshot()
        if snap is None:
            why = ("this turn's world configuration is gone; try again in your next turn" if sid
                   else f"{name} is not enabled for this agent")
            return _err(why, time.monotonic())
        return await self._call(snap, agent, name, args)

    async def _call(self, snap: Snapshot, agent: str, name: str, args: dict[str, Any]) -> Result:
        if snap.get(name) is None:  # not counted: names here come from the agent
            return _err(f"{name} is not enabled for this agent", time.monotonic())
        seen: dict[str, Any] = {}
        res = await self._send(snap, agent, name, args, seen)
        c = self.counters.setdefault(name, ToolCounters())
        if not seen.get("local"):  # local refusals would drag the latency mean down
            c.n += 1
            c.sum_s = round(c.sum_s + res.duration_s, 3)
            c.max_s = round(max(c.max_s, res.duration_s), 3)
        c.timeouts += bool(seen.get("timeout"))
        if res.error:
            c.err += 1
            try:
                body = json.loads(res.output)
            except ValueError:
                body = {}
            c.refused += (seen.get("status") == 429 or body.get("code") in REFUSED_CODES
                          or "retry_after_s" in body)
        return res

    def _held(self, agent: str, name: str, read_only: bool) -> tuple[float, str, str] | None:
        """The active holdoff for this call, if any: the world's, then the (agent, tool)'s."""
        now = self._clock()
        if self._down is not None:  # the controller says the world is down: every call, at once
            return (now + DOWN_HOLD_S, self._down, "world_down")
        if self._world_hold is not None and self._world_hold[0] <= now:
            self._world_hold = None
        if self._world_hold is not None and (not read_only or self._world_hold[2] == "parked"):
            return self._world_hold
        h = self._tool_hold.get((agent, name))
        if h is not None and h[0] <= now:
            del self._tool_hold[(agent, name)]
            return None
        return h

    async def _send(self, snap: Snapshot, agent: str, name: str, args: dict[str, Any],
                    seen: dict[str, Any]) -> Result:
        """One call; `seen` gets the last HTTP status and whether it timed out."""
        t0 = time.monotonic()
        tool = snap.get(name)
        if snap.refused is not None:
            return _err(f"{name}: {snap.refused}", t0)
        if err := check_args(tool.parameters, args):
            return _err(f"{name}: {err}", t0)
        if held := self._held(agent, name, tool.read_only):  # before any capture or upload
            until, msg, code = held
            left = max(0.0, until - self._clock())
            seen["local"] = True
            if code == "world_down":
                return _err(msg, t0, code=code, retry_after_s=math.ceil(left), local=True)
            return _err(f"{msg} (next slot in {max(1, math.ceil(left / 60))} min; do other work "
                        "or end your turn)", t0, code=code, retry_after_s=math.ceil(left),
                        local=True)
        headers = {**(self._auth() if snap.send_token else {}), AGENT_HEADER: agent,
                   TYPE_HEADER: self.type, SWARM_HEADER: self.swarm,
                   REQUEST_HEADER: uuid.uuid4().hex}
        rt = self._runtime(agent)
        if rt is not None and getattr(rt, "turn_id", None):
            headers[TURN_HEADER] = rt.turn_id
        payload = {"args": args}
        ws = tool.workspace if "workspace-upload-v1" in snap.manifest.capabilities else None
        if ws is not None:
            local = args.get(ws.arg, ws.default)
            if local is not None and local not in ws.skip and not (
                    ws.skip_prefix and str(local).startswith(ws.skip_prefix)):
                if rt is None:
                    return _err("no local workspace is available", t0)
                try:
                    payload["workspace"] = await asyncio.to_thread(capture, rt.workdir, local)
                except (ValueError, OSError) as exc:
                    return _err(str(exc), t0)
        wname = snap.manifest.name
        timeout = max(self.timeout, tool.timeout_s or 0)
        # a write may have happened: never resent; a read-only call during a holdoff tries once
        retries = RETRIES if tool.read_only and self._world_hold is None else 0
        for attempt in range(retries + 1):
            try:
                r = await self._http().post(f"{snap.endpoint}/tools/{name}", json=payload,
                                            headers=headers, timeout=timeout)
            except httpx.TimeoutException:
                seen["timeout"] = True
                return _err(f"{name}: world {wname} did not answer within {timeout:g} s", t0)
            except httpx.HTTPError as e:
                return _err(f"{name}: world {wname} is unreachable ({type(e).__name__})", t0)
            seen["status"] = r.status_code
            if r.status_code in (429, 503) and attempt < retries and not _parked(r):
                await asyncio.sleep(_retry_after(r))
                continue
            break
        if r.status_code == 401:
            return _err(f"{name}: world {wname} refused flock's token (401)", t0)
        if parked := _parked(r):  # a world-level holdoff: no call (or upload) until it is back
            wait = parked.get("next_retry_s")
            wait = float(wait) if isinstance(wait, int | float) and wait > 0 else HOLDOFF_DEFAULT_S
            wait = min(wait, HOLDOFF_CAP_S)
            msg = (f"world {wname} unavailable ({str(parked.get('error') or 'parked')[:300]}); "
                   "work locally")
            self._world_hold = (self._clock() + wait, msg, "parked")
            return _err(msg, t0, code="parked", retry_after_s=wait)
        try:
            body = r.json()
        except ValueError:
            body = None
        if r.status_code == 429:
            e = body.get("error") if isinstance(body, dict) else None
            e = e if isinstance(e, dict) else {}
            wait = e.get("retry_after_s")
            if not isinstance(wait, (int, float)):
                wait = _retry_after(r, HOLDOFF_DEFAULT_S, HOLDOFF_CAP_S)
            wait = max(0.0, min(float(wait), HOLDOFF_CAP_S))
            msg = str(e.get("message") or f"world {wname} is overloaded")
            code = str(e.get("code") or "overloaded")
            self._world_hold = (self._clock() + wait, msg, code)
            return _err(msg, t0, code=code, retry_after_s=wait)
        if body is None:
            return _err(f"{name}: world {wname} answered HTTP {r.status_code}: {r.text[:200]}", t0)
        if isinstance(body, dict) and body.get("ok") is True:
            result = body.get("result")
            if result is None and isinstance(body.get("text"), str):
                result = body["text"]
            out = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            if len(out.encode()) > RESULT_MAX_BYTES:
                return _err(f"{name}: result over {RESULT_MAX_BYTES} bytes", t0)
            return Result(out, False, time.monotonic() - t0)
        if isinstance(body, dict) and body.get("ok") is False:
            e = body.get("error")
            msg = e.get("message") if isinstance(e, dict) else e
            extra = {k: e[k] for k in ("code", "retry_after_s") if k in e} if isinstance(e, dict) else {}
            wait = extra.get("retry_after_s")
            if isinstance(wait, (int, float)) and not tool.read_only:
                self._tool_hold[(agent, name)] = (
                    self._clock() + max(0.0, min(float(wait), HOLDOFF_CAP_S)), str(msg or f"{name} failed"),
                    str(extra.get("code") or "overloaded"))
            return _err(str(msg or f"{name} failed"), t0, **extra)
        return _err(f"{name}: world {wname} answered HTTP {r.status_code} without "
                    "{\"ok\": ...}", t0)

    async def aclose(self) -> None:
        if self._tasks:
            await asyncio.wait(list(self._tasks), timeout=HOOK_TIMEOUT_S)
        if self._client is not None:
            await self._client.aclose()
