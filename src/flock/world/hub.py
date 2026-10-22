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
  the harness env and from every tool call's env).
- Hooks the manifest lists are POSTed to `/hooks` in the background: `agent_start` at an agent's
  first prompt on this worker, `turn` at each turn end, `agent_done` at unassign. Best effort.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import httpx

from flock.models import WorldSpec
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
)

log = logging.getLogger(__name__)

BUILTIN_MODULE = "flock:world"
SECTION_START = "<!-- flock:world -->"
SECTION_END = "<!-- /flock:world -->"
RETRIES = 2  # extra attempts after a 429 / 503
RETRY_CAP_S = 10.0
HOOK_TIMEOUT_S = 5.0


def _err(msg: str, t0: float) -> Result:
    return Result(json.dumps({"error": msg}, ensure_ascii=False), True, time.monotonic() - t0)


def _retry_after(r: httpx.Response) -> float:
    try:
        return max(0.0, min(float(r.headers.get("retry-after", "1")), RETRY_CAP_S))
    except ValueError:
        return 1.0


class WorldHub:
    def __init__(self, *, swarm: str, type_name: str = "", env: Mapping[str, str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 runtime: Callable[[str], Any] | None = None,
                 timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self.swarm, self.type = swarm, type_name
        self._env = env if env is not None else os.environ
        self._transport = transport
        self._runtime = runtime or (lambda name: None)
        self.timeout = timeout
        self.spec: WorldSpec | None = None
        self.names: list[str] = []
        self._client: httpx.AsyncClient | None = None
        self._started: set[str] = set()  # agents whose agent_start hook was sent
        self._shown: dict[str, str] = {}  # agent -> world version its instructions show
        self._acked: dict[str, str] = {}  # agent -> world version of its last finished turn
        self._tasks: set[asyncio.Task] = set()

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
            self.configure(None, [])
            return
        raw = dict(raw)
        names = raw.pop("tools", None)
        try:
            spec = WorldSpec.model_validate(raw)
        except ValueError as e:
            log.warning("heartbeat: unreadable world block: %s", e)
            return
        self.configure(spec, names if isinstance(names, list) else [])

    def tooldefs(self) -> tuple[ToolDef, ...]:
        if not self.active:
            return ()
        m = self.spec.manifest  # type: ignore[union-attr]
        return tuple(ToolDef(name=t.name, description=t.description, parameters=t.parameters,
                             module=BUILTIN_MODULE, function=t.name, builtin=f"world:{t.name}")
                     for n in self.names if (t := m.get(n)) is not None)

    def toolset_for(self, agent: str, base: Toolset, gateway: Any = None) -> Toolset:
        """`base` (folder and comms tools of this turn) plus the agent's world tools. Built-in
        calls are routed by prefix: `world:*` here, the rest to `base`'s own handler.
        `gateway` (the comms hub) serves them to `flock-tool`."""
        defs = self.tooldefs()
        if not defs:
            return base
        have = set(base.names())
        prev = base.builtin_call

        async def route(entry: str, args: dict[str, Any]) -> Result:
            if entry.startswith("world:"):
                return await self.call(agent, entry, args)
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
        names = ", ".join(f"`{d.name}`" for d in self.tooldefs())
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
        body = {"event": event, "agent": agent, "type": self.type, "swarm": self.swarm,
                "turn": turn or "", "data": data or {}}
        try:
            t = asyncio.get_running_loop().create_task(self._post_hook(s.endpoint, body))
        except RuntimeError:
            return
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _post_hook(self, endpoint: str, body: dict[str, Any]) -> None:
        try:
            r = await self._http().post(f"{endpoint}/hooks", json=body, headers=self._auth(),
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
        """One world tool call by `agent`. Never raises: errors are results the model reads."""
        t0 = time.monotonic()
        name = entry.split(":", 1)[-1]
        s = self.spec
        if not self.active or s is None or name not in self.names:
            return _err(f"{name} is not enabled for this agent", t0)
        tool = s.manifest.get(name)  # type: ignore[union-attr]
        if tool is None:
            return _err(f"the world has no tool {name!r}", t0)
        if err := check_args(tool.parameters, args):
            return _err(f"{name}: {err}", t0)
        headers = {**self._auth(), AGENT_HEADER: agent, TYPE_HEADER: self.type,
                   SWARM_HEADER: self.swarm, REQUEST_HEADER: uuid.uuid4().hex}
        rt = self._runtime(agent)
        if rt is not None and getattr(rt, "turn_id", None):
            headers[TURN_HEADER] = rt.turn_id
        wname = s.manifest.name  # type: ignore[union-attr]
        for attempt in range(RETRIES + 1):
            try:
                r = await self._http().post(f"{s.endpoint}/tools/{name}", json={"args": args},
                                            headers=headers, timeout=self.timeout)
            except httpx.TimeoutException:
                return _err(f"{name}: world {wname} did not answer within {self.timeout:g} s", t0)
            except httpx.HTTPError as e:
                return _err(f"{name}: world {wname} is unreachable ({type(e).__name__})", t0)
            if r.status_code in (429, 503) and attempt < RETRIES:
                await asyncio.sleep(_retry_after(r))
                continue
            break
        if r.status_code == 401:
            return _err(f"{name}: world {wname} refused flock's token (401)", t0)
        try:
            body = r.json()
        except ValueError:
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
            return _err(str(msg or f"{name} failed"), t0)
        return _err(f"{name}: world {wname} answered HTTP {r.status_code} without "
                    "{\"ok\": ...}", t0)

    async def aclose(self) -> None:
        if self._tasks:
            await asyncio.wait(list(self._tasks), timeout=HOOK_TIMEOUT_S)
        if self._client is not None:
            await self._client.aclose()
