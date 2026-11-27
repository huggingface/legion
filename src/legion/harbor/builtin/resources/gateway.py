"""The worker's gateway (docs/world.md "Gateway"): a sandbox (e2b, hf-sandbox) calls it instead of holding
service keys. It listens on a port of the worker job, which the controller exposes publicly
(`https://<JOB_ID>--<port>.hf.jobs`).

- Each Trial gets a random token at `bind`, its only credential; the Trial's end (`end`), the
  next bind and `remove` revoke it. The gateway names the agent itself, never trusting its
  headers. The token comes as `X-World-Token` (or a bearer): the jobs proxy drops
  `Authorization`.
- `POST /inference/chat/completions`: to the agent's own upstream (`route`: what the Trial
  would have called) with the real key added, only for the agent's model. The answer streams
  through as it arrives; each call's usage (forced on with `stream_options.include_usage`) is
  emitted as a `usage` event (`watch`), which the worker counts as the agent's, live.
  Once the agent's cost reaches its `loop.while.agent_usd` (`capped`, the worker's), 402,
  except in its final round.
- `GET /world/manifest`, `POST /world/tools/<name>`: to resources.world (transport gateway),
  which adds the world secret and the agent's name after the socket proxy's checks, or for an
  env world (`env`) calls the env's upstream with the agent's own credential.
Every other path is 404, a call without a live token 401. One log line per call; refusals at
WARNING. `export` writes each agent's calls and tokens.
"""

import asyncio
import contextlib
import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from secrets import token_urlsafe
from urllib.parse import urlsplit

import httpx
import uvicorn
from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from legion.harbor.results import write_json
from legion.models import TokenUsage
from legion.world.protocol import TOKEN_HEADER

PORT = 8790
CALLS_MAX = 8  # calls one agent has in flight; more are refused with 429
INFERENCE_MAX = 16 << 20  # a chat request's body; a world call's is resources.world's cap
FORWARD_IN = ("content-type", "accept")
FORWARD_OUT = ("content-type", "retry-after")
log = logging.getLogger("legion.gateway")


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    port: int = Field(default=PORT, ge=0, le=65535)  # 0: any free port (tests)


@dataclass
class Upstream:
    url: str
    key: str
    model: str
    headers: dict


class Server(uvicorn.Server):
    def capture_signals(self):  # the worker's own SIGTERM handling stays
        return contextlib.nullcontext()


def usage(u):
    """One call's usage as Pi counts it (pi-ai parseChunkUsage): `input` without cached tokens,
    reasoning inside `output`."""
    details = u.get("prompt_tokens_details") or {}
    read = next((v for v in (details.get("cached_tokens"), u.get("prompt_cache_hit_tokens"),
                             u.get("cached_tokens")) if v is not None), 0)
    write = details.get("cache_write_tokens") or 0
    return TokenUsage(input=max(0, (u.get("prompt_tokens") or 0) - read - write),
                      output=u.get("completion_tokens") or 0, cache_read=read, cache_write=write,
                      reasoning=(u.get("completion_tokens_details") or {}).get("reasoning_tokens")
                      or 0)


def _chunk_usage(data):
    """The usage in one answer or stream chunk, if any (some servers put it in the choice)."""
    if not isinstance(data, dict):
        return None
    choices = data.get("choices") or [{}]
    u = data.get("usage") or (choices[0].get("usage") if isinstance(choices[0], dict) else None)
    return u if isinstance(u, dict) else None


def _no_constant(name):
    raise ValueError(f"{name} is not JSON")  # NaN, Infinity: json.loads takes them otherwise


async def _body(request, limit):
    """The request body, or None past `limit` bytes (refused before reading when declared)."""
    try:
        if int(request.headers.get("content-length") or 0) > limit:
            return None
    except ValueError:
        return None
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            return None
    return bytes(body)


def _tool_error(content):
    """A World API tool's error code (`{"ok": false, "error": {"code"}}`, sent as a 200), or None.
    Only the code is logged: the message may quote the upstream."""
    try:
        body = json.loads(content)
    except ValueError:
        return None
    if isinstance(body, dict) and body.get("ok") is False and isinstance(body.get("error"), dict):
        return str(body["error"].get("code"))
    return None


class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.options = Options.model_validate(options or {})
        self.secrets = secrets
        self.world = None  # resources.world, set by the worker (legion.worker.harbor)
        self.capped = lambda name: False  # the worker's: the agent's cost reached its cap
        self.emit = None  # watch()'s
        self.tokens, self.names, self.upstreams = {}, {}, {}  # name -> token, token -> name
        self.inflight = Counter()
        self.totals = {}  # name -> {"calls", "tokens"}
        self.metadata, self.url = {}, None
        self.client = self.server = self.task = None

    async def start(self):
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=10))
        self.server = Server(uvicorn.Config(self.app, host="0.0.0.0", port=self.options.port,
                                            interface="asgi3", lifespan="off", log_level="warning",
                                            timeout_graceful_shutdown=5))
        self.task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            if self.task.done():
                self.task.result()
                raise RuntimeError(f"the gateway did not start on port {self.options.port}")
            await asyncio.sleep(0.02)
        port = self.server.servers[0].sockets[0].getsockname()[1]
        job = os.environ.get("JOB_ID")  # an HF Job's; its exposed port as sandboxes reach it
        self.url = f"https://{job}--{port}.hf.jobs" if job else f"http://127.0.0.1:{port}"
        self.metadata = {"url": self.url}
        print(f"Gateway: {self.url}", flush=True)

    def bind(self, name):
        """A new token for the agent's new Trial; its last Trial's stops working."""
        self.names.pop(self.tokens.get(name), None)
        token = self.tokens[name] = token_urlsafe(32)
        self.names[token] = name
        self.secrets.append(token)
        env = {}
        if self.world is not None and self.world.options.transport == "gateway":
            env = {"WORLD_URL": f"{self.url}/world", "WORLD_AGENT": name, "WORLD_TOKEN": token}
        return {"env": env, "allowed_hosts": [urlsplit(self.url).hostname],
                "metadata": {"agent": name}}

    def route(self, name, agent, key, headers):
        """The agent's inference through here: keep its upstream (`agent`, its Trial's Harbor
        AgentConfig after the bind, and `key`) and give the Trial this URL and its token in place
        of the key. `headers`: the upstream's own (the router's bill-to)."""
        self.upstreams[name] = Upstream(agent.base_url.rstrip("/"), key, agent.model_name,
                                        dict(headers))
        agent.env[agent.model_api_key_name] = self.tokens[name]
        agent.base_url = f"{self.url}/inference"

    def end(self, token):
        """A Trial ended: its token stops working (its sandbox may outlive it)."""
        if (name := self.names.pop(token, None)) is not None and self.tokens.get(name) == token:
            del self.tokens[name]

    async def remove(self, names):
        for name in names:
            self.names.pop(self.tokens.pop(name, None), None)
            self.upstreams.pop(name, None)

    async def watch(self, emit):
        self.emit = emit
        await asyncio.Event().wait()

    async def wait(self):
        await asyncio.shield(self.task)

    async def export(self, directory):
        write_json(directory / "usage.json", self.totals)
        return {"usage": "usage.json"}

    async def stop(self):
        if self.server is not None:
            self.server.should_exit = True
            await asyncio.gather(self.task, return_exceptions=True)
        if self.client is not None:
            await self.client.aclose()

    # ------------------------------------------------------------------ calls
    async def app(self, scope, receive, send):
        if scope["type"] != "http":
            return
        response = await self._handle(Request(scope, receive), time.monotonic())
        try:
            await response(scope, receive, send)
        finally:
            if isinstance(response, Relay):
                await response.close()

    def _refuse(self, name, path, status, why):
        log.warning("gateway %s %s: %d %s", name or "-", path, status, why)
        return JSONResponse({"error": {"message": why, "code": status}}, status,
                            headers={"Retry-After": "1"} if status == 429 else None)

    async def _handle(self, request, started):
        scope = request.scope
        path = (scope.get("raw_path") or scope["path"].encode()).decode("latin-1")
        token = (request.headers.get(TOKEN_HEADER)
                 or request.headers.get("authorization", "").removeprefix("Bearer ").strip())
        name = self.names.get(token) if token else None
        if name is None:
            return self._refuse(None, path, 401, "no valid gateway token")
        inference = path == "/inference/chat/completions" and request.method == "POST"
        world = (path.startswith("/world/") and self.world is not None
                 and self.world.options.transport == "gateway")
        if not (inference or world):
            return self._refuse(name, path, 404, "not a gateway route")
        if self.inflight[name] >= CALLS_MAX:
            return self._refuse(name, path, 429, f"more than {CALLS_MAX} calls in flight")
        self.inflight[name] += 1
        relay = None
        try:
            body = await _body(request, self.world.upload_max if world else INFERENCE_MAX)
            if body is None:
                return self._refuse(name, path, 413, "request body too large")
            if world:
                query = scope.get("query_string", b"").decode("latin-1")
                route = path.removeprefix("/world") + (f"?{query}" if query else "")
                status, headers, content = await self.world.call(
                    name, request.method, route, request.headers, body)
                error = _tool_error(content) if status == 200 else None
                (log.info if status < 400 and not error else log.warning)(
                    "gateway %s %s: %d%s, %.0f ms", name, path, status,
                    f" {error}" if error else "", (time.monotonic() - started) * 1000)
                return Response(content, status, headers=headers)
            relay = await self._inference(name, request, body, started)
            return relay
        finally:
            if not isinstance(relay, Relay):
                self.inflight[name] -= 1

    async def _inference(self, name, request, body, started):
        path = "/inference/chat/completions"
        up = self.upstreams.get(name)
        if up is None:
            return self._refuse(name, path, 403, "no inference route for this agent")
        if self.capped(name):
            return self._refuse(name, path, 402, "the agent's budget (loop.while.agent_usd) "
                                                 "is used up")
        try:
            payload = json.loads(body, parse_constant=_no_constant)
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            return self._refuse(name, path, 400, "the body is not a JSON object")
        if payload.get("model") != up.model:
            return self._refuse(name, path, 403, f"model must be {up.model}")
        stream = bool(payload.get("stream"))
        if stream:  # usage on the last chunk, also when the agent did not ask for it
            payload["stream_options"] = {**(payload.get("stream_options") or {}),
                                         "include_usage": True}
        headers = {k: v for k, v in request.headers.items() if k in FORWARD_IN}
        headers |= {**up.headers, "Authorization": f"Bearer {up.key}"}
        try:
            r = await self.client.send(self.client.build_request(
                "POST", f"{up.url}/chat/completions", json=payload, headers=headers), stream=True)
        except httpx.HTTPError as exc:
            return self._refuse(name, path, 502, f"upstream unreachable ({type(exc).__name__})")
        return Relay(self, name, r, stream, started)

    def _count(self, name, used, status, started):
        """One inference call's end: its usage to the agent's counters (emit) and the totals."""
        total = self.totals.setdefault(name, {"calls": 0, "tokens": TokenUsage().model_dump()})
        total["calls"] += 1
        if used is not None and used != TokenUsage():
            total["tokens"] = (TokenUsage(**total["tokens"]) + used).model_dump()
            if self.emit is not None:
                self.emit(name, "usage", used.model_dump())
        tokens = (f"{used.input + used.cache_read} in, {used.output} out" if used is not None
                  else "no usage")
        (log.info if isinstance(status, int) and status < 400 else log.warning)(
            "gateway %s /inference/chat/completions: %s, %s, %.0f ms", name, status, tokens,
            (time.monotonic() - started) * 1000)


class Relay(StreamingResponse):
    """An upstream answer, sent on as it arrives. Its usage is counted before the body ends (the
    agent's turn may end on it). `close` (the app's, always) ends the upstream call too, so a
    client that goes away stops the generation; its usage is then unknown. A stream whose usage
    arrived was answered in full (usage comes on its last chunk): a client that hangs up after
    it, before `[DONE]`, is not a broken-off call."""

    def __init__(self, gateway, name, response, stream, started):
        self.gateway, self.name, self.r, self.stream, self.started = (
            gateway, name, response, stream, started)
        self.last, self.done, self.closed = None, False, False
        super().__init__(self.relay(), response.status_code,
                         {k: response.headers[k] for k in FORWARD_OUT if k in response.headers})

    async def relay(self):
        buf = b""
        async for chunk in self.r.aiter_bytes():
            buf += chunk
            if self.stream:  # SSE: parse whole `data:` lines on the side
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                        self.last = self._usage(line[5:]) or self.last
            yield chunk
        if not self.stream:
            self.last = self._usage(buf)
        self.done = True
        await self.close()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.gateway.inflight[self.name] -= 1
        answered = self.done or (self.stream and self.last is not None)
        self.gateway._count(self.name, self.last, self.status_code if answered else "broken off",
                            self.started)
        await self.r.aclose()

    @staticmethod
    def _usage(data):
        try:
            u = _chunk_usage(json.loads(data))
            return usage(u) if u else None
        except (ValueError, TypeError):  # not JSON, or not numbers: no usage from it
            return None
