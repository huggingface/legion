"""Codex through `codex app-server` (JSON-RPC 2.0 over stdio, newline-delimited, the
"jsonrpc" field omitted on the wire). One app-server process per working agent.

Protocol as observed with 0.162.1 (tests/fixtures/codex_appserver.jsonl):
`initialize` -> `initialized` notification -> `thread/start` | `thread/resume` ->
`turn/start` (no active turn) or `turn/steer` with `expectedTurnId` (active turn).
Notifications: turn/started, item/started|completed, thread/tokenUsage/updated
(`total` is cumulative for the thread, across resumes; on resume the previous total is
replayed once, tagged with the previous turn id), turn/completed, error.
`thread/resume` in 0.162.1 takes only `threadId` (no `path`); the rollout file under
$CODEX_HOME/sessions is enough to resume (verified: rollout-only CODEX_HOME resumed with
context intact).

Provider-backed models (design.md section 5, 'Providers') are configured with `-c` overrides
on the app-server command line: `model_providers.<name>.{name, base_url, env_key, wire_api}`,
`model_provider`, `model`, and `model_reasoning_effort`. These are config.toml keys (checked
against 0.162.1 with `--strict-config`); passing them per process instead of writing
$CODEX_HOME/config.toml keeps the shared, synced CODEX_HOME untouched. 0.162.1 rejects
`wire_api = "chat"` ("no longer supported"), so the provider must serve the Responses API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from flock.models import REASONING_EFFORT, TokenUsage, TypeSpec
from flock.worker.adapters.base import AdapterEvent, EventQueue, ResumeError, minimal_env

log = logging.getLogger(__name__)


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def find_rollout(home: Path, thread_id: str) -> Path | None:
    """$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<thread_id>.jsonl (newest if several)."""
    hits = sorted((home / "sessions").glob(f"*/*/*/rollout-*-{thread_id}.jsonl"))
    if not hits:
        hits = sorted((home / "sessions").rglob(f"rollout-*-{thread_id}.jsonl"))
    return hits[-1] if hits else None


class RpcError(RuntimeError):
    pass


# Built-in provider ids Codex 0.162.1 refuses to see redefined ("reserved built-in provider IDs").
CODEX_RESERVED_PROVIDERS = frozenset({"openai", "ollama", "lmstudio"})


def codex_provider_id(name: str) -> str:
    """The `model_providers` key for a flock provider: its name, or `flock-<name>` when the name
    is a Codex built-in (a swarmlab-style `openai:` prefix would otherwise stop the app-server)."""
    return f"flock-{name}" if name in CODEX_RESERVED_PROVIDERS else name


def _toml(v: str | int) -> str:
    """A TOML literal for a `-c key=value` override (JSON string escapes are valid TOML)."""
    return str(v) if isinstance(v, int) else json.dumps(v)


def codex_config_args(spec: TypeSpec) -> list[str]:
    """`-c` overrides for the app-server: the provider table and model for a provider-backed
    type, and the reasoning effort for any type that sets one."""
    sets: list[tuple[str, str | int]] = []
    r = spec.resolved_model
    if r is not None:
        p, name = r.provider, codex_provider_id(r.provider_name)
        table: dict[str, str | int] = {"name": r.provider_name, "base_url": p.base_url or "",
                                       "env_key": p.api_key_env, "wire_api": "responses"}
        if p.max_retries is not None:
            table["request_max_retries"] = p.max_retries
            table["stream_max_retries"] = p.max_retries
        if p.timeout_s is not None:
            table["stream_idle_timeout_ms"] = int(p.timeout_s * 1000)
        sets += [(f"model_providers.{name}.{k}", v) for k, v in table.items()]
        sets += [("model_provider", name), ("model", r.model_id)]
    if spec.effective_reasoning is not None:
        sets.append(("model_reasoning_effort", REASONING_EFFORT[spec.effective_reasoning]))
    return [a for k, v in sets for a in ("-c", f"{k}={_toml(v)}")]


def _tokens(b: dict[str, Any]) -> TokenUsage:
    inp = int(b.get("inputTokens") or 0)
    cached = int(b.get("cachedInputTokens") or 0)
    return TokenUsage(
        input=max(0, inp - cached),
        output=int(b.get("outputTokens") or 0),
        cache_read=cached,
        cache_write=int(b.get("cacheWriteInputTokens") or 0),
        reasoning=int(b.get("reasoningOutputTokens") or 0),
    )


def _ms() -> int:
    return int(time.time() * 1000)


class CodexMapper:
    """Maps app-server notifications to AdapterEvents; tracks the active turn."""

    def __init__(self, model: str | None = None, provider: str = "openai") -> None:
        self.model = model
        self.provider = provider
        self.active_turn: str | None = None
        self.started: set[str] = set()

    def _assistant(self, content: list[dict], stop: str = "stop") -> dict[str, Any]:
        return {"role": "assistant", "content": content, "api": "openai-responses",
                "provider": self.provider, "model": self.model, "stopReason": stop, "timestamp": _ms()}

    def _tool(self, item: dict, name: str, args: dict, output: str, is_error: bool) -> list[AdapterEvent]:
        call = self._assistant([{"type": "toolCall", "id": item["id"], "name": name, "arguments": args}],
                               "toolUse")
        res = {"role": "toolResult", "toolCallId": item["id"], "toolName": name,
               "content": [{"type": "text", "text": output}], "isError": is_error, "timestamp": _ms()}
        return [AdapterEvent("message", message=call, raw=item),
                AdapterEvent("message", message=res, raw=item)]

    def on_turn_id(self, turn_id: str) -> list[AdapterEvent]:
        self.active_turn = turn_id
        if turn_id in self.started:
            return []
        self.started.add(turn_id)
        return [AdapterEvent("turn_started", data={"harness_turn_id": turn_id})]

    def on_notification(self, method: str, p: dict[str, Any]) -> list[AdapterEvent]:
        if method == "turn/started":
            return self.on_turn_id(p["turn"]["id"])
        if method == "turn/completed":
            turn = p.get("turn") or {}
            if turn.get("id") == self.active_turn:
                self.active_turn = None
            err = turn.get("error")
            return [AdapterEvent("turn_ended", status=turn.get("status") or "completed",
                                 error=(err or {}).get("message") if isinstance(err, dict) else None,
                                 data={"harness_turn_id": turn.get("id")}, raw=p)]
        if method == "thread/tokenUsage/updated":
            total = (p.get("tokenUsage") or {}).get("total") or {}
            baseline = p.get("turnId") not in self.started
            return [AdapterEvent("usage", model=self.model, usage=_tokens(total), cumulative=True,
                                 baseline=baseline, raw=p)]
        if method == "error":
            err = p.get("error") or {}
            return [AdapterEvent("error", error=str(err.get("message") or err),
                                 data={"willRetry": p.get("willRetry")}, raw=p)]
        if method == "item/completed":
            return self._item(p.get("item") or {})
        return []

    def _item(self, item: dict[str, Any]) -> list[AdapterEvent]:
        t = item.get("type")
        if t == "userMessage":
            text = "\n".join(c.get("text", "") for c in item.get("content") or [] if c.get("type") == "text")
            return [AdapterEvent("message", message={"role": "user", "content": text, "timestamp": _ms()},
                                 raw=item)]
        if t == "agentMessage":
            return [AdapterEvent("message", message=self._assistant([{"type": "text", "text": item.get("text") or ""}]),
                                 model=self.model, raw=item)]
        if t == "reasoning":
            text = "\n".join(str(s) for s in (item.get("summary") or []))
            if not text:
                return []
            return [AdapterEvent("message", message=self._assistant([{"type": "thinking", "thinking": text}]),
                                 raw=item)]
        if t == "commandExecution":
            code = item.get("exitCode")
            return self._tool(item, "shell", {"command": item.get("command")},
                              item.get("aggregatedOutput") or "", code not in (0, None))
        if t == "fileChange":
            return self._tool(item, "apply_patch", {"changes": item.get("changes")},
                              str(item.get("status")), item.get("status") == "failed")
        if t in ("mcpToolCall", "dynamicToolCall"):
            out = item.get("result") if t == "mcpToolCall" else item.get("contentItems")
            failed = item.get("status") == "failed" or bool(item.get("error"))
            return self._tool(item, str(item.get("tool")), item.get("arguments") or {},
                              json.dumps(out, default=str), failed)
        if t == "webSearch":
            return self._tool(item, "web_search", {"query": item.get("query")},
                              json.dumps(item.get("results"), default=str), False)
        return []


class CodexAdapter:
    harness = "codex"

    def __init__(self, workdir: Path, *, home: Path | None = None, codex_bin: str | None = None,
                 extra_globs: tuple[str, ...] = ()) -> None:
        self.workdir = Path(workdir)
        self.home = home or codex_home()
        self.codex_bin = codex_bin or os.environ.get("FLOCK_CODEX_CLI") or "codex"
        self.extra_globs = extra_globs  # e.g. ('*.sqlite*',) if a future version needs more
        self.thread_id: str | None = None
        self.rollout: Path | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._queue = EventQueue()
        self._mapper = CodexMapper()
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._tasks: list[asyncio.Task] = []
        self._stderr: list[str] = []
        self.effort: str | None = os.environ.get("FLOCK_CODEX_EFFORT") or None
        self._turn_effort: str | None = self.effort

    def process_env(self, env: dict[str, str] | None) -> dict[str, str]:
        """Allowlisted worker env + what the worker passes + CODEX_HOME. Never os.environ whole:
        it holds FLOCK_TOKEN and every job secret."""
        return minimal_env(os.environ, {**(env or {}), "CODEX_HOME": str(self.home)})

    # ------------------------------------------------------------------ json-rpc
    async def _write(self, msg: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RpcError("app-server is not running")
        self._proc.stdin.write((json.dumps(msg) + "\n").encode())
        await self._proc.stdin.drain()

    async def request(self, method: str, params: dict[str, Any], timeout: float = 120) -> dict[str, Any]:
        self._next_id += 1
        rid = self._next_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._write({"method": method, "id": rid, "params": params})
        try:
            msg = await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(rid, None)
        if "error" in msg:
            raise RpcError(f"{method}: {msg['error'].get('message') or msg['error']}")
        return msg.get("result") or {}

    async def _read(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            while line := await self._proc.stdout.readline():
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                await self._dispatch(msg)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            self._queue.put(AdapterEvent("error", error=f"app-server reader: {e}"))
        finally:
            for f in self._pending.values():
                if not f.done():
                    f.set_result({"error": {"message": "app-server exited"}})
            if self._proc is not None:
                try:
                    await asyncio.wait_for(self._proc.wait(), 5)
                except (TimeoutError, ProcessLookupError):
                    pass
            self._queue.put(AdapterEvent("exited", data={"stderr": self._stderr[-5:]}))

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        if method is None and "id" in msg:  # response
            fut = self._pending.get(msg["id"])
            if fut and not fut.done():
                fut.set_result(msg)
            return
        if method and "id" in msg:  # server request; approvals are off, so decline anything
            await self._write({"id": msg["id"], "error": {"code": -32601, "message": "unsupported"}})
            return
        if method:
            for ev in self._mapper.on_notification(method, msg.get("params") or {}):
                self._queue.put(ev)

    async def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        while line := await self._proc.stderr.readline():
            self._stderr = (self._stderr + [line.decode(errors="replace").rstrip()])[-20:]

    # ------------------------------------------------------------------ adapter
    async def start(self, workdir: Path, session_id: str | None, spec: TypeSpec, *,
                    env: dict[str, str] | None = None) -> str:
        self.workdir = Path(workdir)
        if session_id and not find_rollout(self.home, session_id):
            raise ResumeError(f"no rollout for thread {session_id} under {self.home}/sessions")
        self._queue = EventQueue()
        model = spec.harness_model
        self._mapper = CodexMapper(model, spec.provider_name or "openai")
        # The type's own reasoning wins over the deployment-wide FLOCK_CODEX_EFFORT.
        self._turn_effort = None if spec.effective_reasoning else self.effort
        self._proc = await asyncio.create_subprocess_exec(
            self.codex_bin, "app-server", *codex_config_args(spec),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=str(self.workdir), limit=64 * 1024 * 1024,
            env=self.process_env(env),
        )
        self._tasks = [asyncio.create_task(self._read()), asyncio.create_task(self._drain_stderr())]
        try:
            await self.request("initialize", {"clientInfo": {"name": "flock", "version": "0.0.1"}})
            await self._write({"method": "initialized", "params": {}})
            params = {"cwd": str(self.workdir), "model": model, "approvalPolicy": "never",
                      "sandbox": "danger-full-access"}
            if session_id:
                try:
                    r = await self.request("thread/resume", {**params, "threadId": session_id,
                                                             "excludeTurns": True})
                except RpcError as e:
                    raise ResumeError(str(e)) from e
            else:
                r = await self.request("thread/start", {**params, "ephemeral": False})
        except BaseException:
            await self.stop()
            raise
        thread = r.get("thread") or {}
        self.thread_id = thread["id"]
        self._mapper.model = r.get("model") or thread.get("model") or model
        self.rollout = Path(thread["path"]) if thread.get("path") else None
        return self.thread_id

    async def send(self, text: str) -> None:
        inp = [{"type": "text", "text": text, "text_elements": []}]
        turn = self._mapper.active_turn
        if turn:
            try:
                await self.request("turn/steer", {"threadId": self.thread_id, "expectedTurnId": turn,
                                                   "input": inp})
                return
            except RpcError as e:  # the turn ended under us; start a new one
                log.info("steer failed (%s); starting a new turn", e)
        params: dict[str, Any] = {"threadId": self.thread_id, "input": inp}
        if self._turn_effort:
            params["effort"] = self._turn_effort
        r = await self.request("turn/start", params)
        for ev in self._mapper.on_turn_id(r["turn"]["id"]):
            self._queue.put(ev)

    async def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            try:
                if proc.stdin:
                    proc.stdin.close()
                await asyncio.wait_for(proc.wait(), 5)
            except (TimeoutError, ProcessLookupError, ConnectionError):
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
        for t in self._tasks:
            try:
                await asyncio.wait_for(t, 5)
            except (TimeoutError, asyncio.CancelledError):
                t.cancel()
        self._tasks = []

    async def events(self) -> AsyncIterator[AdapterEvent]:
        async for ev in self._queue:
            yield ev

    def session_files(self, session_id: str | None) -> list[Path]:
        """The thread's rollout, $CODEX_HOME/sessions/YYYY/MM/DD/rollout-*-<id>.jsonl. That is the
        verified resume set; Codex rebuilds its SQLite index itself. `extra_globs` (relative to
        CODEX_HOME, off by default) adds more files should a future version need them."""
        files: list[Path] = []
        sid = session_id or self.thread_id
        r = find_rollout(self.home, sid) if sid else None
        if r is None and self.rollout and self.rollout.is_file():
            r = self.rollout
        if r:
            files.append(r)
        for g in self.extra_globs:
            files += sorted(p for p in self.home.glob(g) if p.is_file())
        return files
