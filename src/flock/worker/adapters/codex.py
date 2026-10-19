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

from flock.controller.prices import Price, price_for
from flock.models import TokenUsage, TypeSpec
from flock.worker.adapters.base import AdapterEvent, EventQueue, ResumeError

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


TOOL_ITEMS = ("commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall", "webSearch")


class CodexMapper:
    """Maps app-server notifications to AdapterEvents; tracks the active turn.

    `max_turns` / `max_budget_usd` (per flock turn, which may fold several steered prompts) are
    enforced here; the app-server has no per-turn call or spend limit (0.163.0 schema: only a
    thread-goal `tokenBudget`, in tokens). One `thread/tokenUsage/updated` for the active turn
    is one model call; its `last` usage is priced with the flock price table. When Codex would
    call the model again past a limit (the call started a tool, or ended without a final answer)
    the adapter sends `turn/interrupt`, and the turn ends `error` with `data.reason`
    max_turns|max_budget_usd, as Pi and Claude report it. A steer into a turn that has used its
    allowance is held and sent as the next turn once this one completes.
    """

    max_turns: int | None = None
    max_budget_usd: float | None = None
    prices: dict[str, Price] | None = None  # None: the global table

    def __init__(self, model: str | None = None) -> None:
        self.model = model
        self.active_turn: str | None = None
        self.started: set[str] = set()
        self._reset()

    def _reset(self) -> None:
        self.calls, self.cost = 0, 0.0  # model calls and flock-priced cost in this flock turn
        self.limit: str | None = None  # the limit that made us interrupt this turn
        self.abort_requested = False  # set when `limit` is first hit; the adapter interrupts
        self._saw_tool = self._saw_final = False  # what the current model call produced so far

    def _assistant(self, content: list[dict], stop: str = "stop") -> dict[str, Any]:
        return {"role": "assistant", "content": content, "api": "openai-responses",
                "provider": "openai", "model": self.model, "stopReason": stop, "timestamp": _ms()}

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
        self._reset()
        return [AdapterEvent("turn_started", data={"harness_turn_id": turn_id})]

    def _exceeded(self) -> str | None:
        """The limit another model call would break, if any."""
        if self.max_turns is not None and self.calls >= self.max_turns:
            return "max_turns"
        if self.max_budget_usd is not None and self.cost > self.max_budget_usd:
            return "max_budget_usd"
        return None

    def exhausted(self) -> bool:
        """The active turn may not call the model again (a steer would make it)."""
        return self.active_turn is not None and (self.limit is not None
                                                 or self._exceeded() is not None)

    def _stop(self, hit: str | None) -> None:
        if hit and self.limit is None:
            self.limit, self.abort_requested = hit, True

    def _call(self, u: TokenUsage) -> None:
        """One model call of the active turn finished (its token count arrived)."""
        self._stop(self._exceeded())  # a call we would have stopped happened anyway
        self.calls += 1
        if self.max_budget_usd is not None:
            pr = price_for(self.model or "", self.prices)
            self.cost += (u.input * pr.input + u.output * pr.output + u.cache_read * pr.cache_read
                          + u.cache_write * pr.cache_write) / 1_000_000
        # Codex calls again after tools, or when the call ended without a final answer
        if self._saw_tool or not self._saw_final:
            self._stop(self._exceeded())
        self._saw_tool = self._saw_final = False

    def _note(self, item: dict[str, Any]) -> None:
        t = item.get("type")
        if t in TOOL_ITEMS:
            self._saw_tool = True
        elif t == "agentMessage" and item.get("phase") in (None, "final_answer"):
            self._saw_final = True

    def on_notification(self, method: str, p: dict[str, Any]) -> list[AdapterEvent]:
        if method == "turn/started":
            return self.on_turn_id(p["turn"]["id"])
        if method == "turn/completed":
            turn = p.get("turn") or {}
            if turn.get("id") == self.active_turn:
                self.active_turn = None
            if turn.get("id") in self.started and self.limit:
                limit, lim = self.limit, getattr(self, self.limit)
                self.limit = None
                return [AdapterEvent("turn_ended", status="error", raw=p,
                                     error=f"{limit} exceeded ({lim}): {self.calls} model call(s), "
                                           f"${self.cost:.6f}; interrupted",
                                     data={"harness_turn_id": turn.get("id"), "reason": limit,
                                           "subtype": f"error_{limit}", "num_turns": self.calls})]
            err = turn.get("error")
            return [AdapterEvent("turn_ended", status=turn.get("status") or "completed",
                                 error=(err or {}).get("message") if isinstance(err, dict) else None,
                                 data={"harness_turn_id": turn.get("id")}, raw=p)]
        if method == "thread/tokenUsage/updated":
            total = (p.get("tokenUsage") or {}).get("total") or {}
            baseline = p.get("turnId") not in self.started
            if not baseline and p.get("turnId") == self.active_turn:  # `last`: this call only
                self._call(_tokens((p.get("tokenUsage") or {}).get("last") or {}))
            return [AdapterEvent("usage", model=self.model, usage=_tokens(total), cumulative=True,
                                 baseline=baseline, raw=p)]
        if method == "error":
            err = p.get("error") or {}
            return [AdapterEvent("error", error=str(err.get("message") or err),
                                 data={"willRetry": p.get("willRetry")}, raw=p)]
        if method == "item/started":
            self._note(p.get("item") or {})
            return []
        if method == "item/completed":
            self._note(p.get("item") or {})
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
                 extra_globs: tuple[str, ...] = (), prices: dict[str, Price] | None = None) -> None:
        self.workdir = Path(workdir)
        self.home = home or codex_home()
        self.codex_bin = codex_bin or os.environ.get("FLOCK_CODEX_CLI") or "codex"
        self.extra_globs = extra_globs  # e.g. ('*.sqlite*',) if a future version needs more
        self.prices = prices  # for max_budget_usd; None: the global table
        self._held: list[str] = []  # steers held until an exhausted turn ends
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
            if self._held:
                self._queue.put(AdapterEvent("error", error="app-server exited with held prompts",
                                             data={"dropped_prompts": self._held}))
                self._held = []
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
            ended = False
            for ev in self._mapper.on_notification(method, msg.get("params") or {}):
                self._queue.put(ev)
                ended = ended or ev.kind == "turn_ended"
            m = self._mapper
            if m.abort_requested and m.active_turn:
                m.abort_requested = False
                log.warning("codex: %s exceeded; interrupting turn %s", m.limit, m.active_turn)
                self._tasks.append(asyncio.create_task(self._interrupt(m.active_turn)))
            if ended and self._held:
                held, self._held = self._held, []
                self._tasks.append(asyncio.create_task(self._send_held(held)))

    async def _interrupt(self, turn_id: str) -> None:
        try:
            await self.request("turn/interrupt", {"threadId": self.thread_id, "turnId": turn_id},
                               timeout=30)
        except Exception as e:  # noqa: BLE001 - the turn may have ended on its own
            self._queue.put(AdapterEvent("error", error=f"turn/interrupt after limit: {e}"))

    async def _send_held(self, held: list[str]) -> None:
        """Steers that arrived after the turn used its allowance: the next turn."""
        try:
            for text in held:
                await self.send(text)
        except Exception as e:  # noqa: BLE001
            self._queue.put(AdapterEvent("error", error=f"held prompt not sent: {e}",
                                         data={"dropped_prompts": held}))

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
        self._mapper = CodexMapper(spec.model)
        self._proc = await asyncio.create_subprocess_exec(
            self.codex_bin, "app-server",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=str(self.workdir), limit=64 * 1024 * 1024,
            env={**os.environ, **(env or {}), "CODEX_HOME": str(self.home)},
        )
        self._tasks = [asyncio.create_task(self._read()), asyncio.create_task(self._drain_stderr())]
        try:
            await self.request("initialize", {"clientInfo": {"name": "flock", "version": "0.0.1"}})
            await self._write({"method": "initialized", "params": {}})
            params = {"cwd": str(self.workdir), "model": spec.model, "approvalPolicy": "never",
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
        self._mapper.model = r.get("model") or thread.get("model") or spec.model
        self.rollout = Path(thread["path"]) if thread.get("path") else None
        self._mapper.max_turns, self._mapper.max_budget_usd = spec.max_turns, spec.max_budget_usd
        self._mapper.prices = self.prices
        return self.thread_id

    async def send(self, text: str) -> None:
        inp = [{"type": "text", "text": text, "text_elements": []}]
        turn = self._mapper.active_turn
        if turn and self._mapper.exhausted():  # a steer would make a call past the limit
            self._held.append(text)
            return
        if turn:
            try:
                await self.request("turn/steer", {"threadId": self.thread_id, "expectedTurnId": turn,
                                                   "input": inp})
                return
            except RpcError as e:  # the turn ended under us; start a new one
                log.info("steer failed (%s); starting a new turn", e)
        params: dict[str, Any] = {"threadId": self.thread_id, "input": inp}
        if self.effort:
            params["effort"] = self.effort
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
