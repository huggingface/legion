"""`harness: llm`: a plain chat-completions loop for small open models (docs/llm-harness.md).

No external process: the loop runs in the worker's event loop. One flock prompt is one turn:
the model is called with the transcript and a single `bash` tool; each tool call runs in the
agent's workdir and its result goes back; the turn ends when the model answers without a tool
call (and nothing was steered in), or before a call past `max_turns` / `max_budget_usd`.
Inspired by swarmlab's `LLMAgent`; no swarmlab import.

- Session: `<home>/sessions/<agent>/<session id>.jsonl`, one OpenAI chat message per line
  (system, user, assistant, tool), appended as the turn runs. Resume reloads it; the system
  message is rebuilt from the preamble and the workdir's AGENTS.md on every start. Keys starting
  with `flock_` are ours and never sent.
- Compaction: when the transcript's estimated size (characters / 4) exceeds `context_tokens`,
  keep the system message, one `flock_summary` note (how many messages were dropped) and the
  newest messages that fit in half the budget, starting at a user message; then, if still over,
  shorten old tool outputs. No model-written summary. The file is rewritten; the full history
  stays in the agent's flock session log.
- Tool calls: native `tool_calls`, or, for small models that write the call as text, a reply
  that is only `{"tool": "bash", "command": ...}` (optionally fenced) or a pythonic
  `[bash(command="...")]`. Such a reply is stored as a proper tool call (`flock_from_text`), so
  later requests show the model the native form.
- Steering: text sent during a turn is queued and added as the next user message after the
  current tool results (or after a final answer, which then continues the turn).
- Requests: non-streamed `POST <base_url>/chat/completions`, `temperature: 0`,
  `max_tokens: 4096`, then the alias's `extra` on top (it may override both; `stream` is
  ignored). 429, 5xx and transport errors are retried `max_retries` times (default 3) with
  jittered exponential backoff (Retry-After honoured, capped at 30 s). `reasoning` is ignored.
- Usage: one `usage` event per model call, not cumulative; `input` excludes cached prompt tokens.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import logging
import os
import random
import re
import signal
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import httpx

from flock.controller.prices import Price, price_for
from flock.models import TokenUsage, TypeSpec
from flock.worker.adapters.base import AdapterEvent, EventQueue, ResumeError, minimal_env, pi_usage

log = logging.getLogger(__name__)

ENV_DENY = ("FLOCK_TOKEN",)  # never reaches tool processes, even when listed in secrets/env
INSTRUCTIONS = "AGENTS.md"
DEFAULT_CONTEXT_TOKENS = 32_000
COMPACT_TARGET = 0.5  # share of context_tokens kept after compaction
OLD_TOOL_OUTPUT_CHARS = 1_000  # what compaction's second stage leaves of an old tool output
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 3
BACKOFF_S = 1.0
BACKOFF_MAX_S = 30.0
DEFAULT_MAX_TOKENS = 4096
BASH_TIMEOUT_S = 120.0
BASH_TIMEOUT_MAX_S = 600.0
OUTPUT_LIMIT = 16_000  # characters of tool output sent back to the model (head and tail)
STOP_GRACE = 5.0

PREAMBLE = """You are {agent}, an agent working in the directory {workdir}.
You have one tool, `bash`. It runs a shell command in that directory and returns the output \
(stdout and stderr) and the exit code. Files you write there are kept between prompts.
Use the tool to act. If you cannot make a tool call, reply with only this JSON and nothing else:
{{"tool": "bash", "command": "<the command>"}}
When the task is done, reply in plain text without a tool call."""

BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in your working directory. Returns stdout and stderr "
                       "(truncated if long) and the exit code.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run."},
                "timeout": {"type": "number",
                            "description": "Seconds before the command is killed (default 120, max 600)."},
            },
            "required": ["command"],
        },
    },
}


def llm_home() -> Path:
    return Path(os.environ.get("FLOCK_LLM_HOME") or Path.home() / ".flock-llm")


def harness_env(env: Mapping[str, str] | None, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The allow-listed worker environment plus what the worker passes, never FLOCK_TOKEN."""
    out = minimal_env(os.environ if base is None else base, env or {})
    for k in ENV_DENY:
        out.pop(k, None)
    return out


# --------------------------------------------------------------------------- tool calls in text

_FENCE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)
_WRAP = re.compile(r"^(?:<tool_call>|<\|tool_call_start\|>)\s*(.*?)\s*"
                   r"(?:</tool_call>|<\|tool_call_end\|>)$", re.DOTALL)


def _json_call(s: str, name: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(s)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    args = obj.get("arguments", obj.get("parameters", obj.get("args")))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if args is None:
        args = {k: v for k, v in obj.items() if k not in ("tool", "name")}
    if (obj.get("tool") or obj.get("name")) != name or not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def _pythonic_calls(s: str, name: str) -> list[dict[str, Any]]:
    """`[bash(command="ls")]` or `bash(command="ls")` (LFM2's tool-call syntax), parsed with
    `ast` and literal values only (nothing is evaluated)."""
    try:
        node = ast.parse(s, mode="eval").body
    except SyntaxError:
        return []
    items = node.elts if isinstance(node, ast.List) else [node]
    out = []
    for it in items:
        if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == name
                and not it.args):
            return []
        try:
            args = {kw.arg: ast.literal_eval(kw.value) for kw in it.keywords if kw.arg}
        except ValueError:
            return []
        out.append({"name": name, "arguments": args})
    return out


def text_tool_calls(content: Any, name: str = "bash") -> list[dict[str, Any]]:
    """Calls of tool `name` that a model wrote as text instead of `tool_calls`: the whole reply
    (after stripping a code fence or `<tool_call>` tags) must be a JSON object naming the tool
    (`{"tool": "bash", "command": ...}`, or `{"name": ..., "arguments": {...}}`) or a pythonic
    call list. Anything else is an answer, so prose that mentions JSON is safe."""
    if not isinstance(content, str):
        return []
    s = content.strip()
    for rx in (_FENCE, _WRAP):
        if m := rx.match(s):
            s = m.group(1).strip()
    if s.startswith("{") and s.endswith("}"):
        call = _json_call(s, name)
        return [call] if call else []
    if s.endswith((")", "]")):
        return _pythonic_calls(s, name)
    return []


# --------------------------------------------------------------------------- transcript helpers


def chat_usage(u: dict[str, Any] | None) -> TokenUsage:
    u = u or {}
    prompt = int(u.get("prompt_tokens") or 0)
    cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    reasoning = int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    return TokenUsage(input=max(0, prompt - cached), output=int(u.get("completion_tokens") or 0),
                      cache_read=cached, reasoning=reasoning)


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """chars / 4 over what is sent (tool schemas excluded); a heuristic, not a tokenizer."""
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in wire_messages(messages)) // 4


def wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The request's `messages`: `flock_*` keys dropped, leading system messages (the system
    prompt and a compaction note) merged into one, since many chat templates allow only one."""
    out: list[dict[str, Any]] = []
    for m in messages:
        clean = {k: v for k, v in m.items() if not k.startswith("flock_")}
        if clean.get("role") == "system" and out and out[-1]["role"] == "system" and len(out) == 1:
            out[0] = {**out[0], "content": f"{out[0]['content']}\n\n{clean.get('content') or ''}"}
        else:
            out.append(clean)
    return out


def repair(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give every assistant tool call a tool result (an interrupted turn leaves calls without
    one, and servers reject such a transcript)."""
    out: list[dict[str, Any]] = []
    open_ids: list[str] = []
    for m in [*messages, None]:
        if m is not None and m.get("role") == "tool":
            with contextlib.suppress(ValueError):
                open_ids.remove(m.get("tool_call_id"))
            out.append(m)
            continue
        out += [{"role": "tool", "tool_call_id": i, "content": "[interrupted: no result]"} for i in open_ids]
        open_ids = []
        if m is None:
            break
        out.append(m)
        if m.get("role") == "assistant":
            open_ids = [c.get("id") for c in m.get("tool_calls") or []]
    return out


def compact(messages: list[dict[str, Any]], limit: int, workdir: Path) -> list[dict[str, Any]] | None:
    """The compacted transcript, or None when it fits or cannot shrink. See module docstring."""
    if estimate_tokens(messages) <= limit:
        return None
    system = messages[0]
    dropped_before = sum(int(m.get("flock_dropped") or 0) for m in messages if m.get("flock_summary"))
    body = [m for m in messages[1:] if not m.get("flock_summary")]
    target = int(limit * COMPACT_TARGET)
    starts = [i for i, m in enumerate(body) if m.get("role") == "user"]
    if not starts:
        return None
    cut = next((i for i in starts if estimate_tokens([system, *body[i:]]) <= target), starts[-1])
    kept = [dict(m) for m in body[cut:]]
    # second stage: shorten old tool outputs (all but the newest two messages), oldest first
    for m in kept[:-2]:
        if estimate_tokens([system, *kept]) <= limit:
            break
        c = m.get("content")
        if m.get("role") == "tool" and isinstance(c, str) and len(c) > OLD_TOOL_OUTPUT_CHARS:
            m["content"] = c[:OLD_TOOL_OUTPUT_CHARS] + f"\n[... {len(c) - OLD_TOOL_OUTPUT_CHARS} chars removed]"
    if cut == 0 and kept == body:
        return None
    n = dropped_before + cut
    note = {"role": "system", "flock_summary": True, "flock_dropped": n,
            "content": f"[flock] {n} earlier message(s) of this conversation were removed to stay "
                       f"under context_tokens={limit}. Your files in {workdir} are intact; re-read "
                       "anything you need from them."}
    return [system, note, *kept]


def _ms() -> int:
    return int(time.time() * 1000)


def _truncate(s: str, limit: int = OUTPUT_LIMIT) -> str:
    if len(s) <= limit:
        return s
    head, tail = limit * 2 // 3, limit // 3
    return f"{s[:head]}\n[... {len(s) - head - tail} chars truncated ...]\n{s[-tail:]}"


def _parse_args(raw: Any) -> dict[str, Any] | str:
    """Native tool-call arguments -> dict, or an error string for the model."""
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw or "{}")
    except ValueError:
        return f"arguments are not valid JSON: {str(raw)[:200]}"
    return v if isinstance(v, dict) else f"arguments must be a JSON object, got {str(raw)[:200]}"


class CallError(RuntimeError):
    pass


# --------------------------------------------------------------------------- adapter


class LLMAdapter:
    harness = "llm"

    def __init__(self, workdir: Path, *, agent: str | None = None, home: Path | None = None,
                 prices: dict[str, Price] | None = None, backoff_s: float = BACKOFF_S,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.workdir = Path(workdir)
        self.agent = agent or self.workdir.name
        self.home = Path(home) if home else llm_home()
        self.prices = prices  # for max_budget_usd; None: the global table
        self.backoff_s = backoff_s
        self.transport = transport
        self.session_id: str | None = None
        self.messages: list[dict[str, Any]] = []
        self.env: dict[str, str] = {}
        self.tool_env: dict[str, str] = {}
        self.in_turn = False
        self._spec: TypeSpec | None = None
        self._key: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._queue = EventQueue()
        self._inbox: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._tool_proc: asyncio.subprocess.Process | None = None
        self._exited = True

    @property
    def session_dir(self) -> Path:
        return self.home / "sessions" / self.agent

    def session_path(self, session_id: str) -> Path:
        return self.session_dir / f"{session_id}.jsonl"

    def session_files(self, session_id: str | None) -> list[Path]:
        sid = session_id or self.session_id
        if not sid:
            return []
        p = self.session_path(sid)
        return [p] if p.is_file() else []

    @property
    def context_tokens(self) -> int:
        return (self._spec.context_tokens if self._spec else None) or DEFAULT_CONTEXT_TOKENS

    # ------------------------------------------------------------------ transcript file
    def system_prompt(self) -> str:
        text = PREAMBLE.format(agent=self.agent, workdir=self.workdir)
        f = self.workdir / INSTRUCTIONS
        if f.is_file():
            text += "\n\n" + f.read_text()
        return text

    def _load(self, path: Path) -> list[dict[str, Any]]:
        out = []
        try:
            lines = path.read_text().splitlines()
        except OSError as e:
            raise ResumeError(f"cannot read {path}: {e}") from e
        for n, line in enumerate(lines, 1):
            try:
                m = json.loads(line)
            except ValueError:
                log.warning("%s: skipping unreadable line %d of %s", self.agent, n, path)
                continue
            if isinstance(m, dict) and m.get("role"):
                out.append(m)
        return out

    def _rewrite(self) -> None:
        assert self.session_id
        path = self.session_path(self.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
        tmp.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in self.messages))
        os.replace(tmp, path)

    def _append(self, m: dict[str, Any]) -> None:
        assert self.session_id
        self.messages.append(m)
        with open(self.session_path(self.session_id), "a") as f:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ adapter
    async def start(self, workdir: Path, session_id: str | None, spec: TypeSpec, *,
                    env: dict[str, str] | None = None) -> str:
        r = spec.resolved_model
        if r is None or not r.provider.base_url:
            raise RuntimeError("harness llm needs a provider-backed model on an openai_compat provider")
        self.workdir = Path(workdir)
        messages: list[dict[str, Any]] = []
        if session_id:
            path = self.session_path(session_id)
            if not path.is_file():
                raise ResumeError(f"cannot resume llm session {session_id}: no file {path}")
            messages = self._load(path)
        self._spec = spec
        self.session_id = session_id or str(uuid.uuid4())
        system = {"role": "system", "content": self.system_prompt()}
        if messages and messages[0].get("role") == "system" and not messages[0].get("flock_summary"):
            messages[0] = system
        else:
            messages.insert(0, system)
        self.messages = repair(messages)
        self._rewrite()
        self.env = harness_env(env)
        self._key = self.env.get(r.provider.api_key_env)
        # tools get the agent's environment minus the provider key (the model never needs it)
        self.tool_env = {k: v for k, v in self.env.items() if k != r.provider.api_key_env}
        if spec.effective_reasoning is not None:
            log.debug("%s: reasoning=%s is ignored by the llm harness; use extra", self.agent,
                      spec.effective_reasoning)
        self._client = httpx.AsyncClient(timeout=r.provider.timeout_s or DEFAULT_TIMEOUT_S,
                                         transport=self.transport)
        self._queue = EventQueue()
        self._inbox = asyncio.Queue()
        self._exited = False
        self.in_turn = False
        self._task = asyncio.create_task(self._loop(), name=f"llm-{self.agent}")
        return self.session_id

    async def send(self, text: str) -> None:
        """Starts a turn when idle; queued as the next user message when busy."""
        if self._task is None or self._task.done():
            raise RuntimeError("the llm loop is not running")
        self._inbox.put_nowait(text)

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, STOP_GRACE)
        self._kill_tool()
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None
        if not self._exited:
            self._exited = True
            self._queue.put(AdapterEvent("exited", data={"returncode": 0}))

    async def events(self) -> AsyncIterator[AdapterEvent]:
        async for ev in self._queue:
            yield ev

    # ------------------------------------------------------------------ the loop
    def _drain(self) -> list[str]:
        out = []
        while not self._inbox.empty():
            out.append(self._inbox.get_nowait())
        return out

    async def _loop(self) -> None:
        try:
            while True:
                first = await self._inbox.get()
                await self._turn([first, *self._drain()])
        except asyncio.CancelledError:
            pass
        except Exception as e:  # a bug here must end the burst, not hang it
            log.exception("%s: llm loop failed", self.agent)
            if self.in_turn:
                self.in_turn = False
                self._queue.put(AdapterEvent("turn_ended", status="error", error=f"{type(e).__name__}: {e}"))
            self._queue.put(AdapterEvent("error", error=f"llm loop failed: {type(e).__name__}: {e}"))
            if not self._exited:
                self._exited = True
                self._queue.put(AdapterEvent("exited", data={"returncode": 1}))

    def _user(self, texts: list[str]) -> None:
        for t in texts:
            self._append({"role": "user", "content": t})
            self._queue.put(AdapterEvent("message", message={
                "role": "user", "content": [{"type": "text", "text": t}], "timestamp": _ms()}))

    def _end(self, status: str, error: str | None = None, **data: Any) -> None:
        self.in_turn = False
        self._queue.put(AdapterEvent("turn_ended", status=status, error=error, data=data))

    async def _turn(self, texts: list[str]) -> None:
        spec = self._spec
        assert spec is not None and spec.resolved_model is not None
        model = spec.resolved_model.model_id
        self.in_turn = True
        calls, cost = 0, 0.0
        self._queue.put(AdapterEvent("turn_started"))
        try:
            self._user(texts)
            while True:
                hit = ("max_turns" if spec.max_turns is not None and calls >= spec.max_turns
                       else "max_budget_usd" if spec.max_budget_usd is not None and cost > spec.max_budget_usd
                       else None)
                if hit:
                    self._end("error", f"{hit} exceeded ({getattr(spec, hit)}): {calls} model call(s), "
                                       f"${cost:.6f}; stopped", reason=hit, subtype=f"error_{hit}",
                              num_turns=calls)
                    return
                if (c := compact(self.messages, self.context_tokens, self.workdir)) is not None:
                    log.info("%s: compacted the transcript (%d -> %d messages)", self.agent,
                             len(self.messages), len(c))
                    self.messages = c
                    self._rewrite()
                try:
                    data = await self._complete()
                except CallError as e:
                    self._end("error", str(e), reason="api_error", num_turns=calls)
                    return
                calls += 1
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                usage = chat_usage(data.get("usage"))
                if usage != TokenUsage():
                    self._queue.put(AdapterEvent("usage", model=model, usage=usage))
                    p = price_for(model, self.prices)
                    cost += (usage.input * p.input + usage.output * p.output
                             + usage.cache_read * p.cache_read + usage.cache_write * p.cache_write) / 1e6
                tool_calls = await self._assistant(msg, choice.get("finish_reason"), usage, model)
                for call in tool_calls:
                    await self._run_tool(call)
                queued = self._drain()
                self._user(queued)
                if not tool_calls and not queued:
                    self._end("completed", num_turns=calls)
                    return
        except asyncio.CancelledError:
            self.messages = repair(self.messages)
            with contextlib.suppress(OSError):
                self._rewrite()
            self._end("interrupted")
            raise

    async def _assistant(self, msg: dict[str, Any], finish: str | None, usage: TokenUsage,
                         model: str) -> list[dict[str, Any]]:
        """Store the reply, emit it as a pi message; returns the tool calls to run as
        {id, name, arguments: dict | error str}."""
        content = msg.get("content")
        native = [c for c in msg.get("tool_calls") or [] if isinstance(c, dict)]
        calls: list[dict[str, Any]] = []
        # an empty reply is stored as "" (servers reject an assistant message with neither)
        stored: dict[str, Any] = {"role": "assistant", "content": content if content is not None else ""}
        if native:
            stored = {"role": "assistant", "content": content, "tool_calls": native}
            for c in native:
                fn = c.get("function") or {}
                calls.append({"id": c.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                              "name": fn.get("name"), "arguments": _parse_args(fn.get("arguments"))})
                c.setdefault("id", calls[-1]["id"])
        elif parsed := text_tool_calls(content):
            n = sum(1 for m in self.messages if m.get("role") == "assistant")
            stored = {"role": "assistant", "content": None, "flock_from_text": content, "tool_calls": [
                {"id": f"text_{n}_{i}", "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                for i, c in enumerate(parsed)]}
            calls = [{"id": f"text_{n}_{i}", "name": c["name"], "arguments": c["arguments"]}
                     for i, c in enumerate(parsed)]
        self._append(stored)
        blocks: list[dict[str, Any]] = []
        thinking = msg.get("reasoning_content") or msg.get("reasoning")
        if isinstance(thinking, str) and thinking:
            blocks.append({"type": "thinking", "thinking": thinking})
        if isinstance(content, str) and content:
            blocks.append({"type": "text", "text": content})
        blocks += [{"type": "toolCall", "id": c["id"], "name": c["name"],
                    "arguments": c["arguments"] if isinstance(c["arguments"], dict) else {}} for c in calls]
        r = self._spec.resolved_model if self._spec else None
        pi_msg = {"role": "assistant", "content": blocks, "api": "openai-completions",
                  "provider": r.provider_name if r else None, "model": model, "usage": pi_usage(usage),
                  "stopReason": "toolUse" if calls else ("length" if finish == "length" else "stop"),
                  "timestamp": _ms()}
        if calls and not native:
            pi_msg["toolCallsFromText"] = True
        self._queue.put(AdapterEvent("message", message=pi_msg))
        return calls

    async def _run_tool(self, call: dict[str, Any]) -> None:
        args = call["arguments"]
        if isinstance(args, str):
            out, err = args, True
        elif call["name"] != "bash":
            out, err = f"unknown tool {call['name']!r}; the only tool is bash", True
        elif not isinstance(args.get("command"), str) or not args["command"].strip():
            out, err = "bash needs a non-empty `command` string", True
        else:
            out, err = await self._bash(args["command"], args.get("timeout"))
        self._append({"role": "tool", "tool_call_id": call["id"], "content": out})
        self._queue.put(AdapterEvent("message", message={
            "role": "toolResult", "toolCallId": call["id"], "toolName": call["name"],
            "content": [{"type": "text", "text": out}], "isError": err, "timestamp": _ms()}))

    async def _bash(self, command: str, timeout: Any) -> tuple[str, bool]:
        try:
            limit = min(float(timeout), BASH_TIMEOUT_MAX_S) if timeout else BASH_TIMEOUT_S
        except (TypeError, ValueError):
            limit = BASH_TIMEOUT_S
        self.workdir.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command, cwd=str(self.workdir), env=self.tool_env,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        self._tool_proc = proc
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), limit)
        except TimeoutError:
            self._kill_tool()
            return f"[command timed out after {limit:g}s and was killed]", True
        except asyncio.CancelledError:  # stop() mid-tool: the whole process group goes
            self._kill_tool()
            raise
        finally:
            self._tool_proc = None if self._tool_proc is proc else self._tool_proc
        text = _truncate(out.decode(errors="replace"))
        return f"{text}{'' if not text or text.endswith(chr(10)) else chr(10)}[exit code {proc.returncode}]", \
            proc.returncode != 0

    def _kill_tool(self) -> None:
        proc = self._tool_proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)

    async def _complete(self) -> dict[str, Any]:
        spec = self._spec
        assert spec is not None and spec.resolved_model is not None and self._client is not None
        r = spec.resolved_model
        body: dict[str, Any] = {"model": r.model_id, "messages": wire_messages(self.messages),
                                "tools": [BASH_TOOL], "tool_choice": "auto", "temperature": 0,
                                "max_tokens": DEFAULT_MAX_TOKENS, **(r.extra or {})}
        body.pop("stream", None)
        url = f"{r.provider.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        retries = r.provider.max_retries if r.provider.max_retries is not None else DEFAULT_MAX_RETRIES
        for attempt in range(retries + 1):
            retry_after = None
            try:
                resp = await self._client.post(url, json=body, headers=headers)
            except httpx.TransportError as e:
                err = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError as e:
                        raise CallError(f"the server sent a non-JSON reply: {resp.text[:200]}") from e
                    if not isinstance(data, dict) or not data.get("choices"):
                        raise CallError(f"the reply has no choices: {json.dumps(data)[:300]}")
                    return data
                err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code != 429 and resp.status_code < 500:
                    raise CallError(err)
                with contextlib.suppress(TypeError, ValueError):
                    retry_after = float(resp.headers.get("retry-after"))
            if attempt == retries:
                raise CallError(f"{err} (gave up after {retries + 1} attempts)")
            delay = min(BACKOFF_MAX_S, self.backoff_s * 2 ** attempt) * (0.5 + random.random() / 2)
            if retry_after is not None:
                delay = max(delay, min(retry_after, BACKOFF_MAX_S))
            log.info("%s: %s; retry %d/%d in %.1fs", self.agent, err[:120], attempt + 1, retries, delay)
            await asyncio.sleep(delay)
        raise CallError("unreachable")
