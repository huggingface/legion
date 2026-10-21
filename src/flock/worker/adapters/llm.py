"""`harness: llm`: a plain chat-completions loop for small open models (docs/llm-harness.md).

No external process: the loop runs in the worker's event loop. One flock prompt is one turn:
the model is called with the transcript and a single `bash` tool; each tool call runs in the
agent's workdir and its result goes back; the turn ends when the model answers without a tool
call (and nothing was steered in), or before a call past `max_turns` / `max_budget_usd`.
Inspired by swarmlab's `LLMAgent`; no swarmlab import.

- Session: `<home>/sessions/<agent>/<session id>.jsonl`, one OpenAI chat message per line
  (system, user, assistant, tool), appended as the turn runs. Resume reloads it; the system
  message is rebuilt from the preamble and the workdir's AGENTS.md on every start, and at a
  turn start when either changed (a re-apply's comms protocols, docs/comms.md). Keys starting
  with `flock_` are ours and never sent.
- Exactly one system message, first (LFM2's template requires it): compaction notes and
  steered prompts are user messages.
- Context bounding (pixel run 3: a chars/4 estimate let a 3000-token budget reach 7.9k and
  overflow 8192, because JSON tool output runs ~2 chars/token): before each call the next
  prompt is projected as the server's `prompt_tokens` for the previous call plus the new
  characters / `chars_per_token` (default 2.5, calibrated online from reported tokens). Over
  `context_tokens`, compaction keeps the system message, one `flock_summary` user note (how
  many messages were dropped) and the newest messages within half the budget: whole earlier
  prompts first, then the oldest steps of the current turn (assistant + tool results, never
  split; the newest two kept), then shortened old tool outputs. A 400 that names the context
  length compacts aggressively (last prompt + two steps) and retries once; otherwise the turn
  ends `error`, reason `context_overflow`. Tool output is capped at `tool_output_chars`
  (default 8000). No model-written summary. The file is rewritten; the full history stays in
  the agent's flock session log.
- Thinking: `reasoning_content` / `reasoning`, or inline `<think>...</think>` (also without the
  opening tag, which LFM2's template emits itself), becomes a pi `thinking` block and is never
  sent back; its tokens are in the server's usage.
- `tool_choice` (type setting, else the alias's `extra`, else auto): `required` or `bash`
  forces a call on every request. With the `done` tool offered (`done_tool`, default on for
  `required`) the turn runs until the model calls `done` (its summary is the final assistant
  message) or a limit; without it a forced turn is one action: the forced call, its tool run,
  done (plus one more step per steered prompt). Constrained output is what makes small thinking
  models fast and reliable (2.5 s and 20/20 valid vs 25 s and 7/20 free-form, LFM2.5-2.6B).
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
- Tools folder (docs/tools.md): with a `toolbox` (set by AgentRuntime), each turn takes one
  Toolset and offers its tools next to `bash` / `done` for the whole turn; a call runs in a
  subprocess (flock.tools.runner). Turn events carry `tools_version`.
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
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

from flock.controller.prices import Price, price_for
from flock.models import TokenUsage, TypeSpec
from flock.tools.toolset import Toolset
from flock.worker.adapters.base import AdapterEvent, EventQueue, ResumeError, minimal_env, pi_usage

log = logging.getLogger(__name__)

ENV_DENY = ("FLOCK_TOKEN",)  # never reaches tool processes, even when listed in secrets/env
INSTRUCTIONS = "AGENTS.md"
DEFAULT_CONTEXT_TOKENS = 32_000
COMPACT_TARGET = 0.5  # share of context_tokens kept after compaction
OLD_TOOL_OUTPUT_CHARS = 1_000  # what compaction's last stage leaves of an old tool output
DEFAULT_CHARS_PER_TOKEN = 2.5  # JSON tool output runs ~2 chars/token (pixel run 3); prose ~4
CPT_MIN, CPT_MAX = 1.0, 6.0  # bounds of the online calibration
CPT_EMA = 0.3  # weight of the newest observation
KEEP_STEPS = 2  # model steps (assistant + its tool results) an overflow compaction keeps
CONTEXT_ERROR = re.compile(r"(?i)context|maximum.{0,40}length|too long|too many tokens|max_model_len")
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 3
BACKOFF_S = 1.0
BACKOFF_MAX_S = 30.0
DEFAULT_MAX_TOKENS = 4096
BASH_TIMEOUT_S = 120.0
BASH_TIMEOUT_MAX_S = 600.0
OUTPUT_LIMIT = 8_000  # default `tool_output_chars`: stored/sent tool output, head and tail
STOP_GRACE = 5.0
FORCED_DEFAULT_MAX_TURNS = 50  # forced tool choice with `done` and no max_turns

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


DONE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "done",
        "description": "Call this when the task for this prompt is finished. Ends your turn.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "One line: what you did."}},
        },
    },
}
DONE_PREAMBLE = "\nWhen the task is done, call the `done` tool with a one-line summary."
TOOLS_PREAMBLE = ("\nBesides `bash` you have task tools, offered as functions next to it; their "
                  "descriptions say what they do. Call them directly, not through bash.")


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


def wire_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in wire_messages(messages))


def estimate_tokens(messages: list[dict[str, Any]], chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Characters sent / chars_per_token (tool schemas excluded); a heuristic, not a tokenizer.
    The adapter calibrates chars_per_token from the server's reported prompt tokens."""
    return int(wire_chars(messages) / chars_per_token)


def wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The request's `messages`: `flock_*` keys dropped. There is exactly one system message,
    first (LFM2's template, among others, requires it): compaction notes and steered prompts
    are user messages."""
    return [{k: v for k, v in m.items() if not k.startswith("flock_")} for m in messages]


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


def _steps(tail: list[dict[str, Any]]) -> list[list[int]]:
    """Indexes of each model step in `tail`: an assistant message and the tool results after it."""
    out: list[list[int]] = []
    for i, m in enumerate(tail):
        if m.get("role") == "assistant":
            out.append([i])
        elif m.get("role") == "tool" and out:
            out[-1].append(i)
    return out


def compact(messages: list[dict[str, Any]], limit: int, workdir: Path, *,
            chars_per_token: float = DEFAULT_CHARS_PER_TOKEN, over: bool | None = None,
            aggressive: bool = False) -> list[dict[str, Any]] | None:
    """The compacted transcript, or None when it fits (`over` None: by estimate; the adapter
    passes its own projection) or cannot shrink.

    1. Drop whole earlier prompts: keep from the oldest user message whose tail fits in
       COMPACT_TARGET of the limit (the newest user message at least).
    2. Inside that tail (one long turn), drop the oldest model steps (an assistant message and
       its tool results, never split) until it fits, keeping the newest KEEP_STEPS.
    3. Shorten old tool outputs to OLD_TOOL_OUTPUT_CHARS.
    `aggressive` (after a context-length error): keep only the newest user message and the
    newest KEEP_STEPS steps, then shorten."""
    def est(ms: list[dict[str, Any]]) -> int:
        return estimate_tokens(ms, chars_per_token)

    if over is False or (over is None and not aggressive and est(messages) <= limit):
        return None
    system = messages[0]
    dropped_before = sum(int(m.get("flock_dropped") or 0) for m in messages if m.get("flock_summary"))
    body = [m for m in messages[1:] if not m.get("flock_summary")]
    target = int(limit * COMPACT_TARGET)
    starts = [i for i, m in enumerate(body) if m.get("role") == "user"]
    if not starts:
        return None
    cut = starts[-1] if aggressive else next(
        (i for i in starts if est([system, *body[i:]]) <= target), starts[-1])
    tail = [dict(m) for m in body[cut:]]
    dropped = cut
    # 2. old steps inside the kept tail
    steps = _steps(tail)
    drop: set[int] = set()
    for step in steps[:max(0, len(steps) - KEEP_STEPS)]:
        if not aggressive and est([system, *(m for k, m in enumerate(tail) if k not in drop)]) <= target:
            break
        drop.update(step)
    kept = [m for k, m in enumerate(tail) if k not in drop]
    dropped += len(drop)
    # 3. shorten old tool outputs (all but the newest two messages), oldest first
    for m in kept[:-2]:
        if not aggressive and est([system, *kept]) <= limit:
            break
        c = m.get("content")
        if m.get("role") == "tool" and isinstance(c, str) and len(c) > OLD_TOOL_OUTPUT_CHARS:
            m["content"] = c[:OLD_TOOL_OUTPUT_CHARS] + f"\n[... {len(c) - OLD_TOOL_OUTPUT_CHARS} chars removed]"
    if dropped == 0 and kept == body:
        return None
    n = dropped_before + dropped
    note = {"role": "user", "flock_summary": True, "flock_dropped": n,
            "content": f"[flock] {n} earlier message(s) of this conversation were removed to stay "
                       f"under context_tokens={limit}. Your files in {workdir} are intact; re-read "
                       "anything you need from them."}
    return [system, note, *kept]


_THINK = re.compile(r"^\s*(?:<think>)?(.*?)</think>\s*", re.DOTALL)


def split_thinking(content: Any) -> tuple[str | None, Any]:
    """(thinking, visible content) for a reply that carries its reasoning inline: `<think>...
    </think>answer`, or `...</think>answer` when the chat template already opened the tag (LFM2
    ends its prompt with `<think>`). A reply cut off inside `<think>` is all thinking. Servers
    run with a reasoning parser (vLLM `--reasoning-parser`) send `reasoning_content` instead."""
    if not isinstance(content, str):
        return None, content
    if m := _THINK.match(content):
        return m.group(1).strip() or None, content[m.end():]
    if content.lstrip().startswith("<think>"):
        return content.lstrip()[len("<think>"):].strip() or None, ""
    return None, content


TOOL_CHOICES = ("auto", "required", "none")


def wire_tool_choice(choice: str) -> str | dict[str, Any]:
    """`auto` | `required` | `none` as is; any other value names a tool to force."""
    return choice if choice in TOOL_CHOICES else {"type": "function", "function": {"name": choice}}


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
    def __init__(self, message: str, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status, self.body = status, body

    @property
    def context_overflow(self) -> bool:
        return self.status == 400 and bool(CONTEXT_ERROR.search(self.body))


ERROR_BODY_CHARS = 500
_SECRETS = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+|\b(?:hf|sk|ghp|gho|xox[bp])[_-][A-Za-z0-9_-]{8,}")


def redact(text: str, *secrets: str | None) -> str:
    """Error text safe to log: the given secrets, bearer tokens and common key shapes (hf_,
    sk-, ghp_, ...) replaced by `[redacted]`."""
    for sec in secrets:
        if sec and len(sec) >= 4:
            text = text.replace(sec, "[redacted]")
    return _SECRETS.sub(lambda m: (m.group(1) or "") + "[redacted]", text)


# --------------------------------------------------------------------------- adapter


class LLMAdapter:
    harness = "llm"

    def __init__(self, workdir: Path, *, agent: str | None = None, home: Path | None = None,
                 prices: dict[str, Price] | None = None, backoff_s: float = BACKOFF_S,
                 transport: httpx.AsyncBaseTransport | None = None,
                 toolbox: Callable[[], Toolset] | None = None) -> None:
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
        self.chars_per_token = DEFAULT_CHARS_PER_TOKEN  # calibrated online
        self._last_prompt: tuple[int, int] | None = None  # (reported prompt tokens, chars sent)
        # Tools folder (docs/tools.md): the worker's current Toolset, taken once per turn, so a
        # new version applies from the next turn and never mid-turn. Set by AgentRuntime.
        self.toolbox = toolbox
        self._turn_tools: Toolset | None = None

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
    def forced(self) -> bool:
        """The effective tool_choice forces a call (`required` or a named tool)."""
        spec = self._spec
        r = spec.resolved_model if spec else None
        choice = (spec.tool_choice if spec else None) or ((r.extra or {}).get("tool_choice") if r else None)
        return choice is not None and choice not in ("auto", "none")

    @property
    def done_enabled(self) -> bool:
        """`done` is offered: the type's `done_tool`, by default on when `required` is forced.
        With it a forced turn runs until `done` (or a limit); without it a forced turn is one
        action."""
        spec = self._spec
        if spec is None:
            return False
        if spec.done_tool is not None:
            return spec.done_tool
        r = spec.resolved_model
        choice = spec.tool_choice or ((r.extra or {}).get("tool_choice") if r else None)
        return choice == "required"

    def projected_tokens(self) -> int:
        """Prompt size of the next request: the server's count for the previous request plus
        the estimate for what was added since; the estimate alone when there is none."""
        chars = wire_chars(self.messages)
        if self._last_prompt is None:
            return int(chars / self.chars_per_token)
        tokens, sent = self._last_prompt
        return tokens + int(max(0, chars - sent) / self.chars_per_token)

    def _calibrate(self, prompt_tokens: int, chars: int) -> None:
        if prompt_tokens <= 0 or chars <= 0:
            return
        self._last_prompt = (prompt_tokens, chars)
        obs = chars / prompt_tokens
        cpt = (1 - CPT_EMA) * self.chars_per_token + CPT_EMA * obs
        self.chars_per_token = min(CPT_MAX, max(CPT_MIN, cpt))

    def _compact(self, *, aggressive: bool = False) -> bool:
        over = None if aggressive else self.projected_tokens() > self.context_tokens
        c = compact(self.messages, self.context_tokens, self.workdir,
                    chars_per_token=self.chars_per_token, over=over, aggressive=aggressive)
        if c is None:
            return False
        log.info("%s: compacted the transcript%s (%d -> %d messages)", self.agent,
                 " after a context-length error" if aggressive else "", len(self.messages), len(c))
        self.messages = c
        self._last_prompt = None  # the transcript changed: back to the calibrated estimate
        self._rewrite()
        return True

    @property
    def context_tokens(self) -> int:
        return (self._spec.context_tokens if self._spec else None) or DEFAULT_CONTEXT_TOKENS

    # ------------------------------------------------------------------ transcript file
    def system_prompt(self) -> str:
        text = PREAMBLE.format(agent=self.agent, workdir=self.workdir)
        if self.done_enabled:
            text += DONE_PREAMBLE
        if self.toolbox is not None and self.toolbox().tools:
            text += TOOLS_PREAMBLE
        f = self.workdir / INSTRUCTIONS
        if f.is_file():
            text += "\n\n" + f.read_text()
        return text

    def _refresh_system(self) -> None:
        """Rebuild the system message at a turn start when AGENTS.md or the tools changed (a
        re-apply's protocols or tools reach the next turn, not only the next start)."""
        if not self.messages or self.messages[0].get("role") != "system":
            return
        text = self.system_prompt()
        if self.messages[0].get("content") != text:
            self.messages[0] = {"role": "system", "content": text}
            self._last_prompt = None
            with contextlib.suppress(OSError, AssertionError):
                self._rewrite()

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
        self.chars_per_token = spec.chars_per_token or DEFAULT_CHARS_PER_TOKEN
        self._last_prompt = None
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
        if self._turn_tools is not None and self._turn_tools.version is not None:
            data.setdefault("tools_version", self._turn_tools.version)
        self._queue.put(AdapterEvent("turn_ended", status=status, error=error, data=data))

    @property
    def folder_tools(self) -> Toolset | None:
        """The turn's tools-folder tools, when it has any."""
        ts = self._turn_tools
        return ts if ts is not None and ts.tools else None

    async def _turn(self, texts: list[str]) -> None:
        spec = self._spec
        assert spec is not None and spec.resolved_model is not None
        model = spec.resolved_model.model_id
        self.in_turn = True
        calls, cost = 0, 0.0
        overflow_retried = False
        # a forced model that never calls `done` would loop until turn_timeout: cap it
        max_turns = spec.max_turns
        if max_turns is None and self.forced and self.done_enabled:
            max_turns = FORCED_DEFAULT_MAX_TURNS
        limits = {"max_turns": max_turns, "max_budget_usd": spec.max_budget_usd}
        self._turn_tools = self.toolbox() if self.toolbox is not None else None
        started: dict[str, Any] = {}
        if self._turn_tools is not None and self._turn_tools.version is not None:
            started["tools_version"] = self._turn_tools.version
        if self._turn_tools is not None and self._turn_tools.error:
            started["tools_error"] = self._turn_tools.error
        self._queue.put(AdapterEvent("turn_started", data=started))
        self._refresh_system()
        try:
            self._user(texts)
            while True:
                hit = ("max_turns" if max_turns is not None and calls >= max_turns
                       else "max_budget_usd" if spec.max_budget_usd is not None and cost > spec.max_budget_usd
                       else None)
                if hit:
                    self._end("error", f"{hit} exceeded ({limits[hit]}): {calls} model call(s), "
                                       f"${cost:.6f}; stopped", reason=hit, subtype=f"error_{hit}",
                              num_turns=calls)
                    return
                self._compact()  # when the projected prompt is over context_tokens
                sent = wire_chars(self.messages)
                try:
                    data = await self._complete()
                except CallError as e:
                    if e.context_overflow and not overflow_retried and self._compact(aggressive=True):
                        overflow_retried = True  # one retry per step, on the shrunk transcript
                        continue
                    reason = "context_overflow" if e.context_overflow else "api_error"
                    self._end("error", str(e), reason=reason, num_turns=calls)
                    return
                overflow_retried = False
                calls += 1
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                usage = chat_usage(data.get("usage"))
                self._calibrate(usage.input + usage.cache_read, sent)
                if usage != TokenUsage():
                    self._queue.put(AdapterEvent("usage", model=model, usage=usage))
                    p = price_for(model, self.prices)
                    cost += (usage.input * p.input + usage.output * p.output
                             + usage.cache_read * p.cache_read + usage.cache_write * p.cache_write) / 1e6
                tool_calls = await self._assistant(msg, choice.get("finish_reason"), usage, model)
                done: dict[str, Any] | None = None
                for call in tool_calls:
                    if done is not None:
                        self._tool_result(call, "[skipped: done was called]", True)
                    elif call["name"] == "done" and self.done_enabled:
                        done = call
                        self._tool_result(call, "ok", False)
                    else:
                        await self._run_tool(call)
                if done is not None:
                    self._finish(done, model)
                    self._end("completed", num_turns=calls)
                    return
                queued = self._drain()
                self._user(queued)
                if not queued and (not tool_calls or (self.forced and not self.done_enabled)):
                    # forced tool choice without `done`: one action per prompt (the model could
                    # never answer in text)
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
        inline, content = split_thinking(msg.get("content"))
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
        elif parsed := (text_tool_calls(content)
                        or (text_tool_calls(content, "done") if self.done_enabled else [])
                        or next((c for n in (self.folder_tools.names() if self.folder_tools else [])
                                 if (c := text_tool_calls(content, n))), [])):
            n = sum(1 for m in self.messages if m.get("role") == "assistant")
            stored = {"role": "assistant", "content": None, "flock_from_text": content, "tool_calls": [
                {"id": f"text_{n}_{i}", "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                for i, c in enumerate(parsed)]}
            calls = [{"id": f"text_{n}_{i}", "name": c["name"], "arguments": c["arguments"]}
                     for i, c in enumerate(parsed)]
        self._append(stored)
        blocks: list[dict[str, Any]] = []
        thinking = msg.get("reasoning_content") or msg.get("reasoning") or inline
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

    def _finish(self, call: dict[str, Any], model: str) -> None:
        """`done` was called: its summary is the turn's final assistant message."""
        args = call["arguments"] if isinstance(call["arguments"], dict) else {}
        summary = args.get("summary")
        text = summary.strip() if isinstance(summary, str) and summary.strip() else "done"
        self._append({"role": "assistant", "content": text})
        r = self._spec.resolved_model if self._spec else None
        self._queue.put(AdapterEvent("message", message={
            "role": "assistant", "content": [{"type": "text", "text": text}], "api": "openai-completions",
            "provider": r.provider_name if r else None, "model": model, "usage": pi_usage(TokenUsage()),
            "stopReason": "stop", "fromDoneTool": True, "timestamp": _ms()}))

    def _tool_result(self, call: dict[str, Any], out: str, err: bool, **extra: Any) -> None:
        self._append({"role": "tool", "tool_call_id": call["id"], "content": out})
        self._queue.put(AdapterEvent("message", message={
            "role": "toolResult", "toolCallId": call["id"], "toolName": call["name"],
            "content": [{"type": "text", "text": out}], "isError": err, "timestamp": _ms(), **extra}))

    async def _run_tool(self, call: dict[str, Any]) -> None:
        args = call["arguments"]
        folder = self.folder_tools
        if isinstance(args, str):
            out, err = args, True
        elif folder is not None and folder.get(call["name"]) is not None:
            r = await folder.call(call["name"], args, agent=self.agent, workdir=self.workdir,
                                  env=self.tool_env)
            spec = self._spec
            out = _truncate(r.output, (spec.tool_output_chars if spec else None) or OUTPUT_LIMIT)
            self._tool_result(call, out, r.error, toolsVersion=folder.version,
                              durationMs=int(r.duration_s * 1000))
            return
        elif call["name"] != "bash":
            names = ["bash", *(["done"] if self.done_enabled else []),
                     *(folder.names() if folder else [])]
            tools = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
            out, err = f"unknown tool {call['name']!r}; the tools are {tools}", True
        elif not isinstance(args.get("command"), str) or not args["command"].strip():
            out, err = "bash needs a non-empty `command` string", True
        else:
            out, err = await self._bash(args["command"], args.get("timeout"))
        self._tool_result(call, out, err)

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
        spec = self._spec
        text = _truncate(out.decode(errors="replace"),
                         (spec.tool_output_chars if spec else None) or OUTPUT_LIMIT)
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
        tools = [BASH_TOOL, DONE_TOOL] if self.done_enabled else [BASH_TOOL]
        if self.folder_tools is not None:
            tools += [t.openai() for t in self.folder_tools.tools]
        body: dict[str, Any] = {"model": r.model_id, "messages": wire_messages(self.messages),
                                "tools": tools,
                                "tool_choice": "auto", "temperature": 0,
                                "max_tokens": DEFAULT_MAX_TOKENS, **(r.extra or {})}
        body.pop("stream", None)
        if spec.tool_choice is not None:  # the type's setting wins over the alias's extra
            body["tool_choice"] = wire_tool_choice(spec.tool_choice)
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
                # redact before cutting, so a secret across the cut cannot leak in part
                body_text = redact(resp.text[:4 * ERROR_BODY_CHARS], self._key)[:ERROR_BODY_CHARS]
                err = f"HTTP {resp.status_code}: {body_text[:300]}"
                retrying = (resp.status_code == 429 or resp.status_code >= 500) and attempt < retries
                # the server's reason, in the event stream (run 1: 29 x 400 in 67k calls, unlogged)
                self._queue.put(AdapterEvent("error", error=f"HTTP {resp.status_code} from "
                                             f"{url}: {body_text}",
                                             data={"status": resp.status_code, "attempt": attempt + 1,
                                                   "retrying": retrying}))
                log.warning("%s: HTTP %d from the model server: %s", self.agent, resp.status_code,
                            body_text[:200])
                if resp.status_code != 429 and resp.status_code < 500:
                    raise CallError(err, resp.status_code, body_text)
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
