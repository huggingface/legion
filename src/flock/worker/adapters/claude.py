"""Claude Code through the Python Agent SDK (`ClaudeSDKClient`, streaming-input mode).

The installed SDK exposes every option we need (`resume`, `session_id`, `model`,
`max_turns`, `max_budget_usd`, `permission_mode`, `cwd`, `env`, `cli_path`), so there is
no hand-rolled stream-json fallback. `cli_path` pins the CLI on PATH (the image installs
@anthropic-ai/claude-code at the pinned version) instead of the SDK's bundled binary.

Observed with 2.1.296 (tests/fixtures/claude_stream.jsonl): a prompt sent while a turn is
running is folded into that turn (one `result`), so the mapper opens a turn on the first
send and closes it on `result`; any assistant/user message after a `result` opens a new
turn (a queued prompt the CLI ran separately). `modelUsage` and `total_cost_usd` on
`result` are cumulative for the session, across resumes, so usage is reported as
cumulative and the runtime takes deltas.

Environment: the SDK starts the CLI with the worker's whole `os.environ` merged under
`options.env`, which would hand FLOCK_TOKEN and every job secret to the agent. So `cli_path`
points at a tiny exec wrapper (`env_wrapper`) that drops every name the worker did not mean to
pass (listed in FLOCK_HARNESS_ENV_DROP) and execs the real CLI; the SDK's own variables
(CLAUDE_CODE_ENTRYPOINT, CLAUDE_AGENT_SDK_VERSION, ...) are not on that list and survive.

Provider-backed models (`anthropic` providers only): ANTHROPIC_API_KEY is the provider key
(ANTHROPIC_AUTH_TOKEN, sent as `Authorization: Bearer`, when the provider sets `auth: bearer`) and
ANTHROPIC_BASE_URL its `base_url` when set; `timeout_s` and `max_retries` go to API_TIMEOUT_MS
and CLAUDE_CODE_MAX_RETRIES (names present in the 2.1.296 binary). `reasoning` maps to the SDK's
`effort` (low/medium/high) or `thinking={"type": "disabled"}` (off).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from flock.models import TokenUsage, TypeSpec
from flock.worker.adapters.base import (
    AdapterEvent,
    EventQueue,
    ResumeError,
    minimal_env,
    pi_usage,
)

log = logging.getLogger(__name__)

MAX_SANITIZED_LENGTH = 200
_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9]")


# --------------------------------------------------------------------------- session paths


def claude_home() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def _simple_hash(s: str) -> str:
    """The CLI's JS fallback hash (32-bit, base 36). The Bun-built CLI uses Bun.hash for
    paths over 200 chars, which we cannot reproduce, so lookups also try a prefix match."""
    h = 0
    for ch in s:
        h = ((h << 5) - h + ord(ch)) & 0xFFFFFFFF
    if h >= 0x80000000:
        h -= 0x100000000
    h = abs(h)
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while True:
        out = digits[h % 36] + out
        h //= 36
        if h == 0:
            return out


def encode_cwd(cwd: str | Path) -> str:
    """Directory name Claude Code uses under projects/ for a working directory."""
    name = str(cwd)
    sanitized = _SANITIZE_RE.sub("-", name)
    if len(sanitized) <= MAX_SANITIZED_LENGTH:
        return sanitized
    return f"{sanitized[:MAX_SANITIZED_LENGTH]}-{_simple_hash(name)}"


def project_dir(home: Path, cwd: str | Path) -> Path | None:
    """The existing projects/<encoded cwd> dir for `cwd`, tolerating long-path hash drift."""
    projects = home / "projects"
    cwd = os.path.realpath(cwd)
    exact = projects / encode_cwd(cwd)
    if exact.is_dir():
        return exact
    enc = encode_cwd(cwd)
    if len(enc) > MAX_SANITIZED_LENGTH:
        prefix = enc[:MAX_SANITIZED_LENGTH] + "-"
        for d in projects.glob(glob_escape(prefix) + "*"):
            if d.is_dir():
                return d
    return None


def glob_escape(s: str) -> str:
    return re.sub(r"([*?\[])", r"[\1]", s)


# --------------------------------------------------------------------------- environment

DROP_VAR = "FLOCK_HARNESS_ENV_DROP"
_WRAPPER = """#!{python} -I
# Written by flock's ClaudeAdapter: exec the Claude CLI without the names in {drop}.
import os, sys
drop = set(filter(None, os.environ.get({drop!r}, "").split(",")))
drop.add({drop!r})
os.execve({cli!r}, [{cli!r}, *sys.argv[1:]], {{k: v for k, v in os.environ.items() if k not in drop}})
"""
_wrappers: dict[str, str] = {}


def env_wrapper(cli: str) -> str:
    """Path of an executable that runs `cli` with the names in FLOCK_HARNESS_ENV_DROP removed
    from its environment (one per CLI path per process)."""
    if cli not in _wrappers:
        d = Path(tempfile.mkdtemp(prefix="flock-claude-"))
        w = d / "claude"
        w.write_text(_WRAPPER.format(python=sys.executable, cli=cli, drop=DROP_VAR))
        w.chmod(0o700)
        _wrappers[cli] = str(w)
    return _wrappers[cli]


def claude_env(spec: TypeSpec, env: dict[str, str]) -> dict[str, str]:
    """What the CLI should see: `minimal_env` of the worker's environment plus `env`, with a
    provider-backed type's key moved to ANTHROPIC_API_KEY and its base URL set."""
    env = dict(env)
    r = spec.resolved_model
    if r is not None:
        p = r.provider
        key = env.get(p.api_key_env) if p.api_key_env in spec.secrets else env.pop(p.api_key_env, None)
        if key is not None:  # `auth: bearer` -> Authorization: Bearer instead of x-api-key
            # keep in step with base.CLAUDE_KEY_ALIASES (what tool calls must not inherit)
            env["ANTHROPIC_AUTH_TOKEN" if p.auth == "bearer" else "ANTHROPIC_API_KEY"] = key
        if p.base_url:
            env["ANTHROPIC_BASE_URL"] = p.base_url
        if p.timeout_s is not None:
            env["API_TIMEOUT_MS"] = str(int(p.timeout_s * 1000))
        if p.max_retries is not None:
            env["CLAUDE_CODE_MAX_RETRIES"] = str(p.max_retries)
    if hasattr(os, "geteuid") and os.geteuid() == 0:  # jobs run as root; bypass needs this
        env.setdefault("IS_SANDBOX", "1")
    return minimal_env(os.environ, env, keep=("CLAUDE_CONFIG_DIR",))


def reasoning_options(spec: TypeSpec) -> dict[str, Any]:
    r = spec.effective_reasoning
    if r is None:
        return {}
    if r == "off":
        return {"thinking": {"type": "disabled"}}
    return {"effort": r}


# --------------------------------------------------------------------------- mapping


def _model_usage(mu: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        input=int(mu.get("inputTokens") or 0),
        output=int(mu.get("outputTokens") or 0),
        cache_read=int(mu.get("cacheReadInputTokens") or 0),
        cache_write=int(mu.get("cacheCreationInputTokens") or 0),
        reasoning=int(mu.get("thinkingTokens") or 0),
    )


def _call_usage(u: dict[str, Any] | None) -> TokenUsage:
    u = u or {}
    return TokenUsage(
        input=int(u.get("input_tokens") or 0),
        output=int(u.get("output_tokens") or 0),
        cache_read=int(u.get("cache_read_input_tokens") or 0),
        cache_write=int(u.get("cache_creation_input_tokens") or 0),
    )


_STOP = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "toolUse", "max_tokens": "length"}


def _raw(msg: Any) -> Any:
    return asdict(msg) if is_dataclass(msg) and not isinstance(msg, type) else msg


def _now_ms() -> int:
    return int(time.time() * 1000)


def user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text, "timestamp": _now_ms()}


class ClaudeMapper:
    """Turns SDK messages into AdapterEvents and tracks turn boundaries."""

    def __init__(self, model: str | None = None) -> None:
        self.in_turn = False
        self.model = model

    def on_send(self, text: str) -> list[AdapterEvent]:
        out = []
        if not self.in_turn:
            self.in_turn = True
            out.append(AdapterEvent("turn_started"))
        out.append(AdapterEvent("message", message=user_message(text)))
        return out

    def on_message(self, msg: Any) -> list[AdapterEvent]:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        out: list[AdapterEvent] = []
        if isinstance(msg, (AssistantMessage, UserMessage)) and not self.in_turn:
            self.in_turn = True
            out.append(AdapterEvent("turn_started", data={"queued": True}))
        if isinstance(msg, SystemMessage):
            if msg.subtype == "init":
                self.model = msg.data.get("model") or self.model
            return out
        if isinstance(msg, AssistantMessage):
            self.model = msg.model or self.model
            blocks: list[dict[str, Any]] = []
            for b in msg.content:
                if isinstance(b, TextBlock) and b.text:
                    blocks.append({"type": "text", "text": b.text})
                elif isinstance(b, ThinkingBlock):
                    blocks.append({"type": "thinking", "thinking": b.thinking})
                elif isinstance(b, ToolUseBlock):
                    blocks.append(
                        {"type": "toolCall", "id": b.id, "name": b.name, "arguments": b.input}
                    )
            if not blocks:
                return out
            has_calls = any(b["type"] == "toolCall" for b in blocks)
            m = {
                "role": "assistant",
                "content": blocks,
                "api": "anthropic-messages",
                "provider": "anthropic",
                "model": msg.model,
                "usage": pi_usage(_call_usage(msg.usage)),
                "stopReason": _STOP.get(msg.stop_reason or "", "toolUse" if has_calls else "stop"),
                "timestamp": _now_ms(),
            }
            out.append(AdapterEvent("message", message=m, model=msg.model, raw=_raw(msg)))
        elif isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                out.append(AdapterEvent("message", message=user_message(msg.content), raw=_raw(msg)))
                return out
            for b in msg.content:
                if isinstance(b, ToolResultBlock):
                    content = b.content
                    if isinstance(content, list):
                        text = "\n".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
                    else:
                        text = content or ""
                    m = {
                        "role": "toolResult",
                        "toolCallId": b.tool_use_id,
                        "toolName": "",
                        "content": [{"type": "text", "text": text}],
                        "isError": bool(b.is_error),
                        "timestamp": _now_ms(),
                    }
                    out.append(AdapterEvent("message", message=m, raw=_raw(msg)))
                elif isinstance(b, TextBlock):
                    out.append(AdapterEvent("message", message=user_message(b.text), raw=_raw(msg)))
        elif isinstance(msg, ResultMessage):
            for model, mu in (msg.model_usage or {}).items():
                out.append(
                    AdapterEvent(
                        "usage", model=model, usage=_model_usage(dict(mu)), cumulative=True,
                        cost_usd=msg.total_cost_usd,
                    )
                )
            if not msg.model_usage and msg.usage:  # older CLIs: per-result usage only
                out.append(AdapterEvent("usage", model=self.model, usage=_call_usage(msg.usage),
                                        cost_usd=msg.total_cost_usd))
            status = "error" if msg.is_error else "completed"
            out.append(
                AdapterEvent(
                    "turn_ended", status=status, raw=_raw(msg),
                    data={"subtype": msg.subtype, "result": msg.result, "num_turns": msg.num_turns},
                )
            )
            self.in_turn = False
        return out


# --------------------------------------------------------------------------- adapter


def _bundled_cli() -> str | None:
    """The SDK's bundled CLI, so the env wrapper also covers images without `claude` on PATH."""
    try:
        import claude_agent_sdk
    except ImportError:
        return None
    for name in ("claude", "claude.exe"):
        p = Path(claude_agent_sdk.__file__).parent / "_bundled" / name
        if p.is_file():
            return str(p)
    return None


class ClaudeAdapter:
    harness = "claude"

    def __init__(
        self,
        workdir: Path,
        *,
        home: Path | None = None,
        cli_path: str | None = None,
        client_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self.workdir = Path(workdir)
        self.home = home or claude_home()
        self.cli_path = cli_path or os.environ.get("FLOCK_CLAUDE_CLI") or shutil.which("claude") \
            or _bundled_cli()
        self._client_factory = client_factory
        self._client: Any = None
        self._reader: asyncio.Task | None = None
        self._queue = EventQueue()
        self._mapper = ClaudeMapper()
        self._stderr: list[str] = []

    def _options(self, spec: TypeSpec, session_id: str, resume: bool, env: dict[str, str]) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        kw: dict[str, Any] = {"resume": session_id} if resume else {"session_id": session_id}
        kw.update(reasoning_options(spec))
        final = claude_env(spec, env)
        drop = sorted(k for k in os.environ if k not in final)
        return ClaudeAgentOptions(
            cwd=str(self.workdir),
            model=spec.harness_model,
            permission_mode="bypassPermissions",
            max_turns=spec.max_turns,
            max_budget_usd=spec.max_budget_usd,
            env={**final, DROP_VAR: ",".join(drop)},
            cli_path=env_wrapper(self.cli_path) if self.cli_path else None,
            stderr=self._on_stderr,
            **kw,
        )

    def _on_stderr(self, line: str) -> None:
        self._stderr = (self._stderr + [line])[-20:]

    async def start(
        self, workdir: Path, session_id: str | None, spec: TypeSpec, *, env: dict[str, str] | None = None
    ) -> str:
        from claude_agent_sdk import ClaudeSDKClient

        self.workdir = Path(workdir)
        resume = session_id is not None
        if resume and not self._transcript(session_id):
            raise ResumeError(f"no transcript for session {session_id} under {self.home}/projects")
        sid = session_id or str(uuid.uuid4())
        self._queue = EventQueue()
        self._mapper = ClaudeMapper(spec.harness_model)
        factory = self._client_factory or ClaudeSDKClient
        self._client = factory(self._options(spec, sid, resume, dict(env or {})))
        try:
            await self._client.connect()
        except Exception as e:
            self._client = None
            detail = f"{e}; stderr: {' | '.join(self._stderr[-3:])}"
            if resume:
                raise ResumeError(detail) from e
            raise RuntimeError(detail) from e
        self._reader = asyncio.create_task(self._read())
        return sid

    async def _read(self) -> None:
        try:
            async for msg in self._client.receive_messages():
                for ev in self._mapper.on_message(msg):
                    self._queue.put(ev)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            self._queue.put(AdapterEvent("error", error=str(e)))
        finally:
            self._queue.put(AdapterEvent("exited"))

    async def send(self, text: str) -> None:
        if self._client is None:
            raise RuntimeError("claude process is not running")
        for ev in self._mapper.on_send(text):
            self._queue.put(ev)
        await self._client.query(text)

    async def stop(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await asyncio.wait_for(client.disconnect(), 15)
            except Exception as e:  # noqa: BLE001
                log.warning("claude disconnect: %s", e)
        if self._reader is not None:
            if not self._reader.done():
                self._reader.cancel()
            with contextlib.suppress(BaseException):
                await self._reader
            self._reader = None

    async def events(self) -> AsyncIterator[AdapterEvent]:
        async for ev in self._queue:
            yield ev

    def _transcript(self, session_id: str) -> Path | None:
        d = project_dir(self.home, self.workdir)
        if d and (d / f"{session_id}.jsonl").is_file():
            return d / f"{session_id}.jsonl"
        hits = list((self.home / "projects").glob(f"*/{glob_escape(session_id)}.jsonl"))
        return hits[0] if hits else None

    def session_files(self, session_id: str | None) -> list[Path]:
        """projects/<encoded workdir>/<session_id>.jsonl plus everything under
        projects/<encoded workdir>/<session_id>/ (subagents, tool-results) if present.
        That is the verified resume set (experiments/resume-gate/RESULTS.md)."""
        if not session_id:
            return []
        t = self._transcript(session_id)
        if t is None:
            return []
        extra = t.parent / session_id
        return [t] + (sorted(p for p in extra.rglob("*") if p.is_file()) if extra.is_dir() else [])
