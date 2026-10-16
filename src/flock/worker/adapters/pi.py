"""Pi (`@earendil-works/pi-coding-agent`) through `pi --mode rpc`: JSONL commands on stdin,
responses (`type: response`, `id` echoed) and session events on stdout, diagnostics on stderr.
One process per burst, cwd = the agent's workdir.

Observed with 1.2.0 (tests/fixtures/pi_rpc.jsonl, recorded against tests/fake_openai.py):
- `--session-dir D --session-id X` writes `D/<iso-ts>_X.jsonl` (pi session format v3, header
  `cwd` = the process cwd). The file appears with the first message, not at startup.
  `--session-id` is scoped to the cwd: started from another cwd, Pi warns on stderr and
  silently creates a second file with the same id, so `start` checks the file first.
- `prompt` with `streamingBehavior: "steer"` answers `disposition: started` when idle and
  `queued` when busy; queued text is delivered after the running tool calls, inside the same
  run. With `set_steering_mode: "all"` several queued prompts go out in one model call.
- One run: `agent_start`, then per model call `turn_start`/`message_*`/`tool_execution_*`/
  `turn_end`, then `agent_end` and `agent_settled {aborted}`. `agent_settled` means Pi will not
  continue on its own; it is the end of a flock turn.
- Usage is per model call on each assistant `message_end` (`input` excludes cached tokens;
  `reasoning` is already inside `output`). Nothing is cumulative except `get_session_stats`.
- A model error ends the run with an assistant message `stopReason: "error"` + `errorMessage`.
- Closing stdin exits at once, even mid-tool, and Pi kills the tool's process group
  (backgrounded grandchildren included). SIGKILL does not: the tool's bash runs in its own
  session and survives, so `stop` kills the descendants it saw before killing Pi.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flock.models import TokenUsage, TypeSpec
from flock.worker.adapters.base import AdapterEvent, EventQueue, ResumeError

log = logging.getLogger(__name__)

# Inherited from the worker's environment; nothing else is (FLOCK_TOKEN in particular is not).
ENV_ALLOW = (
    "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ", "TMPDIR",
    "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
)
ENV_DENY = ("FLOCK_TOKEN",)
# Network extras off; the image sets the same values.
PI_ENV_DEFAULTS = {"PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"}
PI_FLAGS = ("--no-extensions", "--no-mcp", "--no-skills", "--no-prompt-templates", "--no-approve")
RPC_TIMEOUT = 60.0
STOP_GRACE = 5.0


def pi_home() -> Path:
    return Path(os.environ.get("PI_CODING_AGENT_DIR") or Path.home() / ".pi" / "agent")


@dataclass(frozen=True)
class PiModelConfig:
    """One OpenAI-compatible endpoint for Pi's models.json. Mirrors what PR A's
    `resolve_model()` returns (provider_name, provider.base_url, provider.api_key_env, model_id,
    reasoning, extra). `api_key_env` names the env var holding the key; the key never reaches
    disk. `reasoning` is a thinking level (off|minimal|low|medium|high|xhigh|max) or None.
    `extra` is merged into the model entry (contextWindow, maxTokens, cost, ...)."""

    base_url: str
    api_key_env: str
    model_id: str
    reasoning: str | None = None
    provider: str = "flock"
    api: str = "openai-completions"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def model_ref(self) -> str:
        return f"{self.provider}/{self.model_id}"

    def models_json(self) -> dict[str, Any]:
        model = {"id": self.model_id, "reasoning": self.reasoning not in (None, "off"), **self.extra}
        return {"providers": {self.provider: {"baseUrl": self.base_url, "api": self.api,
                                              "apiKey": f"${self.api_key_env}", "models": [model]}}}


def write_models_json(home: Path, cfg: PiModelConfig) -> Path:
    """Atomically (re)write `<home>/models.json`. Every agent on a worker has the same type, so
    concurrent writers write the same bytes."""
    home.mkdir(parents=True, exist_ok=True)
    path = home / "models.json"
    tmp = home / f".models.json.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    tmp.write_text(json.dumps(cfg.models_json(), indent=2) + "\n")
    os.replace(tmp, path)
    return path


def session_header(path: Path) -> dict[str, Any]:
    try:
        with open(path) as f:
            h = json.loads(f.readline() or "{}")
    except (OSError, ValueError):
        return {}
    return h if isinstance(h, dict) and h.get("type") == "session" else {}


def _same_dir(a: str | Path | None, b: str | Path) -> bool:
    return a is not None and os.path.realpath(a) == os.path.realpath(b)


def harness_env(spec: TypeSpec, env: dict[str, str] | None, home: Path,
                model: PiModelConfig | None, base: dict[str, str] | None = None) -> dict[str, str]:
    """The allow-listed environment for the Pi process: a few process basics, Pi's own vars,
    the type's secrets and the API key variable (by name, from the worker env), then `env`
    (the type's `env` plus the msgboard identity, from the caller). Never FLOCK_TOKEN."""
    base = dict(os.environ if base is None else base)
    out = {k: base[k] for k in ENV_ALLOW if k in base}
    out.update({k: base.get(k, v) for k, v in PI_ENV_DEFAULTS.items()})
    names = [*spec.secrets, *([model.api_key_env] if model else [])]
    out.update({k: base[k] for k in names if k in base})
    if model and model.api_key_env not in out and model.api_key_env not in (env or {}):
        log.warning("pi: %s is not set; Pi refuses to prompt without a key (set it to any value "
                    "for a keyless endpoint)", model.api_key_env)
    out.update(env or {})
    for k in ENV_DENY:
        out.pop(k, None)
    out["PI_CODING_AGENT_DIR"] = str(home)
    return out


def _usage(u: dict[str, Any] | None) -> TokenUsage:
    u = u or {}
    return TokenUsage(input=int(u.get("input") or 0), output=int(u.get("output") or 0),
                      cache_read=int(u.get("cacheRead") or 0), cache_write=int(u.get("cacheWrite") or 0),
                      reasoning=int(u.get("reasoning") or 0))


class PiMapper:
    """Pi RPC records -> AdapterEvents. A flock turn runs from the prompt that started a run
    (or `agent_start`) to `agent_settled`; Pi may run several model calls, retries, compactions
    and queued steers inside it."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model
        self.in_turn = False
        self.last_assistant: dict[str, Any] | None = None

    def _open(self, data: dict[str, Any]) -> list[AdapterEvent]:
        if self.in_turn:
            return []
        self.in_turn = True
        self.last_assistant = None
        return [AdapterEvent("turn_started", data=data)]

    def on_prompt_response(self, resp: dict[str, Any]) -> list[AdapterEvent]:
        if resp.get("success") and (resp.get("data") or {}).get("disposition") == "started":
            return self._open({})
        return []

    def on_record(self, rec: dict[str, Any]) -> list[AdapterEvent]:
        t = rec.get("type")
        if t == "agent_start":
            return self._open({})
        if t == "message_end":
            msg = rec.get("message") or {}
            out = [AdapterEvent("message", message=msg, raw=rec)]
            if msg.get("role") == "assistant":
                self.last_assistant = msg
                model = f"{msg['provider']}/{msg['model']}" if msg.get("provider") and msg.get("model") \
                    else self.model
                u = _usage(msg.get("usage"))
                if u != TokenUsage():
                    out.append(AdapterEvent("usage", model=model, usage=u, raw=rec))
            return out
        if t == "compaction_end":
            u = ((rec.get("result") or {}).get("usage")) or None
            if u:
                return [AdapterEvent("usage", model=self.model, usage=_usage(u),
                                     data={"compaction": True}, raw=rec)]
            return []
        if t == "agent_settled":
            if not self.in_turn:
                return []
            self.in_turn = False
            last = self.last_assistant or {}
            if rec.get("aborted"):
                return [AdapterEvent("turn_ended", status="interrupted", raw=rec)]
            if last.get("stopReason") == "error":
                return [AdapterEvent("turn_ended", status="error", error=last.get("errorMessage"), raw=rec)]
            return [AdapterEvent("turn_ended", status="completed", raw=rec)]
        if t == "auto_retry_end" and not rec.get("success", True):
            return [AdapterEvent("error", error=str(rec.get("finalError") or rec.get("error") or
                                                    "auto retry failed"), raw=rec)]
        if t == "extension_error":
            return [AdapterEvent("error", error=str(rec.get("error") or rec), raw=rec)]
        if t == "response" and not rec.get("success", True):
            return [AdapterEvent("error", error=f"{rec.get('command')}: {rec.get('error')}",
                                 data={"command": rec.get("command")}, raw=rec)]
        return []


class RpcError(RuntimeError):
    pass


def _descendants(pid: int) -> list[tuple[int, str]]:
    """(pid, start time) of every live descendant of `pid`, from /proc (Linux; [] elsewhere)."""
    children: dict[int, list[int]] = {}
    starts: dict[int, str] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for d in entries:
        if not d.isdigit():
            continue
        try:
            stat = Path(f"/proc/{d}/stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rfind(")") + 2:].split()
        children.setdefault(int(fields[1]), []).append(int(d))
        starts[int(d)] = fields[19]
    out, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            out.append((c, starts[c]))
            todo.append(c)
    return out


def _kill_all(procs: list[tuple[int, str]]) -> int:
    """SIGKILL each (pid, start time) that is still the same process. Returns the count."""
    n = 0
    for pid, start in procs:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            if stat[stat.rfind(")") + 2:].split()[19] != start:
                continue
            os.kill(pid, signal.SIGKILL)
            n += 1
        except (OSError, IndexError):
            continue
    return n


class PiAdapter:
    harness = "pi"

    def __init__(self, workdir: Path, *, agent: str | None = None, home: Path | None = None,
                 pi_bin: str | None = None, model_config: PiModelConfig | None = None) -> None:
        self.workdir = Path(workdir)
        self.agent = agent or self.workdir.name
        self.home = Path(home) if home else pi_home()
        self.pi_bin = pi_bin or os.environ.get("FLOCK_PI_CLI") or "pi"
        self.model_config = model_config
        self.session_id: str | None = None
        self.session_file: Path | None = None
        self.env: dict[str, str] = {}
        self.argv: list[str] = []
        self._proc: asyncio.subprocess.Process | None = None
        self._queue = EventQueue()
        self._mapper = PiMapper()
        self._pending: dict[str, asyncio.Future] = {}
        self._next_id = 0
        self._tasks: list[asyncio.Task] = []
        self._stderr: list[str] = []
        self._seen: list[tuple[int, str]] = []

    @property
    def session_dir(self) -> Path:
        return self.home / "sessions" / self.agent

    # ------------------------------------------------------------------ files
    def find_session(self, session_id: str) -> Path | None:
        """`<session_dir>/<ts>_<id>.jsonl` whose header cwd is this workdir (newest if several)."""
        for p in sorted(self.session_dir.glob(f"*_{session_id}.jsonl"), reverse=True):
            if _same_dir(session_header(p).get("cwd"), self.workdir):
                return p
        return None

    def session_files(self, session_id: str | None) -> list[Path]:
        """The session file, `<home>/sessions/<agent>/<ts>_<id>.jsonl`, and nothing else: a fresh
        agent dir holding only models.json (rewritten on every start) and this file resumes with
        the full history. Any same-id file from another cwd is included so a restart can point at
        it; with the fixed workdir there is exactly one."""
        sid = session_id or self.session_id
        if not sid:
            return []
        return sorted(p for p in self.session_dir.glob(f"*_{sid}.jsonl") if p.is_file())

    # ------------------------------------------------------------------ rpc
    async def _write(self, cmd: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.returncode is not None:
            raise RpcError("pi is not running")
        self._proc.stdin.write((json.dumps(cmd) + "\n").encode())
        await self._proc.stdin.drain()

    async def request(self, cmd: dict[str, Any], timeout: float = RPC_TIMEOUT) -> dict[str, Any]:
        self._next_id += 1
        rid = f"flock-{self._next_id}"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            await self._write({**cmd, "id": rid})
            resp = await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(rid, None)
        if not resp.get("success"):
            raise RpcError(f"{cmd['type']}: {resp.get('error')}")
        return resp

    async def _read(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            while line := await self._proc.stdout.readline():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    self._dispatch(rec)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            self._queue.put(AdapterEvent("error", error=f"pi reader: {e}"))
        finally:
            for f in self._pending.values():
                if not f.done():
                    f.set_result({"success": False, "error": "pi exited"})
            code = None
            if self._proc is not None:
                with contextlib.suppress(TimeoutError, ProcessLookupError):
                    code = await asyncio.wait_for(self._proc.wait(), STOP_GRACE)
            self._queue.put(AdapterEvent("exited", data={"returncode": code, "stderr": self._stderr[-5:]}))

    def _dispatch(self, rec: dict[str, Any]) -> None:
        if rec.get("type") == "response":
            fut = self._pending.get(str(rec.get("id")))
            if fut is not None:
                if rec.get("command") == "prompt":
                    for ev in self._mapper.on_prompt_response(rec):
                        self._queue.put(ev)
                if not fut.done():
                    fut.set_result(rec)
                return  # our own request: the caller handles failures
        for ev in self._mapper.on_record(rec):
            self._queue.put(ev)

    async def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        while line := await self._proc.stderr.readline():
            self._stderr = (self._stderr + [line.decode(errors="replace").rstrip()])[-20:]

    # ------------------------------------------------------------------ adapter
    async def start(self, workdir: Path, session_id: str | None, spec: TypeSpec, *,
                    env: dict[str, str] | None = None) -> str:
        self.workdir = Path(workdir)
        expected: Path | None = None
        if session_id:
            expected = self.find_session(session_id)
            if expected is None:
                others = self.session_files(session_id)
                why = (f"its file {others[-1].name} was written from cwd "
                       f"{session_header(others[-1]).get('cwd')!r}") if others else "no session file"
                raise ResumeError(f"cannot resume pi session {session_id} from {self.workdir}: {why} "
                                  f"under {self.session_dir}")
        sid = session_id or str(uuid.uuid4())
        cfg = self.model_config
        if cfg is not None:
            write_models_json(self.home, cfg)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.argv = [self.pi_bin, "--mode", "rpc", "--session-dir", str(self.session_dir),
                     "--session-id", sid, "--model", cfg.model_ref if cfg else spec.model, *PI_FLAGS]
        if cfg is not None and cfg.reasoning:
            self.argv += ["--thinking", cfg.reasoning]
        self.env = harness_env(spec, env, self.home, cfg)
        self._queue = EventQueue()
        self._mapper = PiMapper(cfg.model_ref if cfg else spec.model)
        self._stderr = []
        self._proc = await asyncio.create_subprocess_exec(
            *self.argv, cwd=str(self.workdir), env=self.env, limit=64 * 1024 * 1024,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self._tasks = [asyncio.create_task(self._read()), asyncio.create_task(self._drain_stderr())]
        try:
            await self.request({"type": "set_steering_mode", "mode": "all"})
            state = (await self.request({"type": "get_state"})).get("data") or {}
        except asyncio.CancelledError:
            await self.stop()
            raise
        except Exception as e:
            detail = f"{e}; stderr: {' | '.join(self._stderr[-3:])}"
            await self.stop()
            raise (ResumeError if session_id else RuntimeError)(detail) from e
        got = state.get("sessionFile")
        if expected is not None and (not got or not _same_dir(got, expected)):
            await self.stop()
            raise ResumeError(f"pi opened {got} instead of {expected}")
        self.session_id = state.get("sessionId") or sid
        self.session_file = Path(got) if got else None
        return self.session_id

    async def send(self, text: str) -> None:
        """`prompt` with steer: starts a run when idle, is queued into the running one when busy.
        A prompt Pi rejects (e.g. "No API key found") is reported as a failed turn rather than
        raised: raising makes the runtime re-buffer it and restart the burst forever. Raises only
        when the process is gone."""
        try:
            await self.request({"type": "prompt", "message": text, "streamingBehavior": "steer"})
        except RpcError as e:
            if self._proc is None or self._proc.returncode is not None or "pi exited" in str(e):
                raise
            if self._mapper.in_turn:
                self._queue.put(AdapterEvent("error", error=str(e)))
                return
            for ev in (AdapterEvent("turn_started", data={"rejected": True}),
                       AdapterEvent("turn_ended", status="error", error=str(e))):
                self._queue.put(ev)

    async def stop(self) -> None:
        """abort (if a run is in flight), close stdin, wait; then SIGKILL Pi and every descendant
        seen just before (tool processes run in their own session and would outlive Pi)."""
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            self._seen = _descendants(proc.pid)
            if self._mapper.in_turn:
                self._proc = proc
                with contextlib.suppress(Exception):
                    await self.request({"type": "abort"}, timeout=STOP_GRACE)
                self._proc = None
            try:
                if proc.stdin:
                    proc.stdin.close()
                await asyncio.wait_for(proc.wait(), STOP_GRACE)
            except (TimeoutError, ProcessLookupError, ConnectionError):
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                    await proc.wait()
            if n := _kill_all(self._seen):
                log.warning("%s: killed %d leftover tool process(es)", self.agent, n)
        for t in self._tasks:
            try:
                await asyncio.wait_for(t, STOP_GRACE)
            except (TimeoutError, asyncio.CancelledError):
                t.cancel()
        self._tasks = []

    async def events(self) -> AsyncIterator[AdapterEvent]:
        async for ev in self._queue:
            yield ev
