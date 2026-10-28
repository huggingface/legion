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
- `max_turns` / `max_budget_usd` (per flock turn, which may fold several steered prompts) are
  enforced here: the mapper counts model calls (`turn_start`) and prices each assistant
  `message_end` from the flock price table; when Pi would make another call past a limit
  (an assistant message that asks for tools, or a `turn_start` for a steered prompt) the
  adapter sends `abort`, and the turn ends `error` with `data.reason` max_turns|max_budget_usd.
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
import re
import shutil
import signal
import tempfile
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flock.controller.prices import Price, price_for
from flock.home import home
from flock.isolation import command as isolate_command
from flock.isolation import enabled as isolation_enabled
from flock.models import TokenUsage, TypeSpec
from flock.worker.adapters.base import (
    BASE_ENV,
    BASE_ENV_PREFIXES,
    AdapterEvent,
    EventQueue,
    ResumeError,
    minimal_env,
)

log = logging.getLogger(__name__)

# Never reaches Pi, even when listed in `secrets` or `env` (the worker's controller credential).
ENV_DENY = ("FLOCK_TOKEN",)
# Network extras off; the image sets the same values.
PI_ENV_DEFAULTS = {"PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"}
PI_FLAGS = ("--no-extensions", "--no-mcp", "--no-skills", "--no-prompt-templates", "--no-approve")
HTTP_TRANSIENT = re.compile(r"(?:^\s*|\bhttp\s+|\bstatus(?: code)?\s*[:=]?\s*)(?:429|502|503)\b", re.IGNORECASE)
RPC_TIMEOUT = 60.0
# Records PiMapper ignores that carry whole messages: each `message_update` repeats the partial
# message so far (per token), `tool_execution_update` the output so far, `turn_end` and
# `agent_end` the turn's / run's messages again. The reader skips them unparsed (Pi writes
# `type` first), which spares the worker most of the JSON it would decode and drop.
SKIPPED_RECORDS = ("message_update", "tool_execution_update", "turn_end", "agent_end")
_SKIP_PREFIXES = tuple(f'{{"type":"{t}"'.encode() for t in SKIPPED_RECORDS)
STOP_GRACE = 5.0


def pi_home() -> Path:
    return Path(os.environ.get("PI_CODING_AGENT_DIR") or home() / ".pi" / "agent")


@dataclass(frozen=True)
class PiModelConfig:
    """One OpenAI-compatible endpoint for Pi's models.json, from a type's `ResolvedModel`
    (`pi_model_config`). `provider` is the flock provider name, so Pi's model ref and usage key
    are `<provider>/<model id>`. `api_key_env` names the env var holding the key; the key never
    reaches disk. `reasoning` is a thinking level (off|minimal|low|medium|high|xhigh|max) or
    None. `extra` is merged into the model entry (contextWindow, maxTokens, cost, ...)."""

    base_url: str
    api_key_env: str
    model_id: str
    provider: str
    reasoning: str | None = None
    api: str = "openai-completions"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def model_ref(self) -> str:
        return f"{self.provider}/{self.model_id}"

    def models_json(self) -> dict[str, Any]:
        model = {"id": self.model_id, "reasoning": self.reasoning not in (None, "off"), **self.extra}
        return {"providers": {self.provider: {"baseUrl": self.base_url, "api": self.api,
                                              "apiKey": f"${self.api_key_env}", "models": [model]}}}


def pi_model_config(spec: TypeSpec, agent: str = "") -> PiModelConfig | None:
    """Pi's endpoint for a provider-backed type (`spec.resolved_model`). None: a harness-native
    model, passed to Pi as given (one of Pi's built-in providers, e.g. `anthropic/claude-...`)."""
    r = spec.resolved_model
    if r is None:
        return None
    from flock.worker.adapters.llm import pick_endpoint

    endpoint = pick_endpoint(agent, r.provider.endpoints) if agent else r.provider.base_url
    return PiModelConfig(base_url=endpoint or "", api_key_env=r.provider.api_key_env,
                         model_id=r.model_id, provider=r.provider_name,
                         reasoning=spec.effective_reasoning)


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


def harness_env(env: Mapping[str, str] | None, home: Path,
                base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The Pi process environment: `minimal_env` of the worker's (allow-list plus Pi's own
    variables) with `env` on top (what `AgentRuntime.harness_env` selected: the type's secrets,
    `env`, the msgboard identity, the provider key), Pi's network-off defaults where unset, its
    agent dir, and never FLOCK_TOKEN."""
    base = os.environ if base is None else base
    out = minimal_env(base, env or {}, keep=PI_ENV_DEFAULTS)
    for k, v in PI_ENV_DEFAULTS.items():
        out.setdefault(k, v)
    for k in ENV_DENY:
        out.pop(k, None)
    out["PI_CODING_AGENT_DIR"] = str(home)
    return out


# Git's env-based config (`git -c` for every git under Pi, e.g. a type's pack memory caps in
# spec.env). Only the count and numbered key/value pairs: GIT_CONFIG_PARAMETERS and the rest stay out.
GIT_CONFIG_ENV = re.compile(r"GIT_CONFIG_COUNT|GIT_CONFIG_(?:KEY|VALUE)_\d+")


ISOLATED_DENY = frozenset({"HF_TOKEN", "FLOCK_TOKEN", "FLOCK_WORLD_TOKEN"})
DROP_VAR = "FLOCK_HARNESS_ENV_DROP"  # comma-separated names a deployment also keeps from Pi


def isolated_env(env: dict[str, str], provider_key: str | None) -> dict[str, str]:
    """Isolation adds only the provider key, agent-scoped tool/controller routing and git's
    GIT_CONFIG_COUNT / GIT_CONFIG_KEY_n / GIT_CONFIG_VALUE_n; never ISOLATED_DENY or the
    names in the worker's FLOCK_HARNESS_ENV_DROP."""
    allowed = BASE_ENV | set(PI_ENV_DEFAULTS) | {"PI_CODING_AGENT_DIR", "FLOCK_TOOLS_PIN",
              "FLOCK_URL", "FLOCK_AGENT", "FLOCK_PARENT", "FLOCK_AGENT_TOKEN", provider_key}
    deny = ISOLATED_DENY | set(filter(None, os.environ.get(DROP_VAR, "").split(",")))
    return {k: v for k, v in env.items()
            if (k in allowed or k.startswith(BASE_ENV_PREFIXES) or GIT_CONFIG_ENV.fullmatch(k))
            and k not in deny}


def _usage(u: dict[str, Any] | None) -> TokenUsage:
    u = u or {}
    return TokenUsage(input=int(u.get("input") or 0), output=int(u.get("output") or 0),
                      cache_read=int(u.get("cacheRead") or 0), cache_write=int(u.get("cacheWrite") or 0),
                      reasoning=int(u.get("reasoning") or 0))


class PiMapper:
    """Pi RPC records -> AdapterEvents. A flock turn runs from the prompt that started a run
    (or `agent_start`) to `agent_settled`; Pi may run several model calls, retries, compactions
    and queued steers inside it."""

    def __init__(self, model: str | None = None, *, max_turns: int | None = None,
                 max_budget_usd: float | None = None, prices: dict[str, Price] | None = None) -> None:
        self.model = model
        self.max_turns = max_turns
        self.max_budget_usd = max_budget_usd
        self.prices = prices
        self.in_turn = False
        self.last_assistant: dict[str, Any] | None = None
        self.calls = 0  # model calls in this flock turn
        self.cost = 0.0  # flock-priced cost of this flock turn
        self.limit: str | None = None  # the limit that made us abort this turn
        self.abort_requested = False  # set when `limit` is first hit; the adapter sends `abort`

    def _open(self, data: dict[str, Any]) -> list[AdapterEvent]:
        if self.in_turn:
            return []
        self.in_turn = True
        self.last_assistant = None
        self.calls, self.cost, self.limit, self.abort_requested = 0, 0.0, None, False
        return [AdapterEvent("turn_started", data=data)]

    def _spend(self, model: str | None, u: TokenUsage) -> None:
        if self.max_budget_usd is not None:
            p = price_for(model or "", self.prices)
            self.cost += (u.input * p.input + u.output * p.output + u.cache_read * p.cache_read
                          + u.cache_write * p.cache_write) / 1_000_000

    def _exceeded(self) -> str | None:
        """The limit another model call would break, if any."""
        if self.max_turns is not None and self.calls >= self.max_turns:
            return "max_turns"
        if self.max_budget_usd is not None and self.cost > self.max_budget_usd:
            return "max_budget_usd"
        return None

    def _check(self, *, next_call: bool) -> None:
        if self.limit is None and next_call and (hit := self._exceeded()):
            self.limit, self.abort_requested = hit, True

    def on_prompt_response(self, resp: dict[str, Any]) -> list[AdapterEvent]:
        if resp.get("success") and (resp.get("data") or {}).get("disposition") == "started":
            return self._open({})
        return []

    def on_record(self, rec: dict[str, Any]) -> list[AdapterEvent]:
        t = rec.get("type")
        if t == "agent_start":
            return self._open({})
        if t == "turn_start":
            # a call already past the limit (a steered prompt after a final answer): stop it
            self._check(next_call=True)
            if self.limit is None:  # after an abort Pi still opens a call it never sends
                self.calls += 1
            return []
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
                    self._spend(model, u)
                # toolUse: Pi runs the tools and calls the model again
                self._check(next_call=msg.get("stopReason") == "toolUse")
            return out
        if t == "compaction_end":
            u = ((rec.get("result") or {}).get("usage")) or None
            if u:
                self._spend(self.model, _usage(u))
                return [AdapterEvent("usage", model=self.model, usage=_usage(u),
                                     data={"compaction": True}, raw=rec)]
            return []
        if t == "agent_settled":
            if not self.in_turn:
                return []
            self.in_turn = False
            last = self.last_assistant or {}
            if self.limit:
                limit = getattr(self, self.limit)
                return [AdapterEvent("turn_ended", status="error", raw=rec,
                                     error=f"{self.limit} exceeded ({limit}): {self.calls} model call(s), "
                                           f"${self.cost:.6f}; aborted",
                                     data={"reason": self.limit, "subtype": f"error_{self.limit}",
                                           "num_turns": self.calls})]
            if rec.get("aborted"):
                return [AdapterEvent("turn_ended", status="interrupted", raw=rec)]
            if last.get("stopReason") == "error":
                message = str(last.get("errorMessage") or "")
                transient = bool(HTTP_TRANSIENT.search(message)) or any(x in message.lower() for x in (
                    "timeout", "timed out", "fetch failed", "econnreset", "connection reset",
                    "connection refused", "socket", "abort"))
                return [AdapterEvent("turn_ended", status="error", error=message, raw=rec,
                                     data={"reason": "model_transport"} if transient else {})]
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
                 pi_bin: str | None = None, model_config: PiModelConfig | None = None,
                 prices: dict[str, Price] | None = None) -> None:
        self.workdir = Path(workdir)
        self.agent = agent or self.workdir.name
        self.home = Path(home) if home else (pi_home() / "agents" / self.agent if isolation_enabled() else pi_home())
        self.pi_bin = pi_bin or os.environ.get("FLOCK_PI_CLI") or "pi"
        self.model_config = model_config
        self.prices = prices  # for max_budget_usd; None: the global table
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
        self._isolation_dir: tempfile.TemporaryDirectory | None = None
        self._seen: list[tuple[int, str]] = []

    def update_spec(self, spec: TypeSpec) -> None:
        """A live `flock update` (docs/live.md): a new endpoint list (vLLM base_urls), model id
        or reasoning level takes effect at Pi's next start (a transport error or timeout
        restarts it), where models.json is rewritten. A change of the provider's key variable
        waits for a worker restart, since the key is set up with the process environment."""
        if self.model_config is None:
            return
        new = pi_model_config(spec, self.agent)
        if new is None or new.api_key_env != self.model_config.api_key_env or new == self.model_config:
            return
        log.info("%s: pi model config updated (endpoint %s); applies at the next pi start",
                 self.agent, new.base_url)
        self.model_config = new

    def scratch_env(self) -> dict[str, str]:
        """Isolated Pi's HOME, TMPDIR and tool caches: under its own agent dir (disk-backed,
        private to this agent, not synced), not in the in-memory root where npm, uv or test
        caches would fill the tmpfs. TMPDIR starts empty on every start; HOME and its caches
        are kept for the agent's next start on this worker."""
        home, tmp = self.home / "home", self.home / "tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        for d in (home, tmp):
            d.mkdir(parents=True, exist_ok=True)
            d.chmod(0o700)
        cache = home / ".cache"
        return {"HOME": str(home), "TMPDIR": str(tmp), "XDG_CACHE_HOME": str(cache),
                "UV_CACHE_DIR": str(cache / "uv"), "npm_config_cache": str(cache / "npm"),
                "PIP_CACHE_DIR": str(cache / "pip")}

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
                if line.startswith(_SKIP_PREFIXES):
                    continue
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
        if self._mapper.abort_requested:
            self._mapper.abort_requested = False
            log.warning("%s: %s exceeded; aborting the pi run", self.agent, self._mapper.limit)
            self._tasks.append(asyncio.create_task(self._abort()))

    async def _abort(self) -> None:
        try:
            await self.request({"type": "abort"}, timeout=STOP_GRACE)
        except Exception as e:  # noqa: BLE001 - the burst ends anyway if Pi is gone
            self._queue.put(AdapterEvent("error", error=f"abort after limit: {e}"))

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
        self.env = harness_env(env, self.home)
        launch = self.argv
        if isolation_enabled():
            self.env = isolated_env(self.env, cfg.api_key_env if cfg else None)
            # Private directories prevent a sibling from reading this agent's gateway key.
            pin = self.env.get("FLOCK_TOOLS_PIN")
            readonly = []
            if pin:
                pin_path = Path(pin)
                if pin_path.parent.name != self.agent:
                    raise RuntimeError("isolated Pi requires an agent-private tools pin directory")
                pin_path.parent.mkdir(parents=True, exist_ok=True)
                readonly.append(pin_path.parent)
            self.env.update(self.scratch_env())
            self._isolation_dir = tempfile.TemporaryDirectory(prefix="flock-pi-root-")
            # The root is a size-capped tmpfs (RAM); its /tmp is this agent's disk-backed TMPDIR,
            # so files a tool writes to a literal /tmp never accumulate in memory.
            launch = isolate_command(self.argv, Path(self._isolation_dir.name), cwd=self.workdir,
                                     readonly=readonly, writable=[self.workdir, self.home], network=True,
                                     tmp=self.home / "tmp")
        self._queue = EventQueue()
        self._mapper = PiMapper(cfg.model_ref if cfg else spec.model, max_turns=spec.max_turns,
                                max_budget_usd=spec.max_budget_usd, prices=self.prices)
        self._stderr = []
        self._proc = await asyncio.create_subprocess_exec(
            *launch, cwd=str(self.workdir), env=self.env, limit=64 * 1024 * 1024,
            start_new_session=isolation_enabled(),
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
        if self._isolation_dir is not None:
            self._isolation_dir.cleanup()
            self._isolation_dir = None

    async def events(self) -> AsyncIterator[AdapterEvent]:
        async for ev in self._queue:
            yield ev
