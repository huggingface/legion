"""FlockPi: Harbor's Pi as a turn server inside one long-lived Trial.

A HarborAdapter (flock.worker.adapters.harbor) selects it through
AgentConfig(import_path="flock.harbor.agent:FlockPi") and sets `channel` after Trial.create.

run(): put back the agent's saved Pi sessions if the channel says so, signal ready, then one
`pi` turn per prompt: turn 1 is Harbor's instruction plus the prompt, later turns
`--continue`. Prompts queued during a turn are joined into the next one. After each turn the
turn's messages and token usage (from pi.txt) go back as flock events, and the session files
are copied to the channel's `home`, where flock's BucketSync pushes them. While idle it
probes the environment every PROBE seconds; a failed probe ends the Trial with an error.
run() returns when the channel closes, after `trial_seconds` (queued prompts wait for the next
Trial), after MAX_ERRORS failed turns in a row (rate-limited ones do not count), or when the
first turn on restored sessions fails without a model error; the Trial then runs its
verifier. That first failure, or MAX_ERRORS with none working on the restored sessions, sets
`restore_error`: the next start begins a fresh session. The worker's loop engine and
controller do loops, steering, budgets and stops.
"""

import asyncio
import json
import os
import shlex
import shutil
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from harbor.agents.installed.pi import Pi
from harbor.models.agent.context import AgentContext

from flock.models import TokenUsage
from flock.worker.adapters.base import AdapterEvent, pi_tokens, transient_error

from .harnesses.pi import terminal_pi_error
from .results import safe_error

PROBE = 60  # seconds between idle probes (`exec true`); also hf-sandbox activity
MAX_ERRORS = 3  # failed turns in a row before the Trial ends
SYNC = 600  # seconds between session downloads when /logs/agent is not mounted (hf-sandbox)
PULL_WAIT = 15  # seconds pull_sessions() may take
STOP_GRACE = 30  # seconds stop() waits for a killed turn to report
# Ends `pi` and its tool processes without procps (absent from slim images): `pi --print`
# command lines and their descendants (parent pids from /proc/*/stat, read before any kill so
# none is reparented first). `[p]` keeps this shell's own command line from matching; it fails
# when no `pi` matched.
KILL_PI = (
    "k=' '; t=; for d in /proc/[0-9]*; do p=${d#/proc/}; "
    "read -r s 2>/dev/null < $d/stat || continue; s=${s##*) }; set -- $s; t=\"$t $p:$2\"; "
    "case \"$(tr '\\0' ' ' 2>/dev/null < $d/cmdline)\" in *[p]'i --print'*) k=\"$k$p \";; esac; "
    "done; [ \"$k\" != ' ' ] || exit 1; n=0; while [ $n != ${#k} ]; do n=${#k}; "
    "for e in $t; do case \"$k\" in *\" ${e%:*} \"*) ;; *\" ${e#*:} \"*) k=\"$k${e%:*} \";; esac; "
    "done; done; kill $k 2>/dev/null; true"
)


@dataclass
class Channel:
    """Between a HarborAdapter and the FlockPi of its Trial."""

    emit: Callable[[AdapterEvent], None]
    prompts: asyncio.Queue  # one str per prompt; None ends run()
    home: Path  # session files are copied to home/sessions after each turn
    model: str  # the model id flock prices the usage under
    trial_seconds: float  # run() returns once idle after this long
    max_turns: int | None = None  # Pi calls per turn (limits.agent_max_turns)
    restore: bool = False  # upload home/sessions before the first turn and --continue
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    restore_error: str | None = None


def read_turn(path):
    """(messages, usage) of one turn's pi.txt."""
    messages, usage = [], TokenUsage()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return messages, usage
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "compaction_end":  # Pi's summary call, as adapters/pi.py counts it
            usage += pi_tokens((event.get("result") or {}).get("usage"))
            continue
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if isinstance(message, dict):
            messages.append(message)
            if message.get("role") in ("assistant", "toolResult"):  # as Harbor counts them
                usage += pi_tokens(message.get("usage"))
    return messages, usage


class FlockPi(Pi):
    channel: Channel  # set by the HarborAdapter after Trial.create
    why: str | None = None  # why run() returned: closed, trial_hours, errors, probe
    busy = False  # a turn is in flight
    _killed = False
    _env = None

    async def run(self, instruction, environment, context):
        # resume() sets this private flag around run(); that call is one --continue turn.
        if self._resume:
            return await super().run(instruction, environment, context)
        ch, self._env = self.channel, environment
        self.idle = asyncio.Event()
        self.idle.set()
        self._synced = float("-inf")  # unmounted: the first turn's sessions are pulled at once
        totals = AgentContext(n_input_tokens=0, n_output_tokens=0, n_cache_tokens=0)
        deadline = time.monotonic() + ch.trial_seconds
        fresh = not ch.restore  # no Pi session in this environment yet
        if ch.restore:
            try:
                await self._restore(environment)
            except Exception as exc:
                ch.restore_error = safe_error(exc)[:500]
                raise
        ch.ready.set()
        errors = 0
        check = ch.restore  # until a turn runs on the restored sessions
        try:
            while errors < MAX_ERRORS:
                prompt = await self._next_prompt(environment, deadline)
                if prompt is None:
                    return
                text = f"{instruction}\n\n{prompt}" if fresh else prompt
                ended, used, model_error = await self._turn(text, fresh, environment, totals)
                transient = ended.data.get("reason") == "model_transport"
                # the model's own error (a 500) says nothing about the restored sessions
                if check and ended.status == "error" and not model_error and not used:
                    ch.restore_error = f"--continue failed: {ended.error}"[:500]
                    self.why = "resume failed"  # the adapter's next start raises ResumeError
                    return
                check = check and ended.status != "completed" and not used
                # A first turn that failed before using tokens left nothing to --continue.
                fresh = fresh and ended.status != "completed" and not used
                if ended.status != "error":
                    errors = 0
                elif not transient:  # a 429 storm is the loop's to back off, not a failure
                    errors += 1
            self.why = f"{MAX_ERRORS} failed turns in a row"
            if check:  # none worked on the restored sessions: the next start begins a fresh one
                ch.restore_error = f"{self.why} on the restored sessions: {ended.error}"[:500]
        finally:
            # A filled context stops Harbor recounting only the last turn after run().
            for name, value in totals.model_dump(exclude_none=True).items():
                setattr(context, name, value)

    async def _next_prompt(self, environment, deadline):
        """The next prompt, every queued one joined; None when the channel closes or, while
        idle, the Trial's time is up. Probes the environment while it waits."""
        queue = self.channel.prompts
        while True:
            left = deadline - time.monotonic()
            if left <= 0:  # queued prompts go to the next Trial
                self.why = "trial_hours"
                return None
            try:
                texts = [await asyncio.wait_for(queue.get(), min(PROBE, max(left, 0.01)))]
                break
            except TimeoutError:
                try:
                    result = await asyncio.wait_for(environment.exec("true", timeout_sec=30), 60)
                    if result.return_code != 0:
                        raise RuntimeError(f"exit code {result.return_code}")
                except Exception as exc:
                    self.why = f"idle probe failed: {safe_error(exc)[:300]}"
                    raise RuntimeError(self.why) from None
        while not queue.empty():
            texts.append(queue.get_nowait())
        if None in texts:
            self.why = "closed"
            return None
        return "\n\n".join(texts)

    async def _turn(self, text, first, environment, totals):
        """One `pi` call; reports it and returns (its turn_ended event, used tokens, Pi's
        own error if any)."""
        ch = self.channel
        self.busy, self._killed = True, False
        self.idle.clear()
        ch.emit(AdapterEvent("turn_started", data={"trial": self.logs_dir.parent.name}))
        output = self.logs_dir / self._OUTPUT_FILENAME
        with suppress(OSError):
            output.unlink()  # a turn that never ran must not recount the last one
        # shortcut: on hf-sandbox a turn that fails before Pi starts recounts the remote pi.txt
        # of the turn before (overcounts spend, the safe side); delete it remotely if it matters.
        error = None
        try:
            try:
                if first:
                    await super().run(text, environment, AgentContext())
                else:
                    await self.resume(text, environment, AgentContext())
            except Exception as exc:
                error = safe_error(exc)  # Harbor's: the command (with the prompt), then output
                error = error if len(error) <= 500 else f"{error[:60]} ... {error[-400:]}"
            try:
                await self._pull(environment, output)
            except Exception as exc:
                ch.emit(AdapterEvent("error", error=f"session sync: {safe_error(exc)[:300]}"))
            pi_error = terminal_pi_error(output, ch.max_turns)
            error = pi_error or error  # Pi's own words first
            messages, usage = read_turn(output)
            for message in messages:
                ch.emit(AdapterEvent("message", message=message))
            if usage != TokenUsage():
                ch.emit(AdapterEvent("usage", model=ch.model, usage=usage))
            totals.n_input_tokens += usage.input + usage.cache_read + usage.cache_write
            totals.n_output_tokens += usage.output
            totals.n_cache_tokens += usage.cache_read
            if self._killed:
                status, data = "interrupted", {}
            elif error:
                status = "error"
                transient = pi_error and transient_error(pi_error)
                data = {"reason": "model_transport"} if transient else {}
            else:
                status, data = "completed", {}
            ended = AdapterEvent("turn_ended", status=status, data=data,
                                 error=error if status == "error" else None)
            ch.emit(ended)
            return ended, usage != TokenUsage(), pi_error
        finally:
            self.busy = False
            self.idle.set()

    async def _pull(self, environment, output):
        """This turn's pi.txt and the session files: to the host when /logs/agent is not
        mounted (sessions every SYNC s), then the sessions to the channel's home."""
        local = self.logs_dir / self._SESSIONS_DIRECTORY
        if not environment.is_path_mounted(self.environment_logs_dir, self.logs_dir):
            remote = self.environment_logs_dir / self._OUTPUT_FILENAME
            await environment.download_file(remote.as_posix(), output)
            if time.monotonic() - self._synced >= SYNC:
                await environment.download_dir(
                    (self.environment_logs_dir / self._SESSIONS_DIRECTORY).as_posix(), local
                )
                self._synced = time.monotonic()
        self.keep_sessions()

    async def pull_sessions(self):
        """Unmounted (hf-sandbox): download the session files now (at most PULL_WAIT s) and
        keep them, so a push between turns is not up to SYNC s old. A failure is an `error`."""
        env = self._env
        if env is None or env.is_path_mounted(self.environment_logs_dir, self.logs_dir):
            return
        remote = (self.environment_logs_dir / self._SESSIONS_DIRECTORY).as_posix()
        try:
            await asyncio.wait_for(
                env.download_dir(remote, self.logs_dir / self._SESSIONS_DIRECTORY), PULL_WAIT)
        except Exception as exc:
            self._synced = float("-inf")  # maybe half a file here: the next turn downloads again
            self.channel.emit(AdapterEvent("error", error=f"session sync: {safe_error(exc)[:300]}"))
            return
        self._synced = time.monotonic()
        self.keep_sessions()

    def keep_sessions(self):
        """The host's copy of the session files to the channel's home, where BucketSync pushes
        them; also when the Trial ended (Harbor then downloaded /logs/agent)."""
        home = self.channel.home / "sessions"
        home.mkdir(parents=True, exist_ok=True)
        for path in (self.logs_dir / self._SESSIONS_DIRECTORY).glob("*.jsonl"):
            kept, now = home / path.name, path.stat()
            with suppress(FileNotFoundError):
                old = kept.stat()
                if (old.st_size, old.st_mtime_ns) == (now.st_size, now.st_mtime_ns):
                    continue  # unchanged since the last copy (copy2 keeps the mtime)
            temporary = home / f".{path.name}.tmp"  # BucketSync never sees half a file
            shutil.copy2(path, temporary)
            os.replace(temporary, kept)

    async def _restore(self, environment):
        """Upload the saved sessions, so the first turn continues them."""
        target = (self.environment_logs_dir / self._SESSIONS_DIRECTORY).as_posix()
        await self.exec_as_agent(environment, f"mkdir -p {shlex.quote(target)}")
        await environment.upload_dir(self.channel.home / "sessions", target)
        if environment.default_user is not None:  # upload_dir copies as root
            user = shlex.quote(str(environment.default_user))
            await self.exec_as_root(environment, f"chown -R {user} {shlex.quote(target)}")

    async def kill(self):
        """End the running `pi`; its turn ends `interrupted`. Unverified in a real container
        (whether --continue works after it is untested). A failed kill is an `error` event."""
        if not self.busy or self._env is None:
            return
        self._killed = True
        try:
            await self.exec_as_root(self._env, KILL_PI, timeout_sec=30)
        except Exception as exc:  # reported; the turn timeout still applies
            self.channel.emit(AdapterEvent("error", error=f"kill pi: {safe_error(exc)[:300]}"))
        with suppress(TimeoutError):
            await asyncio.wait_for(self.idle.wait(), STOP_GRACE)
