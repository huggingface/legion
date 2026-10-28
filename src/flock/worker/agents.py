"""AgentRuntime: one per assigned agent; process-per-burst (design.md section 4).

Phases of the harness process: none -> starting -> running -> (stopping) -> none.
- A prompt with no process starts a burst; prompts arriving while it starts are buffered and
  forwarded in order once it is up; later prompts go straight to the running process.
- The burst ends when the harness has been idle for `idle_grace`, when a turn exceeds its
  timeout (process killed, agent `stuck` until the next turn), on stop/unassign, on drain
  once the in-flight turn ends, or on `recycle` at the next turn boundary (a fresh process
  resumes the same session: memory relief, a new model endpoint).
The worker never schedules or queues beyond the startup buffer.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import shutil
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from flock.models import (
    AgentHeartbeat,
    Event,
    Grant,
    LoopMeta,
    LoopStartCommand,
    TokenUsage,
    TypeSpec,
    granted_spec,
    hit_max_turns,
    parse_duration,
    type_spec_rev,
)
from flock.secrets_file import secret_env
from flock.tools.toolset import PIN_ENV, Toolset, instructions_section, replace_section
from flock.worker import comms as comms_mod
from flock.worker.adapters.base import (
    AdapterEvent,
    HarnessAdapter,
    ResumeError,
    credential_env_names,
    usage_sub,
)
from flock.worker.events import Outbox, SessionLog, utcnow
from flock.worker.loop import render
from flock.worker.sync import WorkdirStats, reset_ephemeral
from flock.world import hub as world_hub

if TYPE_CHECKING:
    from flock.worker.comms import CommsHub
    from flock.worker.identity import Identities
    from flock.worker.sync import BucketSync
    from flock.world.hub import WorldHub

log = logging.getLogger(__name__)

INSTRUCTIONS_FILE = {"claude": "CLAUDE.md", "codex": "AGENTS.md", "pi": "AGENTS.md", "llm": "AGENTS.md"}
LOOP_ERROR_LIMIT = 3  # failed loop turns in a row end the worker-side loop
LOOP_ERROR_BACKOFF = 2.0  # s before re-sending the loop prompt after a failed turn
# Transient failures (turn timeout, model transport) do not count toward LOOP_ERROR_LIMIT at
# first; they are retried with exponential backoff and jitter. After LOOP_TRANSIENT_LIMIT of
# them in a row, each further one counts as an ordinary failed turn, so a dead endpoint ends
# the loop instead of restarting the harness forever.
LOOP_TRANSIENT_BACKOFF = 2.0  # s, first retry; doubles per consecutive transient failure
LOOP_TRANSIENT_BACKOFF_CAP = 120.0  # s
LOOP_TRANSIENT_LIMIT = 20  # consecutive transient retries before they count as errors


def transient_backoff(n: int, rng: Callable[[], float] = random.random) -> float:
    """Delay before the n-th (1-based) consecutive transient retry: 2 s, 4 s, ... capped,
    with jitter in [50%, 100%] so many agents behind one endpoint do not retry in step."""
    base = min(LOOP_TRANSIENT_BACKOFF_CAP, LOOP_TRANSIENT_BACKOFF * 2 ** max(0, n - 1))
    return base * (0.5 + 0.5 * rng())
# Interruptions that are a decision (stop, drain, unassign), not a failure: no re-prompt.
_DECIDED = ("stop", "drain", "unassign", "abandoned", "draining", "drain_deadline")
_UNSET: Any = object()  # set_live: "no news" for the grant (None means revoked)


class AgentState(BaseModel):
    """What must survive a worker change; synced next to the session log."""

    session_id: str | None = None
    turn_seq: int = 0  # last turn number handed out (turn ids are t-<n>)
    turns: int = 0  # turns ended (completed, failed or interrupted)
    tokens: dict[str, TokenUsage] = Field(default_factory=dict)  # cumulative, by model
    harness_cum: dict[str, TokenUsage] = Field(default_factory=dict)  # last cumulative snapshot
    harness_estimate_usd: float | None = None
    loop_round: int = 0  # last loop round sent to this agent (docs/loop.md)
    loop_run: str | None = None  # the loop run (LoopStartCommand.run) loop_round counts in
    last_summary: str | None = None  # what the agent last passed to `done` (else its last reply)
    # comms (docs/comms.md): posts made (max_posts), the last one's wall time (min_post_gap),
    # and the threads the agent posted in (thread-reply wakes), so a relaunch keeps them
    comms_cursor: int | None = None
    comms_board: str | None = None
    comms_created_channels: list[int] = Field(default_factory=list)
    comms_posts: int = 0
    comms_last_post: float | None = None
    comms_threads: list[int] = Field(default_factory=list)
    # board news (DMs, mentions, thread replies) not yet attached to a flock-tool result
    comms_news: list[dict] = Field(default_factory=list)


def restart_prompt(old_session: str | None, transcript: Path | None, session_log: Path, why: str) -> str:
    where = f"{transcript} (harness transcript) and {session_log}" if transcript else str(session_log)
    return (
        f"Your previous session ({old_session}) could not be resumed: {why}. "
        f"Its transcript is at {where}. Read it to recover your context, then continue your work."
    )


class AgentRuntime:
    def __init__(
        self,
        name: str,
        type_name: str,
        spec: TypeSpec,
        *,
        swarm: str,
        work_root: Path,
        adapter: HarnessAdapter,
        outbox: Outbox,
        sync: BucketSync | None = None,
        identities: Identities | None = None,
        working_count: Callable[[], int] | None = None,
        secret_source: Mapping[str, str] | None = None,
        toolbox: Callable[[], Toolset] | None = None,
        comms: CommsHub | None = None,
        world: WorldHub | None = None,
    ) -> None:
        self.name, self.type_name, self.spec, self.swarm = name, type_name, spec, swarm
        # Live changes (docs/live.md): `spec` is `base_spec` (the type spec as applied) with the
        # agent's grant on top. A newer type spec or grant from the controller waits in
        # `_next_*` until the agent's next turn boundary; `_notes` go in front of that prompt.
        self.base_spec: TypeSpec = spec
        self.grant: Grant | None = None
        self._next_spec: TypeSpec | None = None
        self._next_grant: Any = _UNSET
        self._notes: dict[str, None] = {}
        # type_spec_rev of base_spec at the last turn boundary, and the one the harness started on
        self.rev: str | None = None
        self.start_rev: str | None = None
        self.workdir_files: int | None = None  # files in the workdir at the last push
        self.workdir_stats: WorkdirStats | None = None  # the same walk's counts (`{workspace}`)
        self.work_root = Path(work_root)
        self.workdir = self.work_root / name
        self.sessions_dir = self.work_root / "_sessions"
        self.adapter = adapter
        self.outbox = outbox
        self.sync = sync
        self.identities = identities
        self._working_count = working_count
        self._secret_source = secret_source  # default: secret_env(), the secrets file over env
        self.log = SessionLog(self.sessions_dir / f"{name}.jsonl", swarm=swarm, agent=name,
                              harness=spec.harness.value, cwd=str(self.workdir))
        self.state = AgentState()
        self.phase = "none"
        self.buffer: list[str] = []
        self.turn_id: str | None = None
        self.awaiting = False  # a prompt was forwarded and its turn has not started yet
        self.stuck = False
        self.resumed: bool | None = None
        self.idle_since: datetime = utcnow()
        self.last_sync: datetime | None = None
        self.dirty = False
        self._env_dropped_rev: str | None = None
        self.allow_starts = True
        self._stop_req: str | None = None
        # recycle(): end the burst at the next turn boundary; the next prompt starts a fresh
        # process on the same session. Not _stop_req: the burst's finally must restart.
        self._recycle_pending: str | None = None
        self.pss_mb: float | None = None  # the harness's PSS, set by the worker's memory loop
        self._burst_started = 0.0
        self._burst_turns = 0
        self._exit_code: int | None = None  # the harness's exit status (`exited` event)
        self._burst: asyncio.Task | None = None
        self._ready = asyncio.Event()
        # setup() got past its restore (complete, partial or failed-to-list), not cancelled:
        # only then may anything push this agent (flock.worker.sync, "Push safety")
        self.setup_done = False
        self._turn_started_at = 0.0
        # prompt ids for the turn events (tracked broadcasts): the next turn's, the current one's
        self._pids: list[str] = []
        self._turn_pids: list[str] = []
        self._last_activity = 0.0
        self._turn_timeout = parse_duration(spec.turn_timeout).total_seconds()
        self._next_timeout: float | None = None
        self._turn_tokens: dict[str, TokenUsage] = {}
        self._warned_usage = False  # zero-usage warning logged for this agent
        self._instructions_written = False
        self.parent: str | None = None  # the spawning agent (AssignCommand.parent)
        self.flock_env: dict[str, str] = {}  # FLOCK_AGENT, FLOCK_AGENT_TOKEN, ... (worker sets)
        # Tools folder (docs/tools.md): the worker's current Toolset, read at each turn start.
        # An adapter with a `toolbox` attribute (llm) takes its own snapshot per turn.
        self.toolbox = toolbox
        if toolbox is not None and getattr(adapter, "toolbox", False) is None:
            adapter.toolbox = toolbox  # type: ignore[attr-defined]
        self._turn_tools: str | None = None  # tools version of the turn in flight
        self._tools_section: str | None = None  # last generated instructions section
        self._pinned: Toolset | None = None  # pinned when a prompt was forwarded between turns
        # `every: turn_end` (docs/loop.md): set by loop_start, cleared by loop_stop / a limit
        self.loop: LoopStartCommand | None = None
        self._loop_errors = 0
        self._loop_transient = 0  # consecutive timeout / model_transport turns
        self._loop_task: asyncio.Task | None = None
        self._turn_tool_calls = 0  # tool calls this turn, `done` excluded
        self._turn_state_calls = 0  # of those, the ones that are not read-only (idle_rounds)
        self._turn_finished = False  # `done(finished: true)` this turn (stop_when.all_agents_done)
        self._turn_call_names: list[str] = []  # their tool names, in order
        self._turn_summary: str | None = None  # `done`'s summary this turn
        self._turn_text: str | None = None  # last assistant text this turn
        self._loop_inflight: LoopMeta | None = None  # a loop prompt taken, its turn not started
        self.comms = comms  # the worker's comms hub (docs/comms.md); None: no comms
        self._comms_section: str | None = None  # last comms section written to the instructions
        self.world = world  # the worker's world hub (docs/world.md); None: no world
        self._world_section: str | None = None

    # ------------------------------------------------------------------ paths and state
    def purge(self) -> None:
        """Delete the agent's files on this worker: its workdir, session log, state, pin and
        the harness's own session dir (an ephemeral agent, after its final sync; docs/spawn.md)."""
        shutil.rmtree(self.workdir, ignore_errors=True)
        if isinstance(sd := getattr(self.adapter, "session_dir", None), Path):
            shutil.rmtree(sd, ignore_errors=True)
        for p in (*self.sessions_dir.glob(f"{self.name}.*"), self.pin_path):
            p.unlink(missing_ok=True)

    @property
    def state_path(self) -> Path:
        return self.sessions_dir / f"{self.name}.state.json"

    @property
    def pin_path(self) -> Path:
        """The Toolset `flock-tool` uses (FLOCK_TOOLS_PIN), rewritten at each turn start."""
        from flock.isolation import enabled

        if enabled():
            return self.work_root / "_tools_pins" / self.name / "pin.json"
        return self.work_root / "_tools_pins" / f"{self.name}.json"

    def _pin_tools(self) -> Toolset | None:
        """The current Toolset, written to the pin file `flock-tool` reads. The harness has the
        pin's path from its start (PIN_ENV), so a version, an enable or a disable reaches a
        running harness at the next turn boundary: with no tools the pin holds an empty
        toolset (written once a pin exists, or when the worker gateway is open: the `flock`
        CLI reaches the controller through it, docs/spawn.md)."""
        if self.toolbox is None:
            return None
        ts = self.toolbox()
        if ts.version is not None or ts.tools or ts.gateway or self.pin_path.exists():
            deny = credential_env_names(self.spec)  # the key under every name the harness has it
            if self.comms is not None:  # and the board's credentials (docs/comms.md)
                deny += self.comms.deny_env()
            if self.world is not None:  # and the world's secret (docs/world.md)
                deny += self.world.deny_env()
            try:
                ts.write_pin(self.pin_path, agent=self.name, workdir=self.workdir, env_deny=deny)
            except OSError as e:
                log.warning("%s: cannot write the tools pin: %s", self.name, e)
        self._write_tools_section(ts)
        self._write_comms_section()
        self._write_world_section()
        return ts

    @property
    def idle_grace(self) -> float:
        return parse_duration(self.spec.idle_grace).total_seconds()

    def load_state(self) -> None:
        if self.state_path.is_file():
            try:
                self.state = AgentState.model_validate_json(self.state_path.read_text())
            except ValueError as e:
                log.warning("%s: unreadable state file: %s", self.name, e)
        self.log.reload()

    def save_state(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(self.state.model_dump_json())
        os.replace(tmp, self.state_path)

    def harness_files(self) -> list[Path]:
        return self.adapter.session_files(self.state.session_id)

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    # ------------------------------------------------------------------ events
    def _emit(self, ev: Event) -> None:
        self.dirty = True
        self.outbox.add(ev)

    def emit(self, kind: str, data: dict[str, Any] | None = None) -> None:
        self._emit(self.log.custom(kind, data, turn_id=self.turn_id))

    def _report_env_dropped(self) -> None:
        """Names (never values) the harness isolation dropped; once per spec revision."""
        names = getattr(self.adapter, "env_dropped_names", None)
        rev = type_spec_rev(self.spec)
        if names and rev != self._env_dropped_rev:
            self._env_dropped_rev = rev
            log.warning("%s: isolation dropped env names %s", self.name, ", ".join(names))
            self.emit("env_dropped", {"names": names})

    # ------------------------------------------------------------------ live changes (docs/live.md)
    def set_live(self, *, spec: TypeSpec | None = None, grant: Any = _UNSET) -> None:
        """A changed type spec, or the grant now in force (None: none), from the controller.
        Held until the agent's next turn boundary; the latest wins."""
        if spec is not None:
            self._next_spec = spec
        if grant is not _UNSET:
            self._next_grant = grant

    def _grant_now(self) -> Grant | None:
        return self.grant if self._next_grant is _UNSET else self._next_grant

    def _apply_live(self) -> None:
        """At a turn boundary: take the pending type spec and grant. A changed system prompt
        rewrites CLAUDE.md / AGENTS.md (the llm harness rebuilds its system message from it at
        the turn start); an agent that already had a turn gets a one-line note in its next
        prompt for each kind of change."""
        spec, self._next_spec = self._next_spec, None
        grant, self._next_grant = self._next_grant, _UNSET
        before = (self.base_spec.system_prompt, self.base_spec.system_prompt_file)
        self.base_spec = spec or self.base_spec
        self.rev = type_spec_rev(self.base_spec)
        regrant = grant is not _UNSET and grant != self.grant
        if regrant:
            self.grant = grant
        budget = self.grant is not None and self.grant.limits.budget_usd is not None
        self.spec = granted_spec(self.base_spec, self.grant, self.spend_usd() if budget else 0.0)
        if (self.spec.system_prompt, self.spec.system_prompt_file) != before:
            if self._instructions_written:
                self._instructions_written = False
                self._tools_section = self._comms_section = None  # sections are rewritten
                self._write_instructions()
            harness = self.spec.harness.value
            self._note("your system prompt changed" if harness == "llm" else
                       f"your instructions in {INSTRUCTIONS_FILE[harness]} changed; re-read them")
        if regrant:
            self._note(f"your grants changed: {self.grant.describe() if self.grant else 'all revoked'}")
        if self.comms is not None:
            self.comms.set_grant(self.name, self.grant.tools if self.grant else None)
        if (update := getattr(self.adapter, "update_spec", None)) and update(self.spec):
            self.recycle("endpoint")  # the new models.json needs a fresh process

    def _note(self, what: str) -> None:
        if self.state.turn_seq > 0:  # an agent that never had a turn starts on the new setup
            self._notes[what] = None

    def _with_notes(self, text: str) -> str:
        if not self._notes:
            return text
        notes, self._notes = "; ".join(self._notes), {}
        return f"[flock] Since your last turn: {notes}.\n\n{text}"

    def _over_budget(self) -> float | None:
        """The granted limits.budget_usd when the agent's cost reached it, else None."""
        g = self._grant_now()
        if g is None or g.limits.budget_usd is None:
            return None
        return g.limits.budget_usd if self.spend_usd() >= g.limits.budget_usd else None

    # ------------------------------------------------------------------ lifecycle
    async def setup(self, restore: bool = True, resume_prompt: str | None = None) -> None:
        """Run once on assign: restore from the bucket, load state, then allow starts."""
        try:
            res = None
            if restore and self.sync is not None:
                try:
                    res = await self.sync.restore(self)
                except Exception as e:  # noqa: BLE001 - a failed listing means a fresh start
                    log.warning("%s: restore failed (fresh start): %s", self.name, e)
                    self.emit("error", {"error": f"restore failed: {e}"})
            await asyncio.to_thread(reset_ephemeral, self.workdir, self.spec)
            self.setup_done = True
            self.load_state()
            if res is not None and res.failed:
                # files skipped after retries; a missing harness session file makes the next
                # start take the ResumeError path (session_restarted with a restart prompt)
                self.emit("restore_partial", {"failed": len(res.failed), "paths": res.failed[:10]})
        finally:
            self._ready.set()
        if resume_prompt:
            await self.prompt(resume_prompt)

    async def prompt(self, text: str, *, prompt_id: str | None = None, timeout: str | None = None,
                     loop: LoopMeta | None = None) -> None:
        if loop is not None:
            text = self._loop_prompt(text, loop)
        if (budget := self._over_budget()) is not None:
            self.emit("prompt_deferred", {"prompt_id": prompt_id, "text": text, "reason": "budget",
                                          "budget_usd": budget, **self._loop_untaken()})
            return
        if timeout:
            self._next_timeout = parse_duration(timeout).total_seconds()
        if prompt_id:  # joins the turn in flight, or starts the next one
            (self._turn_pids if self.turn_id is not None else self._pids).append(prompt_id)
        if self.phase == "running" and not self._stop_req:
            await self._forward(text)
            return
        if self.phase == "none" and (not self.allow_starts or self._stop_req):
            self.emit("prompt_deferred", {"prompt_id": prompt_id, "text": text,
                                          "reason": "draining" if not self.allow_starts else self._stop_req,
                                          **self._loop_untaken()})
            return
        self.buffer.append(text)
        if self.phase == "none":
            self.phase = "starting"
            self._burst = asyncio.create_task(self._run_burst(), name=f"burst-{self.name}")

    async def _forward(self, text: str) -> None:
        self._last_activity = self._now()
        if self.turn_id is None:
            if not self.awaiting:
                # a turn boundary: the prompt starts a turn, so take the live changes and pin
                # the tools it will run with. While awaiting, an earlier prompt's turn may already
                # run in the harness (its turn_started not seen yet): this prompt joins it and
                # must not repin it.
                self._apply_live()
                if self._recycle_pending:  # the fresh process takes it (_watch ends this one)
                    self.buffer.append(text)
                    return
                self._pinned = self._pin_tools()
                text = self._with_notes(text)
            self.awaiting = True
        try:
            await self.adapter.send(text)
        except Exception as e:  # noqa: BLE001 - process died under us: keep the prompt for the next burst
            log.warning("%s: send failed (%s); buffering for the next start", self.name, e)
            self.buffer.append(text)

    def _defer_buffer(self, reason: str) -> None:
        self._pids = []
        for text in self.buffer:
            self.emit("prompt_deferred", {"text": text, "reason": reason, **self._loop_untaken()})
        self.buffer.clear()

    def _prepare_workdir(self, tools: Toolset | None = None) -> None:
        self._write_instructions()
        if tools is not None:
            self._write_tools_section(tools)
        self._write_comms_section()
        self._write_world_section()

    def _write_world_section(self) -> None:
        """The World section (name, version, tools, a "world changed" note) of CLAUDE.md /
        AGENTS.md, rewritten when it changes (docs/world.md)."""
        if self.world is None or not self.workdir.is_dir():
            return
        section = self.world.section(self.name)
        if section != (self._world_section or ""):
            target = self.workdir / INSTRUCTIONS_FILE[self.spec.harness.value]
            text = target.read_text() if target.is_file() else ""
            target.write_text(replace_section(text, section, world_hub.SECTION_START,
                                              world_hub.SECTION_END))
        self._world_section = section

    def _write_comms_section(self) -> None:
        """The comms section (board identity, rules, protocol files) of CLAUDE.md / AGENTS.md,
        rewritten when it changes; the llm harness reads AGENTS.md into its system message."""
        if self.comms is None or not self.workdir.is_dir():
            return
        section = self.comms.section(self.name)
        if section != (self._comms_section or ""):
            target = self.workdir / INSTRUCTIONS_FILE[self.spec.harness.value]
            text = target.read_text() if target.is_file() else ""
            target.write_text(replace_section(text, section, comms_mod.SECTION_START,
                                              comms_mod.SECTION_END))
            if self._comms_section is not None:  # docs/live.md: rules or protocols changed
                self._note(f"the Team board section of {target.name} changed; re-read it")
        self._comms_section = section

    def _write_tools_section(self, tools: Toolset) -> None:
        """The generated Tools section of CLAUDE.md / AGENTS.md (CLI harnesses; llm gets function
        tools instead), rewritten when the toolset changes."""
        if self.spec.harness.value == "llm" or not self.workdir.is_dir():
            return
        section = instructions_section(tools)
        if section != (self._tools_section or ""):
            target = self.workdir / INSTRUCTIONS_FILE[self.spec.harness.value]
            text = target.read_text() if target.is_file() else ""
            target.write_text(replace_section(text, section))
            if self._tools_section is not None:  # docs/live.md: a running harness reads it again
                self._note(f"your tools changed; re-read the Tools section of {target.name}")
        self._tools_section = section

    def _write_instructions(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        if self._instructions_written:
            return
        target = self.workdir / INSTRUCTIONS_FILE[self.spec.harness.value]
        if self.spec.system_prompt is not None:
            target.write_text(self.spec.system_prompt)
            self._instructions_written = True
            self._tools_section = None  # the base file was replaced: its section is gone
            return
        if not self.spec.system_prompt_file:
            return
        src = Path(self.spec.system_prompt_file)
        candidates = [src] if src.is_absolute() else [
            Path(os.environ.get("FLOCK_PROMPTS_DIR", ".")) / src, Path.cwd() / src]
        for c in candidates:
            if c.is_file():
                target.write_text(c.read_text())
                self._instructions_written = True
                self._tools_section = None  # the base file was replaced: its section is gone
                return
        log.warning("%s: system_prompt_file %s not found", self.name, src)

    async def harness_env(self) -> dict[str, str]:
        """What this agent's harness process gets beyond the adapter's allowlist: the type's
        declared secrets, `spec.env`, the msgboard identity, and a provider-backed model's key
        under its `api_key_env` name (read from the worker's environment, where the job puts
        it). Nothing else from the worker: FLOCK_TOKEN and the rest stay here."""
        source = self._secret_source if self._secret_source is not None else secret_env()
        env = {k: source[k] for k in self.spec.secrets if k in source}
        env.update(self.spec.env)
        if self.identities is not None:
            env.update(await self.identities.env_for(self.name, self.spec))
        if self.comms is not None:  # the board invite stays in the worker: it joins any name
            for k in self.comms.deny_env():
                if k != "MSGBOARD_TOKEN":
                    env.pop(k, None)
        if self.world is not None:  # the world's secret stays in the worker
            for k in self.world.deny_env():
                env.pop(k, None)
        r = self.spec.resolved_model
        if r is not None:
            key = source.get(r.provider.api_key_env)
            if key:
                env[r.provider.api_key_env] = key
            else:
                log.warning("%s: %s (key for provider %s) is not set on this worker",
                            self.name, r.provider.api_key_env, r.provider_name)
                self.emit("error", {"error": f"{r.provider.api_key_env} is not set on the worker; "
                                             f"provider {r.provider_name} will reject requests"})
        return env

    async def _run_burst(self) -> None:
        pump: asyncio.Task | None = None
        try:
            await self._ready.wait()
            if not self.allow_starts or self._stop_req:
                self._defer_buffer("draining" if not self.allow_starts else str(self._stop_req))
                return
            if self._working_count and self._working_count() > self.spec.max_working_per_worker:
                log.warning("%s: %d live harness processes exceed max_working_per_worker=%d",
                            self.name, self._working_count(), self.spec.max_working_per_worker)
            self._apply_live()  # the harness starts on the newest type spec and grant
            if self.comms is not None:  # join the board at the first prompt, not at assign
                await self.comms.on_prompt(self.name, self.spec.harness.value)
            if self.world is not None:  # open the gateway for flock-tool; agent_start hook
                await self.world.on_prompt(self.name, self.spec.harness.value, self.comms)
            self._write_instructions()  # before the tools section is generated into it
            tools = self._pin_tools()
            self._prepare_workdir(tools)
            env = await self.harness_env()
            old = self.state.session_id
            env.update(self.flock_env)
            if tools is not None:  # always: tools enabled mid-burst must reach this process
                env[PIN_ENV] = str(self.pin_path)
            first: list[str] = []
            data: dict[str, Any] = {"harness": self.spec.harness.value, "model": self.spec.model}
            if self.spec.resolved_model is not None:
                data.update(provider=self.spec.provider_name, model_id=self.spec.harness_model)
            try:
                sid = await self.adapter.start(self.workdir, old, self.spec, env=env)
                kind, self.resumed = ("session_resumed", True) if old else ("session_started", False)
            except ResumeError as e:
                transcript = next((p for p in self.adapter.session_files(old) if old and old in p.name), None)
                sid = await self.adapter.start(self.workdir, None, self.spec, env=env)
                kind, self.resumed = "session_restarted", False
                self.state.harness_cum = {}
                first.append(restart_prompt(old, transcript, self.log.path, str(e)))
                data.update(previous_session_id=old, reason=str(e))
            self.state.session_id = sid
            self.start_rev = self.rev
            self._recycle_pending, self._exit_code = None, None
            self._burst_started, self._burst_turns = self._now(), 0
            self.save_state()
            self._report_env_dropped()
            self.emit(kind, {**data, "session_id": sid, "resumed": self.resumed,
                             "parent": self.parent})
            self.phase = "running"
            self._last_activity = self._now()
            pump = asyncio.create_task(self._pump())
            pending, self.buffer = first + self.buffer, []
            for text in pending:
                if self._stop_req and self._stop_req != "drain":
                    self.buffer.append(text)
                    continue
                await self._forward(text)
            if self._stop_req and self._stop_req != "drain":
                self._defer_buffer(self._stop_req)
            await self._watch(pump)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("%s: burst failed", self.name)
            self.emit("error", {"error": f"{type(e).__name__}: {e}", "dropped_prompts": len(self.buffer)})
            self.buffer.clear()
        finally:
            self.phase = "stopping"
            await self.adapter.stop()
            if pump is not None:
                try:
                    await asyncio.wait_for(pump, 10)
                except (TimeoutError, asyncio.CancelledError):
                    pump.cancel()
            if self.turn_id is not None:  # SIGKILLed: the OOM killer (or the memory watchdog)
                self._interrupt("oom_killed" if self._exit_code in (-9, 137) else "process exited")
            self.awaiting = False
            self.phase = "none"
            self._burst = None
            self.dirty = True
            self.save_state()
            if self.buffer and self.allow_starts and not self._stop_req:
                self.phase = "starting"
                self._burst = asyncio.create_task(self._run_burst(), name=f"burst-{self.name}")

    async def _watch(self, pump: asyncio.Task) -> None:
        tick = max(0.01, min(1.0, self.idle_grace / 10, self._turn_timeout / 10))
        while True:
            await asyncio.wait({pump}, timeout=tick)
            if pump.done():
                return
            now = self._now()
            busy = self.turn_id is not None or self.awaiting
            if self._stop_req and (self._stop_req != "drain" or not busy):
                if busy:
                    self._interrupt(self._stop_req)
                return
            timeout = self._turn_timeout
            started = self._turn_started_at if self.turn_id else self._last_activity
            if busy and now - started > timeout:
                log.warning("%s: turn exceeded %.0fs; killing the harness", self.name, timeout)
                self._interrupt("timeout")
                self.stuck = True
                return
            if not busy and self._recycle_pending:
                self.emit("harness_recycled", {"reason": self._recycle_pending, "pss_mb": self.pss_mb})
                self._recycle_pending = None
                return
            if not busy and now - self._last_activity >= self.idle_grace:
                return

    async def _pump(self) -> None:
        try:
            async for ev in self.adapter.events():
                self.on_adapter_event(ev)
        except Exception:
            log.exception("%s: adapter event stream failed", self.name)

    # ------------------------------------------------------------------ turns and usage
    def on_adapter_event(self, ev: AdapterEvent) -> None:
        k = ev.kind
        if k == "turn_started":
            if self.turn_id is None:
                self._begin_turn(ev.data)
        elif k == "message" and ev.message is not None:
            self._note_message(ev.message)
            self._emit(self.log.message(ev.message, turn_id=self.turn_id))
        elif k == "usage" and ev.usage is not None:
            self._account(ev)
        elif k == "turn_ended":
            if self.turn_id is not None:
                tokens = {m: u.model_dump() for m, u in self._turn_tokens.items()}
                extra = {}
                if self._usage_missing(ev.status):
                    extra["usage_missing"] = True
                if self._turn_tools is not None:
                    extra["tools_version"] = self._turn_tools
                if self._turn_finished:
                    extra["finished"] = True
                extra.update(self._pid_fields())
                self.emit("turn_ended", {"status": ev.status, "error": ev.error, "tokens": tokens,
                                         "tool_calls": self._turn_tool_calls,
                                         "state_calls": self._turn_state_calls, **extra,
                                         **{k2: v for k2, v in ev.data.items() if k2 != "tokens"}})
                if self.world is not None:  # the `turn` hook (docs/world.md)
                    self.world.on_turn(self.name, self.turn_id,
                                       {"status": ev.status or "completed",
                                        "tool_calls": self._turn_tool_calls})
                summary = self._turn_summary or self._turn_text
                if self._turn_summary is None and ev.status not in (None, "completed"):
                    summary = self._unfinished_summary(ev.data)  # not a stale or half-way text
                if summary:
                    self.state.last_summary = summary
                finished = self._turn_finished
                self._end_turn()
                self._burst_turns += 1
                self._check_recycle_after()
                lp = self.spec.loop
                if finished and self.loop is not None and lp is not None and lp.stop_when.all_agents_done:
                    self._loop_end("done")  # no next round; the controller takes it out too
                self._after_turn(ev.status or "completed",
                                 "max_turns" if hit_max_turns(ev.data) else ev.data.get("reason"))
        elif k == "error":
            self.emit("error", {"error": ev.error, **ev.data})
        elif k == "exited":
            self._exit_code = ev.data.get("returncode")
        self.dirty = True

    def _unfinished_summary(self, data: dict[str, Any]) -> str:
        """`{last_summary}` after a turn that stopped without `done` (max_turns, budget, error):
        say so, so the next loop prompt does not repeat an older turn's summary."""
        why = data.get("reason") or data.get("subtype") or "an error"
        last = ", ".join(self._turn_call_names[-3:]) or "none"
        return (f"previous turn ended ({why}) after {self._turn_tool_calls} tool call(s) without "
                f"finishing (last calls: {last})")

    def _usage_missing(self, status: str | None) -> bool:
        """A provider-backed turn that completed with zero reported tokens: the server sent no
        usage (the HF router streams zeros for some providers), so its cost is unknown. Logged
        once per agent; the controller counts such agents in swarm status."""
        if self.spec.resolved_model is None or status not in (None, "completed"):
            return False
        if any(sum(u.model_dump().values()) for u in self._turn_tokens.values()):
            return False
        if not self._warned_usage:
            self._warned_usage = True
            log.warning("%s: turn completed with zero reported token usage from provider %s; "
                        "cost is unknown", self.name, self.spec.resolved_model.provider_name)
        return True

    def _begin_turn(self, data: dict[str, Any]) -> None:
        self.state.turn_seq += 1
        self.turn_id = f"t-{self.state.turn_seq}"
        self.stuck = False
        self.awaiting = False
        self._turn_started_at = self._now()
        self._turn_timeout = self._next_timeout or parse_duration(self.spec.turn_timeout).total_seconds()
        self._next_timeout = None
        self._turn_tokens = {}
        data = dict(data)
        if self.toolbox is not None and "tools_version" not in data:
            # CLI harnesses: what the prompt pinned (else pin now); flock-tool uses it to turn end
            ts, self._pinned = self._pinned or self._pin_tools(), None
            if ts is not None and ts.version is not None:
                data["tools_version"] = ts.version
            if ts is not None and ts.error:
                data["tools_error"] = ts.error
        self._turn_tools = data.get("tools_version")
        self._turn_tool_calls, self._turn_summary, self._turn_text = 0, None, None
        self._turn_state_calls, self._turn_finished = 0, False
        self._turn_call_names = []
        self._loop_inflight = None
        self._turn_pids, self._pids = self._pids, []
        self.emit("turn_started", {**data, **self._pid_fields()})

    def _pid_fields(self) -> dict[str, Any]:
        """`prompt_id` (the last prompt of the turn), and `prompt_ids` when it took several."""
        p = self._turn_pids
        return ({"prompt_id": p[-1]} if p else {}) | ({"prompt_ids": list(p)} if len(p) > 1 else {})

    def _end_turn(self) -> None:
        self._turn_pids = []
        self.state.turns += 1
        self.turn_id = None
        self._turn_tools = None
        self._pinned = None
        self.awaiting = False
        self.idle_since = utcnow()
        self._last_activity = self._now()
        self.save_state()

    def _interrupt(self, reason: str) -> None:
        if self.turn_id is not None:
            extra = {"tools_version": self._turn_tools} if self._turn_tools is not None else {}
            self.emit("turn_interrupted", {"reason": reason, **extra})
            self._end_turn()
            self._after_turn("interrupted", reason)
        self.awaiting = False

    def _account(self, ev: AdapterEvent) -> None:
        model = ev.model or self.spec.model
        usage = ev.usage
        assert usage is not None
        if ev.cost_usd is not None:
            self.state.harness_estimate_usd = ev.cost_usd
        if ev.cumulative:
            prev = self.state.harness_cum.get(model)
            self.state.harness_cum[model] = usage
            if ev.baseline:
                return
            usage = usage_sub(usage, prev)
        self.state.tokens[model] = self.state.tokens.get(model, TokenUsage()) + usage
        self._turn_tokens[model] = self._turn_tokens.get(model, TokenUsage()) + usage

    # ------------------------------------------------------------------ loop (docs/loop.md)
    def _note_message(self, m: dict[str, Any]) -> None:
        """Per turn: tool calls (not `done`), those that are not read-only (the `llm` adapter
        marks a call `readonly` from the tools version it ran with: comms read/search,
        `readonly` folder tools), `done`'s summary and the last assistant text."""
        if m.get("role") != "assistant":
            return
        content = m.get("content")
        blocks = content if isinstance(content, list) else []
        texts = [b.get("text") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        if isinstance(content, str):
            texts = [content]
        text = "\n".join(t for t in texts if isinstance(t, str) and t.strip()).strip()
        calls = [b for b in blocks if isinstance(b, dict)
                 and b.get("type") == "toolCall" and b.get("name") != "done"]
        self._turn_tool_calls += len(calls)
        self._turn_state_calls += sum(1 for b in calls if b.get("readonly") is not True)
        if m.get("fromDoneTool") and m.get("finished") is True:
            self._turn_finished = True
        self._turn_call_names += [str(b.get("name")) for b in calls]
        if m.get("fromDoneTool") and text:
            self._turn_summary = text
        elif text:
            self._turn_text = text

    def spend_usd(self) -> float:
        from flock.controller.prices import cost_usd, type_price_table

        return cost_usd(self.state.tokens, type_price_table(self.spec))

    def _loop_prompt(self, template: str, meta: LoopMeta) -> str:
        """Number and render a loop prompt; record it as a `loop_prompt` event."""
        if meta.kind == "loop":
            # a wake prompt outside max_rounds (counted false) keeps the current round
            n = (meta.round if meta.round is not None
                 else self.state.loop_round + int(meta.counted))
            self.state.loop_round = max(self.state.loop_round, n)
            meta = meta.model_copy(update={"round": n})
        elif meta.kind == "final" and self.loop is not None:
            self._loop_end("final")
        self._loop_inflight = meta
        self.emit("loop_prompt", {"kind": meta.kind, "round": meta.round, "why": meta.why,
                                  **({"add_tools": meta.add_tools} if meta.add_tools else {}),
                                  **({} if meta.counted else {"counted": False}),
                                  **({"wake": True} if meta.wake else {})})
        return render(template, meta, spend=self.spend_usd(), last_summary=self.state.last_summary,
                      now=utcnow(), workspace=self.workdir_stats.summary()
                      if self.workdir_stats else None)

    def _loop_untaken(self) -> dict[str, Any]:
        """A deferred prompt that was a loop prompt: it does not count as a round, and the
        controller learns which one it was (`loop`) so it can send it again."""
        m, self._loop_inflight = self._loop_inflight, None
        if m is None:
            return {}
        wake = {"wake": True} if m.wake else {}
        if not m.counted:
            return {"loop": {"kind": m.kind, "round": m.round, "counted": False, **wake}}
        if m.kind == "loop" and m.round is not None:
            self.state.loop_round = min(self.state.loop_round, max(0, m.round - 1))
        return {"loop": {"kind": m.kind, "round": m.round, **wake}}

    def start_loop(self, cmd: LoopStartCommand) -> None:
        """`loop_start`: send the loop prompt now if idle, then after every completed turn. A
        new run (`flock loop --start`) counts rounds from the command's; a repeat of the same
        run keeps the higher count."""
        self.loop = cmd
        if cmd.run is not None and cmd.run != self.state.loop_run:
            self.state.loop_run, self.state.loop_round = cmd.run, cmd.meta.round or 0
        else:
            self.state.loop_round = max(self.state.loop_round, cmd.meta.round or 0)
        self._loop_errors = 0
        self._loop_transient = 0
        if self.turn_id is None and not self.awaiting and not self.buffer:
            self._loop_schedule(0.0, "start")

    def stop_loop(self, reason: str = "stop") -> None:
        """`loop_stop`: no more worker-side loop prompts; the turn in flight finishes. Always
        answered with `loop_stopped` (the controller waits for it before the final round)."""
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
        active, self.loop = self.loop is not None, None
        self.emit("loop_stopped", {"reason": reason, "round": self.state.loop_round, "active": active})

    def _loop_end(self, reason: str) -> None:
        self.loop = None
        self.emit("loop_stopped", {"reason": reason, "round": self.state.loop_round, "active": True})

    def _after_turn(self, status: str, reason: str | None = None) -> None:
        if self.loop is None:
            return
        if status == "completed" or reason == "max_turns":  # max_turns: a full turn, not a failure
            self._loop_errors = self._loop_transient = 0
            self._loop_schedule(0.0, "turn_end")
            return
        if status == "interrupted" and (reason is None or reason in _DECIDED):
            return  # stopped on purpose: the controller decides what comes next
        if reason in ("timeout", "model_transport", "oom_killed"):
            self._loop_transient += 1
            if self._loop_transient <= LOOP_TRANSIENT_LIMIT:
                self._loop_schedule(transient_backoff(self._loop_transient), f"retry after {reason}")
                return
            log.warning("%s: %d %s failures in a row; counting them as errors", self.name,
                        self._loop_transient, reason)
        else:
            self._loop_transient = 0
        self._loop_errors += 1
        if self._loop_errors >= LOOP_ERROR_LIMIT:
            self._loop_end("errors")
            return
        self._loop_schedule(LOOP_ERROR_BACKOFF, f"retry after {reason or status}")

    def _loop_schedule(self, delay: float, why: str) -> None:
        if self._loop_task is not None and not self._loop_task.done():
            return
        self._loop_task = asyncio.create_task(self._loop_next(delay, why), name=f"loop-{self.name}")

    async def _loop_next(self, delay: float, why: str) -> None:
        if delay:
            await asyncio.sleep(delay)
        lp = self.loop
        if lp is None or not self.allow_starts or self._stop_req:
            return
        if self.turn_id is not None or self.awaiting or self.buffer:
            return  # a turn is already on its way; its end schedules the next round
        if lp.max_rounds is not None and self.state.loop_round >= lp.max_rounds:
            self._loop_end("max_rounds")
            return
        if lp.deadline is not None and utcnow() >= lp.deadline:
            self._loop_end("deadline")
            return
        await self.prompt(lp.text, loop=lp.meta.model_copy(update={"round": None, "why": why}))

    # ------------------------------------------------------------------ control
    @property
    def live(self) -> bool:
        return self.phase != "none"

    def recycle(self, reason: str) -> bool:
        """Restart the harness process at the next turn boundary, resuming the same session
        (`harness_recycled`). Only a running process; False when there is none to recycle."""
        if self.phase != "running" or self._recycle_pending:
            return False
        self._recycle_pending = reason
        return True

    def _check_recycle_after(self) -> None:
        """`recycle_after` (TypeSpec), checked at each turn end."""
        ra = self.spec.recycle_after
        if ra is None:
            return
        if ra.turns and self._burst_turns >= ra.turns:
            self.recycle("turns")
        elif ra.hours and self._now() - self._burst_started >= ra.hours * 3600:
            self.recycle("hours")
        elif ra.pss_mb and self.pss_mb is not None and self.pss_mb >= ra.pss_mb:
            self.recycle("pss_mb")

    async def stop(self, reason: str = "stop") -> None:
        """Kill the harness process if any (interrupting a turn); the agent stays assigned."""
        self._stop_req = reason
        try:
            if self._burst is not None:
                await self._burst
        finally:
            self._stop_req = None
        self._defer_buffer(reason)

    async def drain(self, deadline: float | None = None) -> bool:
        """No new starts; let the in-flight turn finish (bounded by its timeout, and by
        `deadline` s when given: then it is interrupted, `drain_deadline`); final sync. True
        when a turn was cut."""
        self.allow_starts = False
        cut = False
        if (burst := self._burst) is not None:
            self._stop_req = "drain"
            try:
                if deadline is None:
                    await burst
                else:
                    try:
                        await asyncio.wait_for(asyncio.shield(burst), deadline)
                    except TimeoutError:
                        cut = self.turn_id is not None or self.awaiting
                        self._stop_req = "drain_deadline"
                        await burst
            finally:
                self._stop_req = None
        self._defer_buffer("draining")
        await self.sync_now("drain")
        return cut

    async def unassign(self) -> None:
        self.allow_starts = False
        await self.stop("unassign")
        await self.sync_now("unassign")

    async def sync_now(self, reason: str) -> None:
        """Push the agent to the bucket; every push path (periodic, drain, preempt, unassign)
        goes through here. Before setup() finished its restore nothing is pushed: the bucket
        copy is the truth and the half-restored workdir must not overwrite it."""
        if self.sync is None:
            return
        if not self.setup_done:
            log.warning("%s: %s sync skipped, the agent's restore has not finished; "
                        "the bucket copy stays as it is", self.name, reason)
            return
        self.emit("sync", {"reason": reason})
        self.save_state()
        try:
            await self.sync.push(self)
            self.last_sync = utcnow()
            self.dirty = False
        except Exception as e:  # noqa: BLE001
            log.warning("%s: sync failed: %s", self.name, e)

    def heartbeat(self) -> AgentHeartbeat:
        working = self.turn_id is not None or self.awaiting or (self.phase == "starting" and bool(self.buffer))
        state = "working" if working else ("stuck" if self.stuck else "idle")
        return AgentHeartbeat(
            name=self.name,
            state=state,
            turn_id=self.turn_id,
            idle_since=None if working else self.idle_since,
            session_id=self.state.session_id,
            resumed=self.resumed,
            tokens=dict(self.state.tokens),
            harness_estimate_usd=self.state.harness_estimate_usd,
            turns=self.state.turns,
            last_sync=self.last_sync,
            msgboard_principal=self.identities.principal(self.name) if self.identities else None,
            rev=self.rev,
            start_rev=self.start_rev,
        )
