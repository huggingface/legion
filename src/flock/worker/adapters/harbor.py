"""Harness `harbor`: each agent is one long-lived Harbor Trial on shepherd's task path
(flock.harbor), in which FlockPi (flock.harbor.agent) runs one `pi` turn per prompt.

The Trial is built as shepherd builds it (build_trial_config, the resource bind, Trial.create,
attach_pi_setup, attach_tool_setup) with FlockPi as the agent, and `trial.run()` runs in the
background for as long as the agent lives here:
- `start` reuses the Trial when it is up; otherwise it starts one, putting back the saved
  sessions when there are any (none for a session to resume: ResumeError), and waits until
  Pi is installed.
- `send` queues the prompt for the next turn (prompts sent during a turn join the next one);
  a Trial that ended on its own (limits.trial_hours) is replaced by a new one for it.
- `stop` kills a turn in flight and ends the burst's event stream; the Trial stays up.
- `close` ends the Trial: FlockPi returns, Harbor runs the verifier and stops the environment.
A Trial that fails (an idle probe, 3 failed turns in a row, a crash) is reported as `error`
and `exited`; the next prompt starts a new one.

The session files live in `home/sessions` (FlockPi copies them there after each turn), where
BucketSync pushes and restores them. Files in the container do not survive a new Trial.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import Any

from flock.models import TypeSpec, parse_duration
from flock.worker.adapters.base import AdapterEvent, EventQueue, ResumeError

log = logging.getLogger(__name__)


class HarborAdapter:
    harness = "harbor"

    def __init__(self, workdir: Path, name: str, spec: TypeSpec, *, resources: Any = None,
                 bundle: Any = None, host: Any = None) -> None:
        self.workdir, self.agent, self.spec = Path(workdir), name, spec
        self.root = self.workdir.parent / "_harbor" / name  # trials/<name>-tN, state/sources
        self.home = self.root / "pi"
        self.resources = resources  # the worker's flock.harbor Resources (bind); None: none
        self.bundle = bundle  # the run's ToolBundle; None: no tools
        self.host = host  # the worker's HarborHost: resources, bundle, task (flock.worker.harbor)
        self.session_id: str | None = None
        self.attempt = 0  # Trials started for this agent here
        self.trial: Any = None
        self.result: Any = None  # the last ended Trial's TrialResult
        self._run: asyncio.Task | None = None
        self._starting: asyncio.Task | None = None  # a Trial started by send() or a rollover
        self._creating = False  # a Trial is being created (no `_run` for it yet)
        self._prompts: asyncio.Queue = asyncio.Queue()
        self._queue = EventQueue()  # one stream; each burst reads it up to its `exited`
        self._open = False  # a burst is reading events (an `exited` is owed)
        self._closing = False

    def session_files(self, session_id: str | None) -> list[Path]:
        """Every Pi session file of the agent: one Pi session dir per agent, and `--continue`
        takes up the newest."""
        return sorted((self.home / "sessions").glob("*.jsonl"))

    def alive(self) -> bool:
        return self._run is not None and not self._run.done()

    def trial_status(self) -> dict[str, Any] | None:
        """The heartbeat's `trial`: the agent's latest Trial here, if any."""
        if self.trial is None:
            return None
        return {"id": self.trial.config.trial_name, "attempt": self.attempt, "alive": self.alive()}

    # ------------------------------------------------------------------ adapter
    async def start(self, workdir: Path, session_id: str | None, spec: TypeSpec, *,
                    env: dict[str, str] | None = None) -> str:
        """`env` is not passed on: a Trial's environment is the task's, plus the inference key,
        the tools and the resources, as shepherd sets them."""
        self.spec = spec
        files = self.session_files(session_id)
        if session_id and not files:
            raise ResumeError(f"no Pi session files under {self.home / 'sessions'}")
        if self._starting is not None and not self._starting.done():
            await asyncio.shield(self._starting)  # its failure is reported in this burst
        if not self.alive():
            await self._new_trial(restore=bool(files))
        self._open = True
        self.session_id = session_id or self.session_id or str(uuid.uuid4())
        return self.session_id

    async def send(self, text: str) -> None:
        self._prompts.put_nowait(text)
        self._ensure()

    async def stop(self) -> None:
        """Kill a turn in flight (reported `interrupted`), drop queued prompts and end the
        burst's event stream. The Trial stays up. Idempotent."""
        while not self._prompts.empty():
            self._prompts.get_nowait()
        agent = getattr(self.trial, "agent", None)
        if agent is not None and getattr(agent, "busy", False):
            await agent.kill()
        self._exit()

    async def close(self) -> Any:
        """End the Trial (Harbor runs the verifier); returns its TrialResult, None if none."""
        if self._starting is not None and not self._starting.done():
            self._starting.cancel()
            with suppress(asyncio.CancelledError):
                await self._starting
        if self.alive():
            self._closing = True
            self._prompts.put_nowait(None)
            try:
                await asyncio.shield(self._run)
            finally:
                self._closing = False
                self._prompts = asyncio.Queue()
        return self.result

    async def events(self) -> AsyncIterator[AdapterEvent]:
        async for ev in self._queue:
            yield ev

    # ------------------------------------------------------------------ Trials
    def _emit(self, ev: AdapterEvent) -> None:
        self._queue.put(ev)  # between bursts it waits for the next one (a failure, usage)

    def _exit(self) -> None:
        if self._open:
            self._open = False
            self._emit(AdapterEvent("exited"))

    def _ensure(self) -> None:
        """A Trial for the queued prompts, unless one is up or starting."""
        if self.alive() or self._creating or self._closing or self._prompts.empty():
            return
        self._creating = True  # until the task has the new Trial's `_run`
        self._starting = asyncio.create_task(self._restart(), name=f"trial-{self.agent}")

    async def _restart(self) -> None:
        try:
            await self._new_trial(restore=bool(self.session_files(None)))
        except Exception as e:  # noqa: BLE001 - reported; the next start tries again
            log.warning("%s: new Trial failed: %s", self.agent, e)
            self._emit(AdapterEvent("error", error=f"new Trial failed: {e}"))
            self._exit()
        finally:
            self._creating = False

    def _instructions(self) -> str:
        from flock.harbor.instructions import build_agent_instructions

        h = self.spec.harbor
        roster = h.roster if self.agent in h.roster else {**h.roster, self.agent: h.role}
        return build_agent_instructions(
            self.agent, roster, {"tools": h.tools.get("tools", {})}, h.team, h.role_text)

    async def _new_trial(self, restore: bool) -> None:
        """Create the agent's next Trial, run it in the background and wait until FlockPi
        is ready. ResumeError: the sessions could not be put back."""
        from harbor.models.task.id import GitTaskId

        from flock.harbor import launch
        from flock.harbor.agent import Channel
        from flock.harbor.sources import ResolvedTask
        from flock.harbor.spec import Spec

        h = self.spec.harbor
        if h is None:
            raise RuntimeError("harness harbor needs the type's `harbor:` block")
        if self.host is not None:
            await self.host.prepare(self.spec)
            self.resources, self.bundle = self.host.resources, self.host.bundle
        if h.tools.get("tools") and self.bundle is None:
            raise RuntimeError(f"the tool bundle {h.tools_sha256} is not on this worker")
        config = Spec.model_validate(h.config)
        task = ResolvedTask(GitTaskId.model_validate(h.task["task"]), h.task["registry_commit"],
                            h.task["dataset"], getattr(self.host, "task_path", None))
        trial_seconds = config.limits.trial_hours * 3600
        self.attempt += 1
        tc = launch.build_trial_config(config, task, self.root, self.agent, self._instructions())
        tc.trial_name = f"{self.agent}-t{self.attempt}"
        tc.agent.name, tc.agent.import_path = None, "flock.harbor.agent:FlockPi"
        # FlockPi serves turns for trial_hours; its last turn may take a turn_timeout more
        tc.agent.override_timeout_sec = (
            trial_seconds + parse_duration(self.spec.turn_timeout).total_seconds())
        tc.environment.kwargs = dict(config.environment.kwargs)
        if self.resources is not None:
            self.resources.bind(self.agent, tc)
            if tc.environment.mounts and config.environment.backend == "hf-sandbox":
                raise RuntimeError("hf-sandbox has no mounts: use the resources' http transport")
        self._creating = True
        try:
            trial = await launch.Trial.create(tc)
            launch.attach_pi_setup(trial, config.inference)
            if self.bundle is not None:
                launch.attach_tool_setup(trial, self.bundle)
            channel = Channel(emit=self._emit, prompts=self._prompts, home=self.home,
                              model=self.spec.harness_model, trial_seconds=trial_seconds,
                              max_turns=config.limits.agent_max_turns, restore=restore)
            trial.agent.channel = channel
            self.trial = trial
            self._run = run = asyncio.create_task(trial.run(), name=tc.trial_name)
            run.add_done_callback(lambda task: self._ended(trial, task))
        finally:
            self._creating = False
        ready = asyncio.create_task(channel.ready.wait())
        try:
            await asyncio.wait({ready, run}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            run.cancel()
            raise
        finally:
            ready.cancel()
        if not channel.ready.is_set():
            why = channel.restore_error or _why(trial, run)
            raise (ResumeError if channel.restore_error else RuntimeError)(
                f"Trial {tc.trial_name} did not start: {why}")

    def _ended(self, trial: Any, task: asyncio.Task) -> None:
        """Done callback of a Trial's run(): roll over after trial_hours, report a failure
        (now, or at the next burst)."""
        self.result = _result(task)
        if trial is not self.trial or not trial.agent.channel.ready.is_set():
            return  # superseded, or never ready: whoever started it reports that
        why = _why(trial, task)
        log.info("%s: Trial %s ended: %s", self.agent, trial.config.trial_name, why)
        if why == "closed" or self._closing:
            return
        if why == "trial_hours":
            self._ensure()  # prompts queued as it ended go to the next Trial
            return
        self._emit(AdapterEvent("error", error=f"Trial {trial.config.trial_name} ended: {why}"))
        self._exit()


def _result(task: asyncio.Task) -> Any:
    return None if task.cancelled() or task.exception() else task.result()


def _why(trial: Any, task: asyncio.Task) -> str:
    """Why a Trial's run() ended: FlockPi's reason, else Harbor's error."""
    if task.cancelled():
        return "cancelled"
    if task.exception() is not None:
        return f"{type(task.exception()).__name__}: {task.exception()}"
    info = task.result().exception_info
    return (getattr(trial.agent, "why", None)
            or (info.exception_message if info is not None else None) or "ended")


def make_adapter(workdir: Path, name: str, spec: TypeSpec, *, host: Any = None) -> HarborAdapter:
    """The worker's factory for harness `harbor` (flock.registry); `host` is the worker's
    HarborHost (its Resources, tool bundle and task download)."""
    return HarborAdapter(workdir, name, spec, host=host)
