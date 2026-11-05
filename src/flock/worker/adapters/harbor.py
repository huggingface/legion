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
- `stop` kills a turn in flight and ends the burst's event stream; the Trial stays up. When
  /logs/agent is not mounted (hf-sandbox), it also downloads the sessions for the next push.
- `close` ends the Trial: FlockPi returns, Harbor runs the verifier and stops the environment;
  `close(0)` cancels it unverified, and a Trial still starting is cancelled.
On docker and podman each Trial gets its own network, as shepherd's launch gives each
participant one (TrialNetworks), removed when the next Trial starts or at `close`.
A Trial that fails (an idle probe, 3 failed turns in a row, a crash) is reported as `error`
and `exited`, after a failed turn for any prompt queued for it; the next prompt starts a new
one. When its first turn on restored sessions failed, the next `start` of that session raises
ResumeError (the runtime then starts a fresh session).

The session files live in `home/sessions` (FlockPi copies them there after each turn and the
adapter again when a Trial ends), where BucketSync pushes and restores them. Files in the
container do not survive a new Trial.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from flock.models import TypeSpec, parse_duration
from flock.worker.adapters.base import AdapterEvent, EventQueue, ResumeError

log = logging.getLogger(__name__)

CANCEL_WAIT = 60.0  # s close() waits for a cancelled Trial's environment to stop


class HarborAdapter:
    harness = "harbor"

    def __init__(self, workdir: Path, name: str, spec: TypeSpec, *, resources: Any = None,
                 bundle: Any = None, host: Any = None) -> None:
        self.workdir, self.agent, self.spec = Path(workdir), name, spec
        self.root = self.workdir.parent / "_harbor" / name  # trials/<name>-tN, state/sources
        self.home = self.root / "pi"
        self.session_dir = self.root  # what purge() deletes for an ephemeral agent
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
        self._resume_failed: str | None = None  # the last Trial's --continue failed: why
        self._closes = 0  # close() calls: a start that saw one begin does not run its Trial
        self._network: Any = None  # the latest Trial's TrialNetworks (docker, podman)
        self._network_id = uuid.uuid4().hex[:8]  # names and labels this adapter's networks

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
        try:
            if session_id and not files:
                raise ResumeError(f"no Pi session files under {self.home / 'sessions'}")
            if session_id and self._resume_failed:
                why, self._resume_failed = self._resume_failed, None
                raise ResumeError(why)
            if self._starting is not None and not self._starting.done():
                await asyncio.shield(self._starting)  # its failure is reported in this burst
            if not self.alive():
                await self._new_trial(restore=bool(session_id and files))
        except ResumeError:
            self.session_id = None  # the runtime's fresh start gets a new id
            raise
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
        # unmounted: the next push gets this burst's sessions (a closing Trial's come with
        # Harbor's download as it ends)
        if self._open and self.alive() and not self._closing and agent.channel.ready.is_set():
            await agent.pull_sessions()
        self._exit()

    async def close(self, timeout: float | None = None) -> Any:
        """End the Trial: FlockPi returns, Harbor runs the verifier and stops the environment;
        returns its TrialResult (None if none). After `timeout` s (0: at once; None: no limit),
        or when this call is cancelled, the Trial is cancelled instead: Harbor stops the
        environment unverified, waited for up to CANCEL_WAIT s (a further cancel stops the
        wait). A Trial still starting is cancelled at once; a start that has not created its
        Trial yet fails."""
        self._closes += 1
        starting = self._starting
        if starting is not None and not starting.done():
            starting.cancel()
            await asyncio.wait({starting})
        run = self._run
        if run is not None and not run.done():
            self._closing = True
            try:
                if timeout != 0 and self.trial.agent.channel.ready.is_set():
                    self._prompts.put_nowait(None)
                    await asyncio.wait({run}, timeout=timeout)
            finally:
                try:
                    if not run.done():
                        if not run.cancelling():  # a second cancel would cut Harbor's cleanup
                            run.cancel()
                        await asyncio.wait({run}, timeout=CANCEL_WAIT)
                finally:
                    self._closing = False
                    self._prompts = asyncio.Queue()
        if run is None or run.done():
            await self._drop_network()
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
        """A Trial for the queued prompts, unless one is up or starting (or the session must
        restart: the runtime's next start gets the ResumeError)."""
        if (self.alive() or self._creating or self._closing or self._resume_failed
                or self._prompts.empty()):
            return
        self._creating = True  # until the task has the new Trial's `_run`
        self._starting = asyncio.create_task(self._restart(), name=f"trial-{self.agent}")

    async def _restart(self) -> None:
        try:
            await self._new_trial(restore=bool(self.session_id and self.session_files(None)))
        except Exception as e:  # noqa: BLE001 - reported; the next start tries again
            log.warning("%s: new Trial failed: %s", self.agent, e)
            self._fail(f"new Trial failed: {e}")
        finally:
            self._creating = False

    def _fail(self, error: str) -> None:
        """A Trial failed or did not start: prompts queued for it end as a failed turn (the
        runtime's loop sends them again), then `error` and `exited`."""
        queued = False
        while not self._prompts.empty():
            queued = self._prompts.get_nowait() is not None or queued
        if queued and self._open:
            self._emit(AdapterEvent("turn_started"))
            self._emit(AdapterEvent("turn_ended", status="error", error=error))
        self._emit(AdapterEvent("error", error=error))
        self._exit()

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
        closes = self._closes
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
        # shortcut: build_trial_config and the board resource read the inference key and the
        # board token from os.environ, which a Slurm worker lacks (they are in its secrets file,
        # secret_env()); read them through secret_env() with the compute commit
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
        await self._drop_network()  # the last Trial's: it has ended
        if config.environment.backend != "hf-sandbox":  # a sandbox is an HF Job: no local network
            # one network per Trial, outside host and VPN routes: docker's default pools run
            # out after about 30 compose projects on a host
            self._network = networks = launch.TrialNetworks(
                self._network_id, engine=config.environment.backend)
            await networks.start()
            self.root.mkdir(parents=True, exist_ok=True)
            tc.environment.extra_docker_compose = [await networks.create(
                tc.trial_name, self.root / f"{tc.trial_name}-network.json")]
        self._creating = True
        try:
            trial = await launch.Trial.create(tc)
            if self._closes != closes:  # closed while this Trial was being created
                await self._drop_network()
                raise RuntimeError(f"Trial {tc.trial_name} not started: the agent was closed")
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

    async def _drop_network(self) -> None:
        """Remove the latest Trial's network once its environment has stopped."""
        networks, self._network = self._network, None
        if networks is None:
            return
        try:
            await networks.stop()
        except Exception as e:  # noqa: BLE001 - a leftover network holds a subnet, not the run
            log.warning("%s: Trial network not removed: %s", self.agent, e)

    def _ended(self, trial: Any, task: asyncio.Task) -> None:
        """Done callback of a Trial's run(): roll over after trial_hours, report a failure
        (now, or at the next burst)."""
        self.result = _result(task)
        if trial is not self.trial or not trial.agent.channel.ready.is_set():
            return  # superseded, or never ready: whoever started it reports that
        try:  # unmounted (hf-sandbox): Harbor's download at the end beats FlockPi's last one
            trial.agent.keep_sessions()
        except OSError as e:
            log.warning("%s: sessions of Trial %s not kept: %s", self.agent,
                        trial.config.trial_name, e)
        why = _why(trial, task)
        log.info("%s: Trial %s ended: %s", self.agent, trial.config.trial_name, why)
        if why == "closed" or self._closing:
            return
        if why == "trial_hours":
            self._ensure()  # prompts queued as it ended go to the next Trial
            return
        self._resume_failed = trial.agent.channel.restore_error
        self._fail(f"Trial {trial.config.trial_name} ended: {why}")


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
