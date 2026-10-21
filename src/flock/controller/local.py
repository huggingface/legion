"""`compute: local`: workers as subprocesses on the controller's machine (docs/local.md).

`LocalJobsClient` speaks the same `JobsClient` protocol as `HubJobsClient`, so the reconcile
loop, placement, heartbeats, admission and events are unchanged: a "job" is a
`python -m flock.worker.main` process with the job env contract (FLOCK_URL, FLOCK_TOKEN,
WORKER_ID, WORKER_INCARNATION, SWARM, TYPE, DEADLINE, ...). What a job has and a process does
not (image, flavor, network group, wheel bootstrap) is ignored, and so are the spec's namespace
and bucket: agent state stays on local disk.

Stage mapping: running process -> RUNNING; exit 0 -> COMPLETED; other exit -> ERROR
("exit N"); alive past its timeout -> killed, ERROR "Job timeout" (what HF Jobs reports, so a
lifetime restart is a planned restart here too); cancelled -> CANCELED.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from flock.controller.jobs import JobListing, JobState
from flock.models import TypeSpec

log = logging.getLogger("flock.local")

WORKER_COMMAND = [sys.executable, "-m", "flock.worker.main"]
# Defaults a local worker gets unless the type's env sets them: a laptop run wants agent states
# within seconds, not the 30 s job heartbeat.
LOCAL_ENV_DEFAULTS = {"HEARTBEAT_INTERVAL": "5", "PYTHONUNBUFFERED": "1"}
# The caller's own flock config never leaks into a worker (it gets the controller's).
_DROP_PREFIXES = ("FLOCK_",)
STOP_GRACE = 5.0  # s between SIGTERM and SIGKILL on cancel


@dataclass
class _Proc:
    job_id: str
    worker_id: str
    popen: subprocess.Popen
    labels: dict[str, str]
    env: dict[str, str]
    started: float
    timeout: float
    log_file: Path
    log_handle: IO[bytes]
    canceled: bool = False
    timed_out: bool = False


@dataclass
class LocalJobsClient:
    """Launch, inspect and cancel worker subprocesses. `root` holds `agents/` (WORK_ROOT,
    shared by all workers so a restarted slot finds its agents' workdirs), `state/llm`
    (FLOCK_LLM_HOME) and `logs/<worker>.log` (stdout and stderr of each process)."""

    root: Path
    command: list[str] = field(default_factory=lambda: list(WORKER_COMMAND))
    base_env: dict[str, str] | None = None  # default: os.environ minus FLOCK_*
    procs: dict[str, _Proc] = field(default_factory=dict)
    calls: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self._lock = threading.Lock()
        self._n = 0

    @property
    def work_root(self) -> Path:
        return self.root / "agents"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    def process_env(self, env: dict[str, str], secrets: dict[str, str]) -> dict[str, str]:
        base = dict(os.environ if self.base_env is None else self.base_env)
        base = {k: v for k, v in base.items() if not k.startswith(_DROP_PREFIXES)}
        local = {
            "WORK_ROOT": str(self.work_root),
            "FLOCK_LLM_HOME": str(self.root / "state" / "llm"),
            # The worker exits when the controller process is gone (no orphans after a kill -9).
            "FLOCK_LOCAL_PARENT_PID": str(os.getpid()),
        }
        # secrets arrive as env, as on a job; the type's env can override the local defaults.
        # The spec's bucket is for hf-jobs: locally agent state stays on disk under `root`, so
        # a spec keeps its namespace and bucket and `compute:` is the only line that changes.
        return {**base, **LOCAL_ENV_DEFAULTS, **local, **env, **secrets, "BUCKET": ""}

    def launch(self, worker_id: str, spec: TypeSpec, env: dict[str, str],
               secrets: dict[str, str], labels: dict[str, str], timeout: float, *,
               namespace: str) -> str:
        self.calls["launch"] += 1
        with self._lock:
            self._n += 1
            job_id = f"local-{self._n:04d}"
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"{worker_id}.log"
        handle = log_file.open("ab")
        handle.write(f"--- {job_id} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
        handle.flush()
        try:
            popen = subprocess.Popen(
                self.command, env=self.process_env(env, secrets), stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, cwd=str(self.root),
                start_new_session=True,  # Ctrl-C in the terminal reaches `flock run`, not workers
            )
        except Exception:
            handle.close()
            raise
        with self._lock:
            self.procs[job_id] = _Proc(job_id, worker_id, popen, dict(labels), dict(env),
                                       time.monotonic(), float(timeout), log_file, handle)
        log.info("launched %s for %s (pid %d)", job_id, worker_id, popen.pid)
        return job_id

    def inspect(self, job_id: str, *, namespace: str) -> JobState:
        self.calls["inspect"] += 1
        p = self.procs.get(job_id)
        if p is None:
            return JobState("DELETED")
        rc = p.popen.poll()
        if rc is None and time.monotonic() - p.started > p.timeout:
            p.timed_out = True
            self._kill(p)
            rc = p.popen.poll()
        if rc is None:
            return JobState("RUNNING")
        self._closed(p)
        if p.timed_out:
            return JobState("ERROR", "Job timeout")
        if p.canceled:
            return JobState("CANCELED")
        return JobState("COMPLETED") if rc == 0 else JobState("ERROR", f"exit {rc}")

    def cancel(self, job_id: str, *, namespace: str) -> None:
        self.calls["cancel"] += 1
        p = self.procs.get(job_id)
        if p is None or p.popen.poll() is not None:
            return
        p.canceled = True
        self._signal(p, signal.SIGTERM)
        threading.Thread(target=self._reap, args=(p, STOP_GRACE), daemon=True).start()

    def list(self, labels: dict[str, str], *, namespace: str) -> list[JobListing]:
        self.calls["list"] += 1
        out = []
        for p in list(self.procs.values()):
            if all(p.labels.get(k) == v for k, v in labels.items()):
                st = self.inspect(p.job_id, namespace=namespace)
                out.append(JobListing(p.job_id, p.labels, st.stage, p.env, st.message))
        return out

    def logs_tail(self, job_id: str, *, namespace: str, n: int = 5) -> list[str]:
        p = self.procs.get(job_id)
        if p is None or not p.log_file.is_file():
            return []
        lines = p.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:]

    # ------------------------------------------------------------------ shutdown helpers

    def alive(self) -> list[str]:
        return [j for j, p in self.procs.items() if p.popen.poll() is None]

    def wait_all(self, timeout: float) -> bool:
        """Wait up to `timeout` s for every process to exit; True if none is left."""
        end = time.monotonic() + timeout
        while self.alive() and time.monotonic() < end:
            time.sleep(0.1)
        return not self.alive()

    def kill_all(self, grace: float = STOP_GRACE) -> None:
        """SIGTERM every live process (and its children), SIGKILL what is left after `grace`."""
        live = [p for p in self.procs.values() if p.popen.poll() is None]
        for p in live:
            p.canceled = True
            self._signal(p, signal.SIGTERM)
        end = time.monotonic() + grace
        for p in live:
            try:
                p.popen.wait(max(0.0, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                self._signal(p, signal.SIGKILL)
                p.popen.wait()
        for p in self.procs.values():
            self._closed(p)

    def _reap(self, p: _Proc, grace: float) -> None:
        try:
            p.popen.wait(grace)
        except subprocess.TimeoutExpired:
            self._kill(p)

    def _kill(self, p: _Proc) -> None:
        self._signal(p, signal.SIGKILL)
        try:
            p.popen.wait(5)
        except subprocess.TimeoutExpired:
            log.warning("%s (pid %d) did not exit after SIGKILL", p.job_id, p.popen.pid)

    @staticmethod
    def _signal(p: _Proc, sig: signal.Signals) -> None:
        try:  # the process group: harness children (claude, codex, pi) go with the worker
            os.killpg(p.popen.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def _closed(p: _Proc) -> None:
        if not p.log_handle.closed:
            p.log_handle.close()
