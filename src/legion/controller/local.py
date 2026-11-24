"""`compute: local`: workers as subprocesses on the controller's machine (docs/local.md).

`LocalJobsClient` speaks the same `JobsClient` protocol as `HubJobsClient`, so the reconcile
loop, placement, heartbeats, admission and events are unchanged: a "job" is a
`python -m legion.worker.main` process with the job env contract (LEGION_URL, LEGION_TOKEN,
WORKER_ID, WORKER_INCARNATION, SWARM, TYPE, DEADLINE, ...). What a job has and a process does
not (image, flavor, network group, wheel bootstrap) is ignored, and so are the spec's namespace
and bucket: agent state stays on local disk.

Stage mapping: running process -> RUNNING; exit 0 -> COMPLETED; other exit -> ERROR
("exit N"); alive past its timeout -> killed, ERROR "Job timeout" (what HF Jobs reports, so a
lifetime restart is a planned restart here too); cancelled -> CANCELED.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from legion.controller.jobs import JobListing, JobState
from legion.models import LOCAL_HEARTBEAT_INTERVAL, TypeSpec

log = logging.getLogger("legion.local")

WORKER_COMMAND = [sys.executable, "-m", "legion.worker.main"]
# Defaults a local worker gets unless the type's env sets them: a laptop run wants agent states
# within seconds, not the 30 s job heartbeat.
LOCAL_ENV_DEFAULTS = {"HEARTBEAT_INTERVAL": f"{LOCAL_HEARTBEAT_INTERVAL.total_seconds():g}",
                      "PYTHONUNBUFFERED": "1"}
# The caller's own legion connection and auth never leak into a worker (it gets the
# controller's): LEGION_* is dropped except these documented harness settings.
KEEP_LEGION_ENV = frozenset({"LEGION_PROMPTS_DIR"})
STOP_GRACE = 5.0  # s between SIGTERM and SIGKILL on cancel
# s a harbor worker gets instead, also at its job timeout: on SIGTERM it cancels its Trials and
# waits up to 60 s (adapters/harbor.py CANCEL_WAIT) for their environments to stop; a SIGKILL
# before that leaves their containers running
HARBOR_STOP_GRACE = 90.0


def _children_map() -> dict[int, list[int]]:
    """ppid -> pids of every process we can see (/proc on Linux, else `ps`)."""
    out: dict[int, list[int]] = {}
    proc = Path("/proc")
    if proc.is_dir():
        for d in proc.iterdir():
            if not d.name.isdigit():
                continue
            try:
                stat = (d / "stat").read_text()
            except OSError:
                continue
            fields = stat[stat.rfind(")") + 2:].split()  # after "pid (comm) "
            if len(fields) > 1 and fields[0] != "Z":
                out.setdefault(int(fields[1]), []).append(int(d.name))
        return out
    try:
        ps = subprocess.run(["ps", "-A", "-o", "pid=", "-o", "ppid="], capture_output=True,
                            text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return out
    for line in ps.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            out.setdefault(int(parts[1]), []).append(int(parts[0]))
    return out


def descendants(pid: int) -> list[int]:
    """Every live descendant of `pid`, also those in their own session (outside the worker's
    process group)."""
    kids, found, todo = _children_map(), [], [pid]
    while todo:
        for child in kids.get(todo.pop(), []):
            if child not in found:
                found.append(child)
                todo.append(child)
    return found


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie is dead for our purposes
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rfind(")") + 2:].split()[0] != "Z"
    except (OSError, IndexError):
        return True


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
    kill_grace: float = 0.0  # s at least between SIGTERM and SIGKILL (HARBOR_STOP_GRACE)


@dataclass
class LocalJobsClient:
    """Launch, inspect and cancel worker subprocesses. `root` holds `agents/` (WORK_ROOT,
    shared by all workers so a restarted slot finds its agents' workdirs) and
    `logs/<worker>.log` (stdout and stderr of each process)."""

    root: Path
    command: list[str] = field(default_factory=lambda: list(WORKER_COMMAND))
    base_env: dict[str, str] | None = None  # default: os.environ minus LEGION_*
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
        base = {k: v for k, v in base.items()
                if not k.startswith("LEGION_") or k in KEEP_LEGION_ENV}
        local = {
            "WORK_ROOT": str(self.work_root),
            # The worker exits when the controller process is gone (no orphans after a kill -9).
            "LEGION_LOCAL_PARENT_PID": str(os.getpid()),
        }
        # secrets arrive as env, as on a job; the type's env can override the local defaults.
        # The spec's bucket is for hf-jobs: locally agent state stays on disk under `root`, so
        # a spec keeps its namespace and bucket and `compute:` is the only line that changes.
        return {**base, **LOCAL_ENV_DEFAULTS, **local, **env, **secrets, "BUCKET_ID": ""}

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
                start_new_session=True,  # Ctrl-C in the terminal reaches the local run, not workers
            )
        except Exception:
            handle.close()
            raise
        with self._lock:
            self.procs[job_id] = _Proc(job_id, worker_id, popen, dict(labels), dict(env),
                                       time.monotonic(), float(timeout), log_file, handle,
                                       kill_grace=HARBOR_STOP_GRACE)
        log.info("launched %s for %s (pid %d)", job_id, worker_id, popen.pid)
        return job_id

    def inspect(self, job_id: str, *, namespace: str) -> JobState:
        self.calls["inspect"] += 1
        p = self.procs.get(job_id)
        if p is None:
            return JobState("DELETED")
        rc = p.popen.poll()
        if rc is None and not p.timed_out and time.monotonic() - p.started > p.timeout:
            p.timed_out = True
            if p.kill_grace:  # RUNNING until its Trials are stopped (or the grace is over)
                threading.Thread(target=self._stop, args=([p], 0), daemon=True).start()
            else:
                self._stop([p], grace=0)
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
        threading.Thread(target=self._stop, args=([p], STOP_GRACE), daemon=True).start()

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
        """Stop every live process; see `_stop`."""
        self._stop([p for p in self.procs.values() if p.popen.poll() is None], grace)
        for p in self.procs.values():
            self._closed(p)

    def _stop(self, procs: list[_Proc], grace: float) -> None:
        """SIGTERM each worker's process group (a local worker then stops its agents, and the
        adapters kill their tools), SIGKILL the group after `grace` (at least its
        `kill_grace`), then SIGKILL any descendant still alive: a worker's detached tool
        processes (own session) are recorded before the worker can exit and leave them
        reparented."""
        tree = {p.job_id: descendants(p.popen.pid) for p in procs}
        for p in procs:
            p.canceled = p.canceled or not p.timed_out
            self._signal(p, signal.SIGTERM)
        start = time.monotonic()
        for p in procs:
            try:
                p.popen.wait(max(0.0, start + max(grace, p.kill_grace) - time.monotonic()))
            except subprocess.TimeoutExpired:
                tree[p.job_id] += [d for d in descendants(p.popen.pid) if d not in tree[p.job_id]]
                self._signal(p, signal.SIGKILL)
                try:
                    p.popen.wait(5)
                except subprocess.TimeoutExpired:
                    log.warning("%s (pid %d) did not exit after SIGKILL", p.job_id, p.popen.pid)
        own = os.getpgrp()
        for pids in tree.values():
            for pid in pids:
                if not _alive(pid):
                    continue
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    if os.getpgid(pid) == pid and pid != own:
                        os.killpg(pid, signal.SIGKILL)  # a session leader: its group too
                    os.kill(pid, signal.SIGKILL)
                log.info("killed leftover process %d", pid)

    @staticmethod
    def _signal(p: _Proc, sig: signal.Signals) -> None:
        try:  # the process group: harness children (pi, tools) go with the worker
            os.killpg(p.popen.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def _closed(p: _Proc) -> None:
        if not p.log_handle.closed:
            p.log_handle.close()


def local_jobs_client(env: Mapping[str, str], dep: Any, data_dir: str) -> LocalJobsClient:
    """The controller's client for compute `local` (legion.registry)."""
    if dep.mode == "job":
        raise SystemExit("LEGION_COMPUTE=local cannot run with LEGION_MODE=job")
    return LocalJobsClient(root=data_dir)
