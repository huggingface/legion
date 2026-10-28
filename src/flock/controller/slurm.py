"""`compute: slurm` and `compute: external`: workers as Slurm jobs, or started by someone else.

`SlurmJobsClient` speaks the `JobsClient` protocol (controller/jobs.py), so the reconcile loop,
placement, heartbeats and events are unchanged. flock only shells out:

    sbatch --parsable -p <partition> [-A <account>] -c <cpus> --mem=<mem> --time=<time>
           --no-requeue -J <worker id> --comment=<labels> -o <log_dir>/<worker id>-%j.out
           <worker_script> <launch_dir>/<worker id>-<incarnation>.env

The script is the cluster side's (experiments/run3/slurm/worker.sbatch is a reference): it
sources the cluster's secrets file and the env file, traps USR1/TERM and runs `flock-worker`.
`--export` is never passed (the job inherits the submitting environment, Slurm's default) and
neither is `--signal`: the cluster's node watcher sends SIGUSR1 about 30 s before it cancels a
reclaimed job. `--no-requeue`: a worker restarts through the controller (fresh incarnation).
The env file name carries the incarnation, so a job that starts late never reads the settings
of the job that replaced it.

The env file (`KEY=value` lines, mode 600) holds NO secret values: only settings (WORKER_ID,
WORKER_INCARNATION, SWARM, TYPE, DEADLINE, DRAIN_WINDOW, BUCKET, RUN, WORK_ROOT, the controller
URL template and host file, the type's env) and a comment that names the secrets the worker
expects from the cluster's secrets file (FLOCK_TOKEN, HF_TOKEN, FLOCK_WORLD_TOKEN, ...). FLOCK_URL is `http://${CTL_HOST}:<port>` when
`host_file` is set: the worker reads the controller's hostname from that file at start and
again when the controller stops answering (worker/main.py), so a controller that moves host
does not relaunch the fleet.

State mapping (squeue first; sacct once the job left squeue):
  PENDING, CONFIGURING, REQUEUE_HOLD, SUSPENDED   -> SCHEDULING (autoscaling from 0 can take
                                                     ~15 min: normal, not a failure)
  RUNNING, COMPLETING, SIGNALING, STAGE_OUT        -> RUNNING
  COMPLETED                                        -> COMPLETED
  TIMEOUT                                          -> ERROR "Job timeout" (planned restart)
  CANCELLED that flock asked for                   -> CANCELED
  CANCELLED that flock did not ask for (a Spot
  reclaim: the node watcher cancels the job), PREEMPTED, REQUEUED, NODE_FAIL,
  exit code PREEMPT_EXIT                           -> ERROR "preempted: ..." with
                                                     `preempted=True` (planned, no backoff)
  FAILED, OUT_OF_MEMORY, BOOT_FAIL, DEADLINE       -> ERROR "<STATE> exit <code>" (a crash)

`ExternalJobsClient` launches nothing: `launch` writes the same env file (when `launch_dir` is
set), logs what it expects to be started and returns an `external-...` id; `inspect` always
says SCHEDULING, so the reconcile loop waits for heartbeats and never counts a missing
external worker as a crash (reconcile.py).
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from flock.controller.jobs import JobListing, JobState
from flock.models import ComputeSpec, TypeSpec, mem_mb, size_problem  # noqa: F401

log = logging.getLogger("flock.slurm")

# The worker's exit code after a preemption drain (worker/main.py): EX_TEMPFAIL.
PREEMPT_EXIT = 75
DEFAULTS = {"partition": "hopper-cpu", "cpus": "4", "mem": "7G", "time": "29:00:00",
            "max_cpus": "28", "max_mem": "54G"}
# compute spec key -> controller env override (the first seven are the cluster's contract)
ENV_KEYS = {
    "partition": "FLOCK_SLURM_PARTITION",
    "cpus": "FLOCK_SLURM_CPUS",
    "mem": "FLOCK_SLURM_MEM",
    "time": "FLOCK_SLURM_TIME",
    "log_dir": "FLOCK_SLURM_LOG_DIR",
    "worker_script": "FLOCK_SLURM_WORKER_SCRIPT",
    "launch_dir": "FLOCK_SLURM_LAUNCH_DIR",
    "account": "FLOCK_SLURM_ACCOUNT",
    "host_file": "FLOCK_SLURM_HOST_FILE",
    "run": "FLOCK_SLURM_RUN",
    "venv": "FLOCK_SLURM_VENV",
    "secrets_file": "FLOCK_SLURM_SECRETS_FILE",
    "max_cpus": "FLOCK_SLURM_MAX_CPUS",
    "max_mem": "FLOCK_SLURM_MAX_MEM",
    "container_root": "FLOCK_SLURM_CONTAINER_ROOT",
    "container_image": "FLOCK_SLURM_CONTAINER_IMAGE",
}
# The only variables sbatch (and squeue/sacct/scancel) get from the controller: without
# --export the job inherits sbatch's environment and Slurm may store it in its accounting
# (sacct --env-vars), so no token or key may be in it.
KEEP_ENV = frozenset({"PATH", "HOME", "USER", "LOGNAME", "LANG", "LANGUAGE", "TZ", "TMPDIR",
                      "TERM", "SHELL"})
KEEP_PREFIXES = ("LC_", "SLURM_")
# Secrets every Slurm worker reads from the secrets file besides those the launch names:
# HF_TOKEN for the bucket sync (HfApi() reads it from the env).
ALWAYS_SECRETS = ("FLOCK_TOKEN", "HF_TOKEN")
_SAFE = re.compile(r"^[A-Za-z0-9_@%+=:,./-]*$")
# Fixed per slot (restores go to the same absolute paths): node-local, wiped at job start.
WORK_BASE = "/tmp/flock"
NOT_IN_SACCT_GRACE = 120.0  # s after sbatch during which an unknown job is "still SCHEDULING"
CMD_TIMEOUT = 60.0  # s per sbatch/squeue/sacct/scancel call

SCHEDULING = {"PENDING", "CONFIGURING", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD",
              "SUSPENDED", "RESIZING", "STOPPED"}
RUNNING = {"RUNNING", "COMPLETING", "SIGNALING", "STAGE_OUT", "SPECIAL_EXIT"}
PREEMPTED = {"PREEMPTED", "REQUEUED", "NODE_FAIL"}
_JOB_ID = re.compile(r"^\d+(_\d+)?$")

Runner = Callable[[list[str]], subprocess.CompletedProcess]


class SlurmConfigError(ValueError):
    """A launch that cannot work until the spec changes: the type is marked launch_refused."""

    launch_refused = True


def scrubbed_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The controller's environment minus everything but KEEP_ENV and LC_*/SLURM_*."""
    env = os.environ if env is None else env
    return {k: v for k, v in env.items() if k in KEEP_ENV or k.startswith(KEEP_PREFIXES)}


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=CMD_TIMEOUT, check=False,
                          env=scrubbed_env())




def resolve(compute: ComputeSpec | None, type_name: str,
            env: Mapping[str, str] = os.environ) -> dict[str, str]:
    """Settings for one worker launch: defaults < spec keys < FLOCK_SLURM_* env < the spec's
    `types.<type>` entry."""
    out: dict[str, str] = dict(DEFAULTS)
    for key, var in ENV_KEYS.items():
        spec_value = getattr(compute, key, None) if compute is not None else None
        if spec_value not in (None, ""):
            out[key] = str(spec_value)
        if env.get(var):
            out[key] = env[var]
    per_type = compute.types.get(type_name) if compute is not None else None
    if per_type is not None:
        for key, value in per_type.model_dump(exclude_none=True).items():
            out[key] = str(value)
    return out


def url_template(public_url: str, host_file: str | None) -> str:
    """The controller URL a worker gets: `http://${CTL_HOST}:<port>` with a host file (expanded
    by the worker, worker/main.py), else the controller's own URL."""
    if not host_file or "${" in public_url:  # FLOCK_WORKER_URL already a template: as is
        return public_url
    u = urlsplit(public_url)
    port = f":{u.port}" if u.port else ""
    return f"{u.scheme or 'http'}://${{CTL_HOST}}{port}{u.path}"


def _sh_quote(value: str) -> str:
    """Raw when safe, else single quotes with a literal ' written '\'' (the cluster's format)."""
    return value if _SAFE.match(value) else "'" + value.replace("'", "'\\''") + "'"


def env_file_text(env: Mapping[str, str], secret_names: list[str]) -> str:
    """A shell-sourceable env file (`set -a; . FILE; set +a`): `KEY=value` lines, single-quoted
    where needed (so `${CTL_HOST}` reaches the worker unexpanded) and a comment naming the
    secrets the job must provide."""
    lines = [
        "# flock worker env (written by the controller; no secret values).",
        "# Secrets this worker expects from the cluster's secrets file: "
        + " ".join(secret_names),
    ]
    for key in sorted(env):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"env name {key!r} is not a shell variable name")
        lines.append(f"{key}={_sh_quote(str(env[key]))}")
    return "\n".join(lines) + "\n"


def write_private(path: Path, text: str) -> None:
    """Write `path` atomically with mode 600 (a job starting meanwhile reads old or new)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def read_env_file(path: Path) -> dict[str, str]:
    """The `KEY=value` lines of an env file written by `env_file_text`."""
    out: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, raw = line.removeprefix("export ").partition("=")
            try:
                parts = shlex.split(raw)
            except ValueError:
                continue
            out[key] = parts[0] if parts else ""
    return out


def worker_env(wid: str, env: Mapping[str, str], secrets: Mapping[str, str],
               cfg: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """The env file's settings and the secret names, for one worker. Secret values are dropped
    here: they never reach a file or a command line."""
    host_file = cfg.get("host_file") or None
    run = cfg.get("run") or (Path(cfg["launch_dir"]).parent.name if cfg.get("launch_dir") else "")
    if root := (cfg.get("container_root") or "").rstrip("/"):
        # a container: its /tmp is RAM counted against the job's memory, so nothing goes there
        base = f"{root}/flock/{wid}"
        paths = {"WORK_ROOT": f"{base}/work", "FLOCK_LLM_HOME": f"{base}/llm",
                 "HOME": f"{base}/home", "HF_HOME": f"{base}/hf", "TMPDIR": f"{base}/tmp"}
    else:  # bare node: node-local and fixed per slot (restores go to the same paths)
        base = f"{WORK_BASE}/{wid}"
        paths = {"WORK_ROOT": f"{base}/work", "FLOCK_LLM_HOME": f"{base}/llm"}
    out = {
        **paths,
        "PYTHONUNBUFFERED": "1",
        "RUN": run,
        **{k: v for k, v in env.items() if k not in secrets},
    }
    out["FLOCK_URL"] = url_template(env.get("FLOCK_URL", ""), host_file)
    if host_file:
        out["FLOCK_CTL_HOST_FILE"] = host_file
    if cfg.get("venv"):
        out["FLOCK_VENV"] = cfg["venv"]
    if cfg.get("secrets_file"):
        out["FLOCK_SECRETS_FILE"] = cfg["secrets_file"]
    if cfg.get("container_image"):  # the worker script runs flock-worker in it (srun/pyxis)
        out["FLOCK_CONTAINER_IMAGE"] = cfg["container_image"]
    names = list(dict.fromkeys([*ALWAYS_SECRETS, *secrets]))
    return out, names


def env_file_path(launch_dir: str, worker_id: str, env: Mapping[str, str]) -> Path:
    """`<launch_dir>/<worker id>-<incarnation>.env` (one file per launch)."""
    inc = env.get("WORKER_INCARNATION") or uuid.uuid4().hex
    return Path(launch_dir) / f"{worker_id}-{inc}.env"


def latest_env(launch_dir: str, worker_id: str) -> dict[str, str]:
    """The settings of a slot's newest env file (rebuild_from_jobs reads DEADLINE and the
    incarnation from it)."""
    files = sorted(Path(launch_dir).glob(f"{glob_escape(worker_id)}-*.env"),
                   key=lambda p: p.stat().st_mtime)
    return read_env_file(files[-1]) if files else {}


def glob_escape(s: str) -> str:
    return re.sub(r"([*?\[])", r"[\1]", s)


def _labels_comment(labels: Mapping[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


def _parse_comment(raw: str) -> dict[str, str]:
    out = {}
    for part in raw.split(","):
        k, sep, v = part.partition("=")
        if sep:
            out[k.strip()] = v.strip()
    return out


@dataclass
class SlurmJobsClient:
    """sbatch / squeue / sacct / scancel. `env` is the controller's environment (for the
    FLOCK_SLURM_* overrides); `run` runs one command (tests: shell stubs on PATH)."""

    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    run: Runner = _run
    calls: Counter = field(default_factory=Counter)
    # reconcile passes the swarm's `compute` block to launch() (ComputeSpec keys)
    uses_compute_spec: bool = True
    external: bool = False

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._launched: dict[str, tuple[str, float, str]] = {}  # job id -> (worker, time, log dir)
        self._requested_cancel: set[str] = set()
        self._dirs: dict[str, str] = {}  # job id -> launch dir (for list/cancel markers)

    # ------------------------------------------------------------------ helpers
    def _call(self, kind: str, argv: list[str]) -> subprocess.CompletedProcess:
        self.calls[kind] += 1
        return self.run(argv)

    def _cancel_marker(self, launch_dir: str, job_id: str) -> Path:
        return Path(launch_dir) / "cancelled" / job_id

    def _was_cancel_requested(self, job_id: str) -> bool:
        if job_id in self._requested_cancel:
            return True
        launch_dir = self._dirs.get(job_id) or resolve(None, "", self.env).get("launch_dir")
        return bool(launch_dir) and self._cancel_marker(launch_dir, job_id).exists()

    # ------------------------------------------------------------------ protocol
    def launch(self, worker_id: str, spec: TypeSpec, env: dict[str, str],
               secrets: dict[str, str], labels: dict[str, str], timeout: float, *,
               namespace: str, compute: ComputeSpec | None = None) -> str:
        cfg = resolve(compute, env.get("TYPE", ""), self.env)
        missing = [k for k in ("worker_script", "log_dir", "launch_dir") if not cfg.get(k)]
        if missing:
            raise SlurmConfigError(f"compute slurm needs {', '.join(missing)} (spec `compute:` keys "
                             f"or {', '.join(ENV_KEYS[k] for k in missing)})")
        if problem := size_problem(cfg["cpus"], cfg["mem"], cfg["max_cpus"], cfg["max_mem"]):
            raise SlurmConfigError(f"{env.get('TYPE', '')}: {problem}")
        settings, names = worker_env(worker_id, env, secrets, cfg)
        env_path = env_file_path(cfg["launch_dir"], worker_id, env)
        write_private(env_path, env_file_text(settings, names))
        Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)
        argv = ["sbatch", "--parsable", "-p", cfg["partition"]]
        if cfg.get("account"):
            argv += ["-A", cfg["account"]]
        argv += [
            "-c", cfg["cpus"], f"--mem={cfg['mem']}", f"--time={cfg['time']}",
            # the worker relaunches through the controller (fresh incarnation), never in place
            "--no-requeue",
            "-J", worker_id, f"--comment={_labels_comment(labels)}",
            "-o", f"{cfg['log_dir']}/{worker_id}-%j.out",
            cfg["worker_script"], str(env_path),
        ]
        r = self._call("sbatch", argv)
        if r.returncode != 0:
            raise RuntimeError(f"sbatch exit {r.returncode}: {(r.stderr or r.stdout).strip()[:300]}")
        job_id = (r.stdout or "").strip().splitlines()[-1].split(";")[0].strip() if r.stdout else ""
        if not _JOB_ID.match(job_id):
            raise RuntimeError(f"sbatch printed no job id: {r.stdout!r}")
        with self._lock:
            self._launched[job_id] = (worker_id, time.monotonic(), cfg["log_dir"])
            self._dirs[job_id] = cfg["launch_dir"]
        log.info("sbatch %s for %s (%s cpus, %s, %s)", job_id, worker_id, cfg["cpus"],
                 cfg["mem"], cfg["partition"])
        return job_id

    def _squeue_state(self, job_id: str) -> str | None:
        r = self._call("squeue", ["squeue", "-h", "-j", job_id, "-o", "%T"])
        if r.returncode != 0:
            return None  # "Invalid job id": purged from squeue, ask sacct
        out = (r.stdout or "").strip().split()
        return out[0] if out else None

    def _sacct_state(self, job_id: str) -> tuple[str, str] | None:
        r = self._call("sacct", ["sacct", "-j", job_id, "-X", "-n", "-P", "-o", "State,ExitCode"])
        if r.returncode != 0:
            return None
        for line in (r.stdout or "").splitlines():
            parts = [p.strip() for p in line.split("|")]
            if parts and parts[0]:
                return parts[0], parts[1] if len(parts) > 1 else ""
        return None

    def map_state(self, job_id: str, state: str, exit_code: str = "") -> JobState:
        word = state.split()[0].rstrip("+") if state else ""
        code = exit_code.split(":")[0] if exit_code else ""
        if word in SCHEDULING:
            return JobState("SCHEDULING", word)
        if word in RUNNING:
            return JobState("RUNNING")
        if word == "COMPLETED":
            return JobState("COMPLETED")
        if word == "TIMEOUT":
            return JobState("ERROR", "Job timeout")
        if word == "CANCELLED":
            if self._was_cancel_requested(job_id):
                return JobState("CANCELED")
            return JobState("ERROR", "preempted: CANCELLED (not requested by flock)",
                            preempted=True)
        if word in PREEMPTED:
            return JobState("ERROR", f"preempted: {word}", preempted=True)
        if code == str(PREEMPT_EXIT):
            return JobState("ERROR", f"preempted: {word} exit {code}", preempted=True)
        return JobState("ERROR", f"{word or 'UNKNOWN'} exit {code}".strip())

    def inspect(self, job_id: str, *, namespace: str = "") -> JobState:
        state = self._squeue_state(job_id)
        if state is not None:
            if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                         "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"):
                acct = self._sacct_state(job_id)  # the exit code lives in sacct
                if acct is not None:
                    return self.map_state(job_id, *acct)
            return self.map_state(job_id, state)
        acct = self._sacct_state(job_id)
        if acct is not None:
            return self.map_state(job_id, *acct)
        launched = self._launched.get(job_id)
        if launched is not None and time.monotonic() - launched[1] < NOT_IN_SACCT_GRACE:
            return JobState("SCHEDULING", "not in squeue/sacct yet")
        return JobState("DELETED")

    def cancel(self, job_id: str, *, namespace: str = "") -> None:
        with self._lock:
            self._requested_cancel.add(job_id)
        launch_dir = self._dirs.get(job_id) or resolve(None, "", self.env).get("launch_dir")
        if launch_dir:  # survives a controller restart: CANCELLED then still means "ours"
            marker = self._cancel_marker(launch_dir, job_id)
            try:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
            except OSError as e:
                log.warning("cancel marker %s: %s", marker, e)
        r = self._call("scancel", ["scancel", job_id])
        if r.returncode != 0:
            log.warning("scancel %s: %s", job_id, (r.stderr or "").strip()[:200])

    def list(self, labels: dict[str, str], *, namespace: str = "") -> list[JobListing]:
        r = self._call("squeue", ["squeue", "--me", "-h", "-o", "%i|%j|%k|%T"])
        if r.returncode != 0:
            raise RuntimeError(f"squeue exit {r.returncode}: {(r.stderr or '').strip()[:200]}")
        out = []
        for line in (r.stdout or "").splitlines():
            parts = line.split("|")
            if len(parts) < 4:
                continue
            job_id, name, comment, state = (p.strip() for p in parts[:4])
            have = _parse_comment(comment)
            if not all(have.get(k) == v for k, v in labels.items()):
                continue
            launch_dir = self._dirs.get(job_id) or resolve(None, "", self.env).get("launch_dir")
            env = latest_env(launch_dir, name) if launch_dir else {}
            st = self.map_state(job_id, state)
            out.append(JobListing(job_id, have, st.stage, env, st.message))
        return out

    def log_path(self, job_id: str) -> str | None:
        """`<log_dir>/<worker id>-<job id>.out` of a job this client launched (reconcile keeps
        it on the slot row, so a restarted controller still finds a crash's last lines)."""
        launched = self._launched.get(job_id)
        if launched is None:
            return None
        worker_id, _, log_dir = launched
        return str(Path(log_dir) / f"{worker_id}-{job_id}.out")

    def logs_tail(self, job_id: str, *, namespace: str = "", n: int = 5) -> list[str]:
        path_s = self.log_path(job_id)
        if path_s is None:  # launched by an earlier controller: look in the env's log dir
            log_dir = resolve(None, "", self.env).get("log_dir")
            found = sorted(Path(log_dir).glob(f"*-{job_id}.out")) if log_dir else []
            if not found:
                return []
            path_s = str(found[0])
        path = Path(path_s)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return lines[-n:]


@dataclass
class ExternalJobsClient:
    """`compute: external`: the controller launches nothing. Each "launch" writes the slot's
    env file (when a launch dir is configured) and logs what it waits for; the worker started
    by someone else registers through its heartbeat."""

    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    calls: Counter = field(default_factory=Counter)
    uses_compute_spec: bool = True
    external: bool = True
    waiting: dict[str, str] = field(default_factory=dict)  # job id -> worker id

    def launch(self, worker_id: str, spec: TypeSpec, env: dict[str, str],
               secrets: dict[str, str], labels: dict[str, str], timeout: float, *,
               namespace: str, compute: ComputeSpec | None = None) -> str:
        self.calls["launch"] += 1
        cfg = resolve(compute, env.get("TYPE", ""), self.env)
        settings, names = worker_env(worker_id, env, secrets, cfg)
        where = ""
        if cfg.get("launch_dir"):
            path = env_file_path(cfg["launch_dir"], worker_id, env)
            write_private(path, env_file_text(settings, names))
            where = f"; env file {path}"
        job_id = f"external-{uuid.uuid4().hex[:8]}"
        self.waiting[job_id] = worker_id
        log.info("waiting for an external start of %s (WORKER_ID=%s, FLOCK_URL=%s, secrets %s%s)",
                 worker_id, worker_id, settings["FLOCK_URL"], " ".join(names), where)
        return job_id

    def inspect(self, job_id: str, *, namespace: str = "") -> JobState:
        return JobState("SCHEDULING", "waiting for an external start")

    def cancel(self, job_id: str, *, namespace: str = "") -> None:
        self.calls["cancel"] += 1
        self.waiting.pop(job_id, None)

    def list(self, labels: dict[str, str], *, namespace: str = "") -> list[JobListing]:
        return []

    def logs_tail(self, job_id: str, *, namespace: str = "", n: int = 5) -> list[str]:
        return []
