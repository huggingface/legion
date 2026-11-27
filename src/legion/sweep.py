"""Run a pinned Harbor experiment matrix on Slurm, with a receipt for each attempt."""

from __future__ import annotations

import fcntl
import getpass
import hashlib
import itertools
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from legion import ops, rundir
from legion.client import Client, ClientError
from legion.harbor.results import safe_error, write_json
from legion.harbor.spec import Duration, Spec, substitute
from legion.models import load_spec_yaml
from legion.secrets_file import SecretsFileError, read_file
from legion.specrender import read_env_file

_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")
_POLL = 5.0
_DRAIN_TIMEOUT = 15 * 60.0
_OPEN = {"preparing", "launching", "running", "draining"}


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: str = "spec.yaml"
    overrides: dict[str, Any] = Field(default_factory=dict)
    matrix: dict[str, dict[str, dict[str, Any]]] = Field(min_length=1)
    max_runs: int = Field(default=1, strict=True, ge=1)
    max_agents: int | None = Field(default=None, strict=True, ge=1)
    infrastructure_retries: int = Field(default=0, strict=True, ge=0)
    run_timeout: Duration = 3600


@dataclass(frozen=True)
class Cell:
    key: str
    labels: dict[str, str]
    raw: dict
    spec: Spec
    fingerprint: str
    dependencies: dict[Path, Path]
    dependency_hash: str

    @property
    def agents(self) -> int:
        return sum(role.count for role in self.spec.roles().values())

    def describe(self) -> dict:
        return {"cell": self.key, "axes": self.labels, "fingerprint": self.fingerprint,
                "agents": self.agents, "model": self.spec.model, "revision": self.spec.task.revision,
                "selector": self.spec.task.selector}


@dataclass(frozen=True)
class Plan:
    config: SweepConfig
    source: Path
    cells: list[Cell]
    fingerprint: str
    source_hash: str
    credentials: tuple[str, ...] = field(repr=False)


def _digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _overlay(base: dict, patch: dict) -> dict:
    """Merge mappings, replace scalars/lists, and let null remove inherited keys."""
    merged = dict(base)
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict):
            old = merged.get(key)
            merged[key] = _overlay(old if isinstance(old, dict) else {}, value)
        else:
            merged[key] = value
    return merged


def _yaml(path: Path) -> dict:
    try:
        data = load_spec_yaml(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a YAML mapping")
    return data


def _tree_hash(root: Path) -> str:
    """Hash the staged paths and contents, including links that packaging can reject."""
    return _digest({
        str(path.relative_to(root)): (
            ["link", os.readlink(path)] if path.is_symlink()
            else ["file", path.stat().st_mode & 0o111, hashlib.sha256(path.read_bytes()).hexdigest()]
        )
        for path in root.rglob("*")
        if "__pycache__" not in path.parts and (path.is_symlink() or path.is_file())
    })


def _bind_dependencies(raw: dict, spec: Spec, source: Path, secrets_file: Path | None
                       ) -> dict[Path, Path]:
    """Keep spec dependencies relative to the staged run, including absolute inputs."""
    outside: dict[Path, Path] = {}

    def bind(mapping: dict, name: str, original: str | Path | None) -> None:
        if original is None:
            return
        path = source / Path(original).expanduser()
        target = path.resolve()
        if secrets_file is not None and target.is_file() and target.samefile(secrets_file):
            raise ValueError("a sweep spec input points to its secrets file")
        if path.is_relative_to(source) and ".." not in path.parts:
            relative = path.relative_to(source)
        else:
            relative = Path(".sweep-inputs") / _digest(str(target)) / target.name
            outside[relative] = target
        mapping[name] = relative.as_posix()

    for name in ("tools_dir", "resources_dir"):
        bind(raw, name, getattr(spec, name))
    bind(raw, "team_file", spec.team_file)
    loops = [(raw, spec.loop)]
    for name, role in (spec.agents.roles or {}).items():
        data = raw["agents"]["roles"][name]
        bind(data, "role_file", role.role_file)
        loops.append((data, role.loop))
    for data, loop in loops:
        if loop is not None:
            bind(data["loop"], "prompt_file", loop.prompt_file)
            if loop.final is not None:
                bind(data["loop"]["final"], "prompt_file", loop.final.prompt_file)
    return outside


def _copy_dependencies(dependencies: dict[Path, Path], root: Path,
                       secrets_file: Path | None) -> None:
    for relative, source in dependencies.items():
        dst = root / relative
        if source.is_dir():
            rundir.copy_experiment_tree(source, dst, exclude=secrets_file, symlinks=True)
        elif source.is_file():
            if secrets_file is not None and source.samefile(secrets_file):
                raise ValueError("a sweep spec input points to its secrets file")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dst)


def _sweep_error(exc: BaseException, credentials: tuple[str, ...]) -> str:
    if not isinstance(exc, ValidationError):
        return safe_error(exc, credentials)
    # Pydantic's display trims input values. Render only validation locations and
    # messages, so the redactor never has to recognize a partial input credential.
    details = "\n".join(
        f"{'.'.join(map(str, item['loc'])) or 'spec'}: {item['msg']}"
        for item in exc.errors(include_input=False, include_context=False, include_url=False)
    )
    return safe_error(ValueError(details), credentials)


def make_plan(path: Path, *, profile: Path | None = None, secrets_file: Path | None = None) -> Plan:
    """Expand and validate a matrix without contacting a controller, Hub or scheduler."""
    env = dict(os.environ)
    if profile is not None:
        env.update(read_env_file(profile))
    credentials: tuple[str, ...] = ()
    if secrets_file is not None:
        try:
            file_values = read_file(secrets_file)
        except SecretsFileError as exc:
            raise ValueError(str(exc)) from None
        credentials = tuple(file_values.values())
        env.update(file_values)
    try:
        return _plan(path, profile, secrets_file, env, credentials)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(_sweep_error(exc, credentials)) from None


def _plan(path: Path, profile: Path | None, secrets_file: Path | None,
          env: dict[str, str], credentials: tuple[str, ...]) -> Plan:
    config = SweepConfig.model_validate(_yaml(path))
    source = (path.parent / config.spec).resolve()
    base = _overlay(_yaml(source), config.overrides)
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary) / "experiment"
        rundir.copy_experiment(source.parent, stage, rundir.InitReport(),
                               need=source.name, exclude=secrets_file)
        source_hash = _tree_hash(stage)
    axes = sorted(config.matrix)
    for axis in axes:
        choices = config.matrix[axis]
        if not choices or any(not _LABEL.fullmatch(label) for label in (axis, *choices)):
            raise ValueError(f"matrix axis {axis!r} needs choices with simple labels")
    cells = []
    hashes: dict[tuple, str] = {}
    for values in itertools.product(*(sorted(config.matrix[axis]) for axis in axes)):
        labels = dict(zip(axes, values, strict=True))
        key = _digest(labels)[:12]
        raw = base
        for axis, label in labels.items():
            raw = _overlay(raw, config.matrix[axis][label])
        # The actual attempt gets its own name, independent of any name in the base spec.
        raw = deepcopy({**raw, "name": f"sweep-{key}"})
        spec = Spec.model_validate(substitute(raw, env))
        dependencies = _bind_dependencies(raw, spec, source.parent, secrets_file)
        spec = Spec.model_validate(substitute(raw, env))
        if dependencies:
            paths = tuple(sorted(dependencies.items()))
            if paths not in hashes:
                with tempfile.TemporaryDirectory() as temporary:
                    _copy_dependencies(dependencies, Path(temporary), secrets_file)
                    hashes[paths] = _tree_hash(Path(temporary))
            dependency_hash = hashes[paths]
        else:
            dependency_hash = _digest({})
        roles = spec.roles()
        if (spec.compute.kind != "slurm" or spec.spawn or spec.stop_on_success
                or any(role.ramp is not None or role.count == 0 for role in roles.values())):
            raise ValueError(f"{labels}: sweeps need a fixed-team Slurm spec that grades every agent")
        models = [spec.inference, *(role.inference for role in roles.values())]
        if any(inference and inference.hf_jobs for inference in models):
            raise ValueError(f"{labels}: provide existing inference endpoints for a sweep")
        if spec.type_loops().keys() != roles.keys() or not re.fullmatch(r"[0-9a-f]{40}", spec.task.revision):
            raise ValueError(f"{labels}: configure a loop and pin task.revision to a commit SHA")
        fingerprint = _digest([labels, spec.model_dump(mode="json"), source_hash, dependency_hash])
        cell = Cell(key, labels, raw, spec, fingerprint, dependencies, dependency_hash)
        if cell.agents < 1 or (config.max_agents is not None and cell.agents > config.max_agents):
            raise ValueError(f"{labels}: {cell.agents} agents do not fit the sweep's capacity")
        cells.append(cell)
    if len({cell.key for cell in cells}) != len(cells):
        raise ValueError("matrix cell keys collide")
    fingerprint = _digest([config.model_dump(), [c.fingerprint for c in cells],
                           hashlib.sha256(profile.read_bytes()).hexdigest() if profile else None])
    return Plan(config, source, cells, fingerprint, source_hash, credentials)


class Attempt(BaseModel):
    phase: Literal["preparing", "launching", "running", "draining",
                   "completed", "infrastructure_failure", "setup_failed"] = "preparing"
    started: float = Field(default_factory=time.time)
    workers: dict[str, str] = Field(default_factory=dict)
    agents: list[str] = Field(default_factory=list)
    failure: str | None = None
    note: str | None = None
    input_hash: str | None = None


class _Paused(Exception):
    """The attempt's ownership or state needs to be reconciled before any new work."""


def _load(run: Path) -> Attempt:
    return Attempt.model_validate_json((run / "attempt.json").read_text())


def _save(run: Path, attempt: Attempt) -> None:
    write_json(run / "attempt.json", attempt.model_dump())


def _check_stop(stop: Event, delay: float = 0) -> None:
    if stop.wait(delay):
        raise _Paused("driver stopped; resume this sweep to continue watching the attempt")


def _prepare(plan: Plan, cell: Cell, run: Path, profile: Path | None, secrets_file: Path | None) -> str:
    target = run / rundir.EXPERIMENT
    with tempfile.TemporaryDirectory(dir=run) as temporary:
        stage = Path(temporary) / rundir.EXPERIMENT
        rundir.copy_experiment(plan.source.parent, stage, rundir.InitReport(),
                               need=plan.source.name, exclude=secrets_file)
        if _tree_hash(stage) != plan.source_hash:
            raise ValueError("experiment files changed since the sweep was planned")
        staged_deps = Path(temporary) / "dependencies"
        _copy_dependencies(cell.dependencies, staged_deps, secrets_file)
        if _tree_hash(staged_deps) != cell.dependency_hash:
            raise ValueError("external experiment files changed since the sweep was planned")
        if staged_deps.is_dir():
            shutil.copytree(staged_deps, stage, symlinks=True, dirs_exist_ok=True)
        (stage / "spec.yaml").write_text(yaml.safe_dump({**cell.raw, "name": run.name}, sort_keys=False))
        input_hash = _tree_hash(stage)
        if target.exists():
            if _tree_hash(target) != input_hash:
                raise ValueError("the staged experiment files changed since preparation")
        else:
            stage.rename(target)
    report = rundir.init(run, profile=profile, secrets_file=secrets_file)
    if report.failed:
        raise ValueError("legion init failed: " + ", ".join(c.name for c in report.checks if c.ok is False))
    return input_hash


def _jobs_active(names: list[str]) -> bool:
    return bool(names and ops._state(",".join(names)))


def _control_jobs(run: Path) -> list[str]:
    return [ops._job_name(role, run) for role in ("controller", "world")]


def _client(run: Path) -> Client:
    # Sweep attempts always talk to their own controller and its generated token.
    env = {k: v for k, v in os.environ.items() if k not in {"LEGION_CURRENT", "LEGION_AGENT_TOKEN"}}
    return Client.for_run(str(run), env=env, need_host=False)


def _status(api: Client, run: Path) -> dict | None:
    if not api.answers():
        return None
    try:
        return api.swarm(run.name)["status"]
    except ClientError as exc:
        if exc.status != 404:
            raise
        return None


def _running_status(api: Client, run: Path) -> dict:
    status = _status(api, run)
    if status is None:
        raise _Paused("the running controller is unreachable; resume to reconcile its jobs")
    if status.get("generation", 1) != 1:
        raise _Paused("the running swarm has been edited outside the sweep")
    return status


def _launch(cell: Cell, run: Path, api: Client, fresh: bool, stop: Event,
            input_hash: str | None) -> None:
    """Reattach if the swarm exists. Retry create only on the same live controller."""
    if _status(api, run) is not None:
        return
    if not fresh:
        controller = _control_jobs(run)[0]
        deadline = time.monotonic() + ops.UP_WAIT_S
        while not api.answers():
            if not _jobs_active([controller]) or time.monotonic() >= deadline:
                raise _Paused("launch outcome unknown; reconcile the attempt's Slurm jobs before retrying")
            _check_stop(stop, _POLL)
            api = _client(run)
        if _status(api, run) is not None:
            return
        if not _jobs_active([controller]):
            raise _Paused("the original controller is no longer running; launch outcome unknown")
    _check_stop(stop)
    path = run / rundir.EXPERIMENT / "spec.yaml"
    if input_hash is None or _tree_hash(path.parent) != input_hash:
        raise _Paused("the staged experiment files changed since preparation")
    effective = Spec.model_validate(substitute(_yaml(path), ops.environment(api, api.env_files)))
    if effective.model_dump(exclude={"name"}) != cell.spec.model_dump(exclude={"name"}):
        raise _Paused("the attempt's resolved spec differs from the sweep plan")
    spec = ops.read_spec(str(path), api.env_files)
    if spec.swarm != run.name or sum(t.replicas for t in spec.types.values()) != cell.agents:
        raise _Paused("the compiled swarm does not match the attempt's name or agent reservation")
    ops.launch(api, spec, str(path), env_files=api.env_files, start_controller=fresh,
               say=lambda _: _check_stop(stop))


def _remember(run: Path, attempt: Attempt, api: Client) -> None:
    for worker in api.workers(run.name):
        if worker["crashes"]:
            attempt.failure = "worker_crash"
        if job := worker["job_id"]:
            old = attempt.workers.setdefault(worker["id"], job)
            if old != job:
                attempt.failure = "worker_replacement"
        attempt.agents = sorted(set(attempt.agents) | set(worker["agents"]))
    _save(run, attempt)


def _finished(status: dict, agents: int) -> bool:
    return (
        status["agents_total"] == agents and not status["pending_prompts"]
        and all(
            row["loop"] and row["loop"]["state"] == "done"
            and row["agents_by_state"].get("idle", 0) == row["replicas"]
            for row in status["types"].values()
        )
    )


def _drain(run: Path, attempt: Attempt, stop: Event) -> None:
    api = _client(run)
    status = _status(api, run)
    if status is None:
        if _jobs_active(_control_jobs(run)):
            raise _Paused("controller or world is still live; resume when its API is reachable")
        return
    deadline = time.monotonic() + _DRAIN_TIMEOUT

    def progress(_):
        _check_stop(stop)
        if time.monotonic() >= deadline:
            raise _Paused("workers are still draining; resume to check again")

    if not status.get("released"):
        api.stop(run.name, release=True, interrupt=bool(attempt.failure), reason="sweep")
    if ops.release_wait(api, run.name, progress=progress):
        attempt.failure = "unsynced_workers"
        _save(run, attempt)
    ops.down(api, run.name)


def _account_workers(run: Path, attempt: Attempt, stop: Event) -> bool:
    names = list(attempt.workers)
    deadline = time.monotonic() + 60
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(attempt.started - 60))
    while True:
        if not _jobs_active([*_control_jobs(run), *names]):
            break
        if time.monotonic() >= deadline:
            raise _Paused("attempt jobs have not finished draining")
        _check_stop(stop, _POLL)
    if not names:
        return False
    while True:
        output = ops._slurm("sacct", "-X", "-n", "-P", "--user", getpass.getuser(), "--starttime", since,
                            "--name", ",".join(names), "--format", "JobName%128,JobIDRaw,State,ExitCode").stdout
        rows: dict[str, dict[str, tuple[str, str]]] = {name: {} for name in names}
        for line in output.splitlines():
            name, job, state, code = line.split("|")[:4]
            if name in rows:
                rows[name][job] = (state, code)
        if all(job in rows[name] for name, job in attempt.workers.items()):
            return all(jobs == {attempt.workers[name]: ("COMPLETED", "0:0")} for name, jobs in rows.items())
        if time.monotonic() >= deadline:
            raise _Paused("worker accounting is incomplete; resume to reconcile the attempt")
        _check_stop(stop, _POLL)


def _results_present(run: Path, agents: list[str], expected: int) -> bool:
    root = run / "results" / run.name
    trials: dict[str, list[Path]] = {}
    for path in root.glob("*/trials/*"):
        if path.is_dir():
            trials.setdefault(path.parents[1].name, []).append(path)
    if len(trials) != expected or sorted(trials) != agents:
        return False

    def trial_number(path: Path) -> int:
        match = re.search(r"(?:^|-)t(\d+)$", path.name)
        return int(match[1]) if match else 0

    for paths in trials.values():
        path = max(paths, key=trial_number) / "result.json"
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            return False
        verifier = row.get("verifier_result") if isinstance(row, dict) else None
        rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
        steps = (row.get("step_results") or []) if isinstance(row, dict) else []
        if (not isinstance(rewards, dict) or not rewards or row.get("exception_info")
                or not isinstance(steps, list)
                or any(not isinstance(step, dict) or step.get("exception_info") for step in steps)
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in rewards.values())):
            return False
    return True


def _run_attempt(plan: Plan, cell: Cell, run: Path, profile: Path | None, secrets_file: Path | None,
                 stop: Event, init_lock: Lock) -> bool:
    """True if settled. False leaves the receipt at its current phase for reconciliation."""
    attempt = _load(run)
    try:
        fresh = attempt.phase == "preparing"
        if fresh:
            _check_stop(stop)
            try:
                with init_lock:
                    _check_stop(stop)
                    attempt.input_hash = _prepare(plan, cell, run, profile, secrets_file)
            except (OSError, TypeError, ValueError, ops.SpecError) as exc:
                attempt.phase = "setup_failed"
                attempt.note = _sweep_error(exc, plan.credentials)
                _save(run, attempt)
                return True
            attempt.phase = "launching"
            attempt.started = time.time()
            _save(run, attempt)
        if attempt.phase == "launching":
            _launch(cell, run, _client(run), fresh, stop, attempt.input_hash)
            attempt.phase = "running"
            _save(run, attempt)
        while attempt.phase == "running":
            _check_stop(stop)
            api = _client(run)
            status = _running_status(api, run)
            _remember(run, attempt, api)
            finished = _finished(status, cell.agents)
            if any(t["health"] == "launch_refused" for t in status["types"].values()):
                attempt.failure = "worker_launch_refused"
            elif any(t["health"] == "crashlooping" for t in status["types"].values()):
                attempt.failure = "worker_unavailable"
            elif not finished and time.time() - attempt.started > plan.config.run_timeout:
                attempt.failure = "run_timeout"
            elif status.get("released") and not finished:
                attempt.failure = "released_before_completion"
            elif finished and any(
                agent.get("done_reason") == "errors"
                for loop in api.loops(run.name) for agent in (loop.get("per_agent") or {}).values()
            ):
                attempt.failure = "agent_execution_error"
            if attempt.failure or finished:
                attempt.phase = "draining"
                _save(run, attempt)
            else:
                _check_stop(stop, _POLL)
        _drain(run, attempt, stop)
        clean = _account_workers(run, attempt, stop)
        if not attempt.failure and not clean:
            attempt.failure = "worker_accounting_failed"
        if not attempt.failure and not _results_present(run, attempt.agents, cell.agents):
            attempt.failure = "missing_or_failed_trials"
        if attempt.failure == "worker_launch_refused":
            attempt.phase = "setup_failed"
        else:
            attempt.phase = "infrastructure_failure" if attempt.failure else "completed"
        attempt.note = None
        _save(run, attempt)
        return True
    except (_Paused, ClientError, ops.SpecError, OSError, TypeError, ValueError) as exc:
        attempt.note = _sweep_error(exc, plan.credentials)
        _save(run, attempt)
        return False


def _attempts(root: Path, cell: Cell) -> list[Path]:
    return sorted((root / cell.key).glob("*/attempt.json"), key=lambda path: int(path.parent.name.rsplit("-r", 1)[1]))


def _rows(root: Path, plan: Plan) -> list[dict]:
    rows = []
    for cell in plan.cells:
        attempts = []
        for path in _attempts(root, cell):
            run = path.parent
            attempt = _load(run)
            attempts.append({"run": str(run.relative_to(root)), "status": attempt.phase,
                             "failure": attempt.failure, "note": attempt.note})
        selected = attempts[-1] if attempts else {}
        artifacts = None
        if selected.get("status") == "completed":
            relative = Path(selected["run"])
            artifacts = str(relative / "results" / relative.name)
        rows.append({**cell.describe(), "status": selected.get("status", "pending"),
                     "selected_attempt": selected.get("run"), "artifacts": artifacts, "attempts": attempts})
    return rows


def _write_rows(root: Path, plan: Plan) -> list[dict]:
    rows = _rows(root, plan)
    temporary = root / ".rows.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(root / "rows.jsonl")
    return rows


def _pending(root: Path, plan: Plan) -> list[tuple[Cell, Path | None, int]]:
    open_runs, new_runs = [], []
    for cell in plan.cells:
        paths = _attempts(root, cell)
        if not paths:
            new_runs.append((cell, None, 0))
            continue
        last = paths[-1].parent
        phase = _load(last).phase
        if phase in _OPEN:
            open_runs.append((cell, last, len(paths) - 1))
        elif phase == "infrastructure_failure" and len(paths) <= plan.config.infrastructure_retries:
            new_runs.append((cell, None, len(paths)))
    return open_runs + new_runs


def _reconcile(plan: Plan, cell: Cell, run: Path, stop: Event) -> str | None:
    """Check resumed receipts without admitting another job or waiting for them to finish."""
    attempt = _load(run)
    if attempt.phase == "preparing":
        return None  # This receipt was saved before any submission.
    try:
        api = _client(run)
        if attempt.phase == "launching":
            _launch(cell, run, api, False, stop, attempt.input_hash)
            attempt.phase = "running"
        if attempt.phase == "running":
            _running_status(_client(run), run)
        if (attempt.phase == "draining" and _status(api, run) is None
                and _jobs_active(_control_jobs(run))):
            raise _Paused("controller or world is still live; resume when its API is reachable")
        attempt.note = None
    except (_Paused, ClientError, ops.SpecError, OSError, TypeError, ValueError) as exc:
        attempt.note = _sweep_error(exc, plan.credentials)
    _save(run, attempt)
    return attempt.note


def execute(plan: Plan, root: Path, *, profile: Path | None = None, secrets_file: Path | None = None,
            say: Callable[[str], None] = print) -> list[dict]:
    """Resume in-flight attempts before spending slots on new ones."""
    root = root.resolve()
    if root.is_relative_to(plan.source.parent):
        raise ValueError("place the sweep output outside the experiment folder")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / ".sweep.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"another sweep driver holds {root}") from None
        manifest = root / "sweep.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved["fingerprint"] != plan.fingerprint:
                raise ValueError("the sweep inputs changed; use a new output directory")
        else:
            saved = {"id": secrets.token_hex(4), "fingerprint": plan.fingerprint,
                     "cells": [cell.describe() for cell in plan.cells]}
            write_json(manifest, saved)
        return _schedule(plan, root, saved["id"], profile, secrets_file, say)


def _schedule(plan: Plan, root: Path, sweep_id: str, profile: Path | None, secrets_file: Path | None,
              say: Callable[[str], None]) -> list[dict]:
    stop, init_lock = Event(), Lock()
    active = {}
    pending = _pending(root, plan)
    for cell, run, _ in pending:
        if run is None:
            break
        if note := _reconcile(plan, cell, run, stop):
            say(f"{run.name}: {note}")
            return _write_rows(root, plan)
    with ThreadPoolExecutor(max_workers=plan.config.max_runs) as pool:
        try:
            while active or (pending and not stop.is_set()):
                while pending and not stop.is_set() and len(active) < plan.config.max_runs:
                    cell, run, index = pending[0]
                    if plan.config.max_agents is not None and (
                        sum(item[0].agents for item in active.values()) + cell.agents > plan.config.max_agents
                    ):
                        break
                    pending.pop(0)
                    if run is None:
                        run = root / cell.key / f"sweep-{sweep_id}-{cell.key}-r{index}"
                        run.mkdir(parents=True, exist_ok=True)
                        _save(run, Attempt())
                    say(f"{run.name}: {_load(run).phase}")
                    future = pool.submit(_run_attempt, plan, cell, run, profile, secrets_file, stop, init_lock)
                    active[future] = (cell, run, index)
                if not active:
                    break
                done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    cell, run, index = active.pop(future)
                    attempt = _load(run)
                    say(f"{run.name}: {attempt.failure or attempt.note or attempt.phase}")
                    if not future.result():
                        stop.set()
                    elif attempt.phase == "infrastructure_failure" and index < plan.config.infrastructure_retries:
                        pending.append((cell, None, index + 1))
                    _write_rows(root, plan)
        except BaseException:
            stop.set()
            raise
        finally:
            stop.set()
    return _write_rows(root, plan)
