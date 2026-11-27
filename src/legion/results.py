"""A run's result (docs/results.md): `load(path)` reads a finished run, local or Slurm, into one
typed `RunResult` with an outcome per Trial, per agent and for the run.

The result is read from the evidence a run already leaves; nothing is copied or written:

    local run bundle (`.legion/<swarm>-<time>/`, docs/local.md)
      run.json                                  the spec, agents (and their worker), workers
      agents/_harbor/<agent>/trials/<trial>/    Harbor's result.json, verifier/
      agents/_harbor/_workers/<worker>/<incarnation>.json
      agents/_harbor/_worker/resources/<name>/  the resources' exports (the workers share it)
    Slurm run dir (`legion init`'s RUN_DIR, docs/slurm.md)
      experiment/spec.yaml, cluster.env, run.env  the spec as the run resolved it
      attempt.json                              a sweep's receipt (legion sweep), if any
      results/<swarm>/<agent>/trials/<trial>/   Harbor's result.json, verifier/
      results/<swarm>/<worker>/resources/<name>/  the resources' exports (the board's)
      results/<swarm>/_workers/<worker>/<incarnation>.json
      launch/<worker>-<incarnation>.env         one per worker launch (older runs: no records)

Every path in a result is relative to its `root`, so a copied run directory reads the same.

A worker writes its incarnation record (legion.worker.harbor.IncarnationRecord) at start, as
agents are placed on it and at exit. A worker relaunched mid-run reuses its Trial names, so its
agents' Trial files can be overwritten (huggingface/legion#121): such agents are
`worker_replaced`, whatever their Trials say. Read a Slurm run after `legion stop --release`: its
Trials are verified as the workers drain.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from legion.harbor.results import safe_error
from legion.specrender import read_env_file

WORKERS = "_workers"  # the incarnation records' folder, beside the agents' folders
RECEIPT = "attempt.json"  # a sweep attempt's receipt in its run dir: run_id and attempt
# Harbor's own timeouts (harbor.trial.errors); any other exception in a Trial is a harness error
TIMEOUTS = frozenset({"AgentTimeoutError", "AgentSetupTimeoutError", "VerifierTimeoutError",
                      "EnvironmentStartTimeoutError"})
_TRIAL = re.compile(r".*-t(\d+)")
_LAUNCH = re.compile(r"(.+)-([0-9a-f]{32})\.env")
_ATTEMPT = re.compile(r"(.+)-r(\d+)")


class Outcome(StrEnum):
    """`completed` is a benchmark result (a reward of 0 included). Every other value is an
    infrastructure outcome: nothing to score. The order is the run's priority: a run takes its
    agents' first outcome in this order."""

    COMPLETED = "completed"  # the verifier gave finite rewards; Harbor recorded no exception
    PREEMPTED = "preempted"  # the agent's worker exited preempted (Slurm exit 75)
    WORKER_REPLACED = "worker_replaced"  # its worker was relaunched: Trials may be overwritten
    WORKER_LOST = "worker_lost"  # its worker started and never recorded an exit
    TIMEOUT = "timeout"  # one of Harbor's TIMEOUTS
    HARNESS_ERROR = "harness_error"  # any other exception in the Trial (Pi, Harbor, cancelled)
    UNKNOWN = "unknown"  # no evidence: no result.json, no verifier result, the run didn't end

    @property
    def completed(self) -> bool:
        return self is Outcome.COMPLETED


_ORDER = list(Outcome)
# A sweep receipt's `failure` (legion sweep's attempt.json) as an outcome. Its `phase` and
# `failure` stay in `reasons` as written; a failure not listed here is `unknown`.
RECEIPT_FAILURES = {
    "worker_crash": Outcome.WORKER_LOST,  # worker_replaced if a worker launched twice (_receipt)
    "worker_replacement": Outcome.WORKER_REPLACED,
    "worker_launch_refused": Outcome.WORKER_LOST,
    "worker_unavailable": Outcome.WORKER_LOST,
    "released_before_completion": Outcome.WORKER_LOST,
    "unsynced_workers": Outcome.WORKER_LOST,
    "worker_accounting_failed": Outcome.WORKER_LOST,
    "run_timeout": Outcome.TIMEOUT,
    "agent_execution_error": Outcome.HARNESS_ERROR,
}


class TrialResult(BaseModel):
    """One Harbor Trial, by reference: `result_path` is Harbor's own result.json. Paths are
    relative to the run's `root`."""

    name: str
    outcome: Outcome
    reward: float | None = None  # rewards["reward"], or the only reward
    rewards: dict[str, float] = Field(default_factory=dict)
    exception_type: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result_path: Path
    result_sha256: str | None = None  # None: no result.json
    verifier_dir: Path | None = None


class Incarnation(BaseModel):
    """One launch of a worker (its record, else its Slurm launch env file only)."""

    id: str
    job_id: str | None = None
    host: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    exit: Literal["drained", "preempted"] | None = None  # None: no exit recorded
    agents: list[str] = Field(default_factory=list)
    recorded: bool = True  # False: known from a launch env file only


class WorkerResult(BaseModel):
    id: str
    incarnations: list[Incarnation] = Field(default_factory=list)
    crashes: int = 0  # a local run.json's count
    agents: list[str] = Field(default_factory=list)

    def outcome(self) -> Outcome | None:
        """The infrastructure outcome of its agents; None when the worker was not at fault."""
        return _infra(self.incarnations, self.crashes)


def _infra(incarnations: list[Incarnation], crashes: int = 0) -> Outcome | None:
    """What the worker launches an agent ran on say: preempted, relaunched (two launches: its
    Trial names were reused), or a launch that never recorded its exit."""
    if any(i.exit == "preempted" for i in incarnations):
        return Outcome.PREEMPTED
    if len(incarnations) > 1 or crashes:
        return Outcome.WORKER_REPLACED
    if incarnations and incarnations[-1].recorded and incarnations[-1].exit is None:
        return Outcome.WORKER_LOST
    return None


class ResourceExport(BaseModel):
    """A file a resource exported as its worker exited (the board's `board.jsonl` and
    `channels.json`). `worker` is None where the workers share one folder (a local run)."""

    worker: str | None = None
    resource: str  # its name in the spec's `resources:` (message_board, ...)
    path: Path  # relative to the run's root


class AgentResult(BaseModel):
    name: str
    workers: list[str] = Field(default_factory=list)  # every worker that held it
    outcome: Outcome
    reward: float | None = None  # its latest Trial's, when that Trial completed
    trials: list[TrialResult] = Field(default_factory=list)


class RunResult(BaseModel):
    run_id: str  # the logical run: a sweep receipt's, else the swarm's name
    attempt_id: str  # this execution: the run directory's name
    attempt: int = 0  # a sweep receipt's attempt number (`-rN`), else 0
    swarm: str
    compute: Literal["local", "slurm"]
    root: Path  # absolute where it was loaded; every other path is relative to it
    outcome: Outcome
    reasons: list[str] = Field(default_factory=list)  # why the run is not completed
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_s: float | None = None
    stop_reason: str | None = None  # a local run's (idle, until, budget, success, interrupted, error)
    limits: dict[str, Any] = Field(default_factory=dict)  # the spec's `limits`
    task: dict[str, Any] = Field(default_factory=dict)  # the spec's `task` (repo, revision, selector)
    planned_agents: int | None = None  # None: not known (spawned agents)
    agents: list[AgentResult] = Field(default_factory=list)
    workers: list[WorkerResult] = Field(default_factory=list)
    exports: list[ResourceExport] = Field(default_factory=list)

    def agent(self, name: str) -> AgentResult:
        return next(a for a in self.agents if a.name == name)


def load(path: str | Path) -> RunResult:
    """The result of the run at `path`: a local run bundle, a folder holding exactly one (under
    `.legion/`), or a Slurm run directory. ValueError if it is none of those."""
    root = Path(path).resolve()
    if (run := _bundle(root)) is not None:
        return _local(root, run)
    bundles = [p.parent for p in sorted(root.glob(".legion/*/run.json")) if _bundle(p.parent)]
    if len(bundles) == 1:  # before the Slurm layout: a local run dir may hold experiment/ too
        return _local(bundles[0], _bundle(bundles[0]))
    if (root / "experiment").is_dir() or (root / "results").is_dir():
        return _slurm(root)
    found = f"{len(bundles)} run bundles under .legion/" if bundles else "no run"
    raise ValueError(f"{root}: not a local run bundle or a Slurm run directory ({found})")


# --------------------------------------------------------------------------- evidence


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _bundle(root: Path) -> dict | None:
    """A local run's run.json (the bucket's viewer run.json has no spec)."""
    doc = _read(root / "run.json")
    return doc if isinstance(doc, dict) and "spec" in doc and "agents" in doc else None


def _time(value: Any) -> datetime | None:
    """An ISO time, UTC when it names no zone."""
    try:
        t = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None
    return t.replace(tzinfo=UTC) if t is not None and t.tzinfo is None else t


def _trial(root: Path, path: Path) -> TrialResult:
    """A Trial's outcome from its own result.json."""
    result = path / "result.json"
    doc = _read(result)
    verifier = path / "verifier"
    out = TrialResult(name=path.name, outcome=Outcome.UNKNOWN,
                      result_path=result.relative_to(root),
                      verifier_dir=verifier.relative_to(root) if verifier.is_dir() else None)
    if not isinstance(doc, dict):
        return out
    out.result_sha256 = hashlib.sha256(result.read_bytes()).hexdigest()
    out.started_at, out.finished_at = _time(doc.get("started_at")), _time(doc.get("finished_at"))
    rewards = (doc.get("verifier_result") or {}).get("rewards")
    if isinstance(rewards, dict) and rewards and all(
            type(v) in (int, float) and math.isfinite(v) for v in rewards.values()):
        out.rewards = {k: float(v) for k, v in rewards.items()}
        out.reward = out.rewards.get("reward", next(iter(out.rewards.values()))
                                     if len(out.rewards) == 1 else None)
    # a multi-step Trial may record a failed step only there, with the other steps' rewards
    steps = [s.get("exception_info") for s in doc.get("step_results") or [] if isinstance(s, dict)]
    info = next((i for i in [doc.get("exception_info"), *steps] if i), None)
    if info:
        out.exception_type = str((info if isinstance(info, dict) else {}).get("exception_type")
                                 or "Exception")
        out.outcome = Outcome.TIMEOUT if out.exception_type in TIMEOUTS else Outcome.HARNESS_ERROR
    elif out.rewards:
        out.outcome = Outcome.COMPLETED
    return out


def _trials(root: Path, base: Path) -> dict[str, list[TrialResult]]:
    """Each agent's Trials under `base/<agent>/trials/`, oldest first (`-tN`)."""
    def order(p: Path):
        m = _TRIAL.fullmatch(p.name)
        return (int(m[1]) if m else 0, p.name)

    found = {}
    for agent in sorted(base.iterdir()) if base.is_dir() else []:
        if agent.name.startswith("_") or not (agent / "trials").is_dir():
            continue
        dirs = sorted((p for p in (agent / "trials").iterdir() if p.is_dir()), key=order)
        found[agent.name] = [_trial(root, p) for p in dirs]
    return found


def _incarnations(folder: Path) -> dict[str, list[Incarnation]]:
    """Worker id -> its recorded incarnations, by start."""
    out: dict[str, list[Incarnation]] = {}
    for path in sorted(folder.glob("*/*.json")) if folder.is_dir() else []:
        doc = _read(path)
        if isinstance(doc, dict):
            try:
                out.setdefault(path.parent.name, []).append(Incarnation.model_validate(
                    {k: doc.get(k) for k in Incarnation.model_fields if k in doc} | {"id": path.stem}))
            except ValueError:
                continue
    for incs in out.values():
        incs.sort(key=lambda i: (i.started_at is None, i.started_at or datetime.min.replace(tzinfo=UTC)))
    return out


def _agent(name: str, trials: list[TrialResult], workers: list[WorkerResult]) -> AgentResult:
    """Its infrastructure outcome, from every worker launch it was placed on (a re-placed agent
    restarts its Trial names on the new worker too), else its latest Trial's."""
    held = [w for w in workers if name in w.agents]
    recorded = [i for w in held for i in w.incarnations if i.recorded]
    if recorded:  # the launches it was placed on: crashes before it arrived are not its own
        infra = _infra([i for i in recorded if name in i.agents])
    else:  # no records (older runs): the whole worker's history
        infra = _infra([i for w in held for i in w.incarnations], sum(w.crashes for w in held))
    last = trials[-1] if trials else None
    return AgentResult(name=name, workers=[w.id for w in held], trials=trials,
                       outcome=infra or (last.outcome if last else Outcome.UNKNOWN),
                       reward=last.reward if last and last.outcome.completed else None)


def _receipt(root: Path, workers: list[WorkerResult]) -> tuple[Outcome, str] | None:
    """What a sweep attempt's receipt says went wrong, in its own words; None: nothing. A
    crash is a relaunch only where a worker shows another incarnation (a second launch or
    record, a counted crash)."""
    receipt = _read(root / RECEIPT)
    if not isinstance(receipt, dict) or receipt.get("phase") == "completed":
        return None
    phase, failure = str(receipt.get("phase")), receipt.get("failure")
    why = f"{RECEIPT}: {phase}" + (f", {failure}" if failure else "")
    if isinstance(receipt.get("note"), str) and receipt["note"]:
        why += f" ({receipt['note']})"
    if failure == "worker_crash" and any(len(w.incarnations) > 1 or w.crashes for w in workers):
        return Outcome.WORKER_REPLACED, why
    if failure:
        return RECEIPT_FAILURES.get(str(failure), Outcome.UNKNOWN), why
    # no failure code: a setup that failed (legion init, the launch), or an attempt not settled
    return (Outcome.HARNESS_ERROR if phase == "setup_failed" else Outcome.UNKNOWN), why


def _settle(result: RunResult, run_reasons: list[tuple[Outcome, str]]) -> RunResult:
    """The run's outcome: completed only when every agent is and nothing else is wrong. A sweep
    receipt's failure is evidence too: an agent without any of its own takes it."""
    reasons = list(run_reasons)
    if (receipt := _receipt(result.root, result.workers)) is not None:
        reasons.append(receipt)
        for a in result.agents:
            if a.outcome is Outcome.UNKNOWN:
                a.outcome = receipt[0]
    for a in result.agents:
        if a.outcome.completed:
            continue
        last = a.trials[-1] if a.trials else None
        if a.workers and a.outcome in (Outcome.PREEMPTED, Outcome.WORKER_REPLACED,
                                       Outcome.WORKER_LOST):
            why = f"worker {', '.join(a.workers)}"
        elif last is not None and last.outcome is a.outcome:
            why = f"Trial {last.name}" + (f" ({last.exception_type})" if last.exception_type else "")
        else:  # no evidence of its own: none at all, or the receipt's
            why = "no Trials" if last is None else f"Trial {last.name} unverified"
            why += f"; see {RECEIPT}" if receipt is not None and a.outcome is receipt[0] else ""
        reasons.append((a.outcome, f"{a.name}: {a.outcome.value}, {why}"))
    if result.planned_agents is not None and len(result.agents) < result.planned_agents:
        why = f"{len(result.agents)} of the {result.planned_agents} planned agents found"
        reasons.append((Outcome.UNKNOWN, why))
    if not result.agents:
        reasons.append((Outcome.UNKNOWN, "no agents"))
    reasons.sort(key=lambda r: _ORDER.index(r[0]))
    result.outcome = reasons[0][0] if reasons else Outcome.COMPLETED
    result.reasons = [why for _, why in reasons]
    return result


def _exports(root: Path, folder: Path, worker: str | None) -> list[ResourceExport]:
    """The files under `folder/<resource>/`."""
    return [ResourceExport(worker=worker, resource=p.relative_to(folder).parts[0],
                           path=p.relative_to(root))
            for p in sorted(folder.rglob("*")) if p.is_file() and len(p.relative_to(folder).parts) > 1]


def _identity(swarm: str, root: Path) -> dict:
    """A sweep attempt's run dir holds its receipt: its `run_id` and `attempt` when it records
    them, else the attempt's `-rN` name. Without one, the swarm's name is the run's, `-rN` or
    not: a standalone run may be named that way."""
    receipt = _read(root / RECEIPT)
    run_id, attempt = swarm, 0
    if isinstance(receipt, dict):
        if m := _ATTEMPT.fullmatch(swarm):
            run_id, attempt = m[1], int(m[2])
        if isinstance(receipt.get("run_id"), str) and receipt["run_id"]:
            run_id = receipt["run_id"]
        if type(receipt.get("attempt")) is int:
            attempt = receipt["attempt"]
    return {"swarm": swarm, "attempt_id": root.name, "run_id": run_id, "attempt": attempt}


# --------------------------------------------------------------------------- local


def _local(root: Path, run: dict) -> RunResult:
    spec = run.get("spec") or {}
    types = spec.get("types") or {}
    config = next(((t.get("harbor") or {}).get("config") or {}
                   for t in types.values() if t.get("harbor")), {})
    harbor = root / "agents" / "_harbor"
    trials = _trials(root, harbor)
    records = _incarnations(harbor / WORKERS)
    workers = {}
    for w in run.get("workers") or []:
        workers[w["id"]] = WorkerResult(id=w["id"], crashes=int(w.get("crashes") or 0),
                                        incarnations=records.pop(w["id"], []))
    for wid, incs in records.items():
        workers[wid] = WorkerResult(id=wid, incarnations=incs)
    placed = {a["name"]: a.get("worker") for a in run.get("agents") or []}
    for name, wid in placed.items():
        if wid in workers:
            workers[wid].agents.append(name)
    for w in workers.values():  # the records name agents a run.json may have moved since
        w.agents = sorted(set(w.agents) | {a for i in w.incarnations for a in i.agents})
    names = sorted(set(placed) | set(trials))
    replicas = [int(t.get("replicas") or 0) for t in types.values()]
    planned = sum(replicas) if replicas and all(replicas) else None  # 0: a spawned type
    started, ended = _time(run.get("started_at")), _time(run.get("ended_at"))
    result = RunResult(
        **_identity(str(run.get("swarm") or root.name), root), compute="local", root=root,
        outcome=Outcome.UNKNOWN, started_at=started, ended_at=ended,
        duration_s=run.get("duration_s"), stop_reason=run.get("stop_reason"),
        limits={k: v for k, v in (spec.get("limits") or {}).items() if v is not None}
        | (config.get("limits") or {}),
        task={k: v for k, v in (config.get("task") or {}).items() if v is not None},
        planned_agents=planned,
        agents=[_agent(n, trials.get(n, []), list(workers.values())) for n in names],
        workers=sorted(workers.values(), key=lambda w: w.id),
        exports=_exports(root, harbor / "_worker" / "resources", None))  # legion.worker.harbor
    reasons = []
    if ended is None:
        reasons.append((Outcome.UNKNOWN, "the run has not ended"))
    if run.get("stop_reason") == "error":
        reasons.append((Outcome.UNKNOWN, "the run stopped on an error"))
    return _settle(result, reasons)


# --------------------------------------------------------------------------- slurm


def _slurm(root: Path) -> RunResult:
    # deferred: Spec imports Harbor, which the `legion` CLI does not load at start
    from legion.harbor.spec import Spec

    reasons: list[tuple[Outcome, str]] = []
    spec = None
    env: dict[str, str] = {}
    try:
        for name in ("cluster.env", "run.env"):  # later wins, as `legion --run` reads them
            if (root / name).is_file():
                env |= read_env_file(root / name)
        spec = Spec.load(root / "experiment" / "spec.yaml", env)
    except ValidationError as e:  # where and which check: its message may quote the input,
        err = e.errors(include_url=False, include_input=False)[0]  # a resolved secret
        reasons.append((Outcome.UNKNOWN, f"spec: {'.'.join(map(str, err['loc']))}: {err['type']}"))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as e:
        why = safe_error(e, [v for v in env.values() if len(v) >= 4]).splitlines()[0]
        reasons.append((Outcome.UNKNOWN, f"spec: {why}"))
    results = root / "results"
    swarms = sorted(p.name for p in results.iterdir() if p.is_dir()) if results.is_dir() else []
    swarm = spec.name if spec and spec.name else swarms[0] if len(swarms) == 1 else root.name
    base = results / swarm
    trials = _trials(root, base)
    records = _incarnations(base / WORKERS)
    # A launch env file is written before sbatch, also for a submission that fails: only a run
    # without records (older) counts its launches from them
    launches: dict[str, list[str]] = {}
    for path in [] if records else sorted((root / "launch").glob("*.env")):
        if m := _LAUNCH.fullmatch(path.name):
            launches.setdefault(m[1], []).append(m[2])
    workers = {}
    for wid in sorted(set(records) | set(launches)):
        incs = records.get(wid, [])
        known = {i.id for i in incs}
        incs += [Incarnation(id=i, recorded=False) for i in launches.get(wid, []) if i not in known]
        workers[wid] = WorkerResult(id=wid, incarnations=incs,
                                    agents=sorted({a for i in incs for a in i.agents}))
    for w in workers.values():  # an older run: no records say which agents it held
        if not w.agents and (outcome := w.outcome()) is not None:
            why = f"worker {w.id}: {outcome.value} ({len(w.incarnations)} launches), agents unknown"
            reasons.append((outcome, why))
    planned = names = None
    if spec is not None:
        planned = None if spec.spawn else sum(r.count for r in spec.roles().values())
        if spec.naming == "consecutive-id" and planned is not None:
            names = [f"agent-{k:03d}" for k in range(1, planned + 1)]
    names = sorted(set(names or []) | set(trials) | {a for w in workers.values() for a in w.agents})
    times = [t for ts in trials.values() for x in ts for t in (x.started_at, x.finished_at) if t]
    times += [t for w in workers.values() for i in w.incarnations
              for t in (i.started_at, i.ended_at) if t]
    started, ended = (min(times), max(times)) if times else (None, None)
    result = RunResult(
        **_identity(swarm, root), compute="slurm", root=root, outcome=Outcome.UNKNOWN,
        started_at=started, ended_at=ended,
        duration_s=round((ended - started).total_seconds(), 1) if times else None,
        limits=spec.limits.model_dump(mode="json") if spec else {},
        task=spec.task.model_dump(mode="json", exclude_none=True) if spec else {},
        planned_agents=planned,
        agents=[_agent(n, trials.get(n, []), list(workers.values())) for n in names],
        workers=list(workers.values()),
        exports=[e for d in sorted(base.glob("*/resources")) if not d.parent.name.startswith("_")
                 for e in _exports(root, d, d.parent.name)])
    return _settle(result, reasons)
