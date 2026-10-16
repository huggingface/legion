"""Shared contracts between controller, worker and CLI.

This file is the contract. Everything in docs/design.md section 5 ("Contracts")
is expressed here. Change it deliberately and update the design doc.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- durations

_DURATION_RE = re.compile(r"^\s*(\d+)\s*(ms|s|m|h|d)\s*$")
_UNIT_SECONDS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(value: str | int | float | timedelta) -> timedelta:
    """Parse '30s', '10m', '24h', '7d' (or a number of seconds) into a timedelta."""
    if isinstance(value, timedelta):
        return value
    if isinstance(value, (int, float)):
        return timedelta(seconds=float(value))
    m = _DURATION_RE.match(value)
    if not m:
        raise ValueError(f"bad duration {value!r}; use e.g. 30s, 10m, 24h, 7d")
    return timedelta(seconds=int(m.group(1)) * _UNIT_SECONDS[m.group(2)])


# --------------------------------------------------------------------------- enums


class Harness(StrEnum):
    claude = "claude"
    codex = "codex"


class AgentState(StrEnum):
    working = "working"  # a harness process is running a turn
    idle = "idle"  # no process; idle_since set
    stuck = "stuck"  # a turn exceeded its timeout and was killed; cleared on next turn
    unknown = "unknown"  # worker heartbeat older than 90 s
    dead = "dead"  # heartbeat older than 180 s, or job terminal


class WorkerState(StrEnum):
    launching = "launching"
    running = "running"
    draining = "draining"
    unknown = "unknown"
    dead = "dead"


class TypeHealth(StrEnum):
    ok = "ok"
    crashlooping = "crashlooping"


# --------------------------------------------------------------------------- spec

AGENT_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*-\d{4,}$")
TYPE_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")


class TypeSpec(BaseModel):
    """One agent type inside a swarm. See design.md 'Spec'."""

    harness: Harness
    model: str
    image: str | None = None  # None -> default worker image
    system_prompt_file: str | None = None  # written into the workdir as CLAUDE.md / AGENTS.md
    # The file's contents, inlined by `flock apply` (the controller and workers cannot
    # read the caller's files). Wins over system_prompt_file on the worker.
    system_prompt: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    secrets: list[str] = Field(default_factory=list)  # names of secrets the job receives
    labels: dict[str, str] = Field(default_factory=dict)
    replicas: int = 0
    agents_per_worker: int = 50  # measured default, experiments/packing
    max_working_per_worker: int = 20  # measured default, experiments/packing
    # Worker slots for this type, declared only (docs/spawn.md). Required for a type that
    # `spawn:` lets others create; None keeps v1's placement.
    workers: WorkersSpec | None = None
    flavor: str = "cpu-basic"
    worker_lifetime: str = "24h"
    lifetime_jitter: str = "4h"
    drain_window: str = "10m"
    idle_grace: str = "120s"
    turn_timeout: str = "30m"
    max_turns: int | None = None  # per prompt, passed to the harness
    max_budget_usd: float | None = None  # per prompt, passed to the harness
    persist_workdir: bool = True
    resume_prompt: str | None = None
    msgboard_invite: str | None = None  # name of the secret holding the invite code

    @field_validator(
        "worker_lifetime", "lifetime_jitter", "drain_window", "idle_grace", "turn_timeout"
    )
    @classmethod
    def _valid_duration(cls, v: str) -> str:
        parse_duration(v)
        return v

    @field_validator("replicas", "agents_per_worker", "max_working_per_worker")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return v


class SwarmSpec(BaseModel):
    swarm: str
    namespace: str  # HF namespace jobs are launched under
    bucket: str  # e.g. hf://buckets/cmpatino/flock
    types: dict[str, TypeSpec]

    @field_validator("swarm")
    @classmethod
    def _valid_swarm(cls, v: str) -> str:
        if not TYPE_NAME_RE.match(v):
            raise ValueError("swarm name must be lowercase [a-z0-9-], starting with a letter")
        return v

    # Run policy for spawned agents (docs/spawn.md), separate from what a type is. A type
    # absent from `spawn` cannot spawn.
    spawn: dict[str, SpawnRule] = Field(default_factory=dict)
    limits: LimitsSpec = Field(default_factory=lambda: LimitsSpec())

    @field_validator("spawn")
    @classmethod
    def _valid_spawn(cls, v: dict[str, SpawnRule], info) -> dict[str, SpawnRule]:
        types = info.data.get("types")
        if types is not None:
            validate_spawn(v, types)
        return v

    @field_validator("types")
    @classmethod
    def _valid_types(cls, v: dict[str, TypeSpec]) -> dict[str, TypeSpec]:
        for name in v:
            if not TYPE_NAME_RE.match(name):
                raise ValueError(f"type name {name!r} must be lowercase [a-z0-9-]")
        return v


def agent_name(type_name: str, slot: int) -> str:
    return f"{type_name}-{slot:04d}"


# --------------------------------------------------------------------------- spawn


class WorkersSpec(BaseModel):
    """Worker slots a type may use: `initial` (default: what v1 computes from replicas and
    agents_per_worker) plus up to `max_extra` opened on demand for spawned agents."""

    initial: int | None = None
    max_extra: int = 0

    @field_validator("initial", "max_extra")
    @classmethod
    def _non_negative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("must be >= 0")
        return v


class SpawnRule(BaseModel):
    """What an agent of one type may spawn. `types` is a list (with one total `max`) or a map
    of per-type caps (`max` then an optional total cap). Caps count the parent's existing
    children. `depth` is the generations allowed below the agent (0: cannot spawn). Unknown
    keys are rejected, so a misspelled or unsupported policy field never passes silently."""

    model_config = {"extra": "forbid"}

    types: list[str] | dict[str, int]
    max: int | None = None
    depth: int = 1

    @field_validator("max", "depth")
    @classmethod
    def _non_negative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("must be >= 0")
        return v

    @field_validator("types")
    @classmethod
    def _valid_caps(cls, v: list[str] | dict[str, int]) -> list[str] | dict[str, int]:
        if isinstance(v, dict) and any(n < 0 for n in v.values()):
            raise ValueError("per-type caps must be >= 0")
        return v

    @property
    def allowed(self) -> list[str]:
        return list(self.types)

    def cap_for(self, type_name: str) -> int | None:
        return self.types.get(type_name) if isinstance(self.types, dict) else None


class LimitsSpec(BaseModel):
    """Swarm-wide limits checked when an agent spawns (None: no limit). `alive` counts agents
    not removed, replicas included; `spawned_total` the spawn counter; `budget_usd` the cost
    of every agent in the swarm, removed ones included."""

    alive: int | None = None
    spawned_total: int | None = None
    budget_usd: float | None = None

    @field_validator("alive", "spawned_total", "budget_usd")
    @classmethod
    def _non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("must be >= 0")
        return v


def validate_spawn(spawn: dict[str, SpawnRule], types: dict[str, TypeSpec]) -> None:
    """Apply-time checks: known types, no cycle, `workers` on every spawnable type, a total
    `max` for the list form."""
    for parent, rule in spawn.items():
        if parent not in types:
            raise ValueError(f"spawn.{parent}: unknown type")
        if isinstance(rule.types, list) and rule.max is None:
            raise ValueError(f"spawn.{parent}: a list of types needs `max`")
        for child in rule.allowed:
            if child not in types:
                raise ValueError(f"spawn.{parent}.types: unknown type {child!r}")
            if types[child].workers is None:
                raise ValueError(f"types.{child}: spawnable by {parent!r}, so it needs `workers`")
    state: dict[str, int] = {}  # 1 on the DFS stack, 2 done

    def visit(t: str, path: list[str]) -> None:
        state[t] = 1
        for child in spawn[t].allowed if t in spawn else []:
            if state.get(child) == 1:
                cycle = " -> ".join([*path[path.index(child):], child])
                raise ValueError(f"spawn: cycle {cycle}")
            if child not in state:
                visit(child, [*path, child])
        state[t] = 2

    for t in sorted(spawn):
        if t not in state:
            visit(t, [t])


def child_name(parent: str, type_name: str, n: int) -> str:
    return f"{parent}-{type_name}-{n:04d}"


def initial_workers(ts: TypeSpec) -> int:
    """v1's worker count for a type's replicas, or the declared `workers.initial`."""
    if ts.workers is not None and ts.workers.initial is not None:
        return ts.workers.initial
    return -(-ts.replicas // max(1, ts.agents_per_worker))


def max_workers(ts: TypeSpec) -> int:
    return initial_workers(ts) + (ts.workers.max_extra if ts.workers else 0)


def root_depth(spec: SwarmSpec, type_name: str) -> int:
    """Spawn depth of a replica: its type's `spawn.depth`, 0 if the type cannot spawn."""
    rule = spec.spawn.get(type_name)
    return rule.depth if rule else 0


def child_depth(spec: SwarmSpec, parent_depth: int, type_name: str) -> int:
    return max(0, min(parent_depth - 1, root_depth(spec, type_name)))


TypeSpec.model_rebuild()
SwarmSpec.model_rebuild()


# --------------------------------------------------------------------------- usage


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
            reasoning=self.reasoning + other.reasoning,
        )


# --------------------------------------------------------------------------- worker -> controller


class AgentHeartbeat(BaseModel):
    """One record per agent inside a worker heartbeat (~100 bytes)."""

    name: str
    state: Literal["working", "idle", "stuck"]
    turn_id: str | None = None
    idle_since: datetime | None = None
    session_id: str | None = None
    resumed: bool | None = None  # None until the first (re)start on this worker
    tokens: dict[str, TokenUsage] = Field(default_factory=dict)  # by model, cumulative
    harness_estimate_usd: float | None = None  # Claude's client-side total, stored not trusted
    turns: int = 0
    last_sync: datetime | None = None
    msgboard_principal: str | None = None


class WorkerHeartbeat(BaseModel):
    worker: str
    deadline: datetime
    draining: bool = False
    agents: list[AgentHeartbeat]


class Event(BaseModel):
    """One line of an agent's event stream.

    The on-disk / over-the-wire shape is pi session format v3 (see swarmlab's
    export.py for the reference encoding). `type`, `ts`, `agent` are required by us;
    the rest of the pi record lives in `data`. Harness-neutral types the worker adds:
    session_started, session_resumed, session_restarted, turn_started, turn_ended,
    turn_interrupted, sync.
    """

    type: str
    ts: datetime
    agent: str
    turn_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class EventBatch(BaseModel):
    worker: str
    events: list[Event]


# --------------------------------------------------------------------------- controller -> worker


class PromptCommand(BaseModel):
    kind: Literal["prompt"] = "prompt"
    agent: str
    prompt_id: str
    text: str
    timeout: str | None = None
    reply_to: str | None = None  # opaque, for the caller's bookkeeping


class StopCommand(BaseModel):
    kind: Literal["stop"] = "stop"
    agent: str  # kill the agent's harness process if any; agent stays assigned


class DrainCommand(BaseModel):
    kind: Literal["drain"] = "drain"  # worker-wide: finish in-flight turns, sync, exit 0


class AssignCommand(BaseModel):
    """Assign an agent to this worker; the worker restores it from the bucket if present."""

    kind: Literal["assign"] = "assign"
    agent: str
    type: str
    spec: TypeSpec
    restore: bool = True  # restore workdir + session from bucket before first prompt
    resume_prompt: str | None = None  # sent once after restore when a turn was interrupted
    parent: str | None = None  # the spawning agent, None for a replica
    # The agent's scoped controller token, injected into its harness env as FLOCK_AGENT_TOKEN.
    agent_token: str | None = None


class UnassignCommand(BaseModel):
    kind: Literal["unassign"] = "unassign"
    agent: str  # stop, final sync, forget


Command = PromptCommand | StopCommand | DrainCommand | AssignCommand | UnassignCommand


class CommandBatch(BaseModel):
    commands: list[Command]


# --------------------------------------------------------------------------- API shapes


class PromptRequest(BaseModel):
    text: str
    timeout: str | None = None
    reply_to: str | None = None


class PromptAck(BaseModel):
    prompt_id: str
    agent: str
    status: Literal["queued", "pending"]  # queued: worker is up; pending: held until a worker is
    turn_id: str | None = None  # set later via the event stream, may be None here


class SpawnRequest(BaseModel):
    type: str
    prompt: str | None = None  # first prompt, held until the child is placed
    key: str | None = None  # idempotency key, unique per parent


class SpawnAck(BaseModel):
    name: str
    created: bool
    parent: str
    type: str
    depth: int


class AgentStatus(BaseModel):
    name: str
    swarm: str
    type: str
    harness: Harness
    model: str
    labels: dict[str, str]
    state: AgentState
    idle_since: datetime | None = None
    turn_id: str | None = None
    worker: str | None = None
    session_id: str | None = None
    resumed: bool | None = None
    pending: int = 0
    tokens: dict[str, TokenUsage] = Field(default_factory=dict)
    cost_usd: float = 0.0  # derived by the controller from its price table
    harness_estimate_usd: float | None = None
    turns: int = 0
    created_at: datetime
    last_heartbeat: datetime | None = None
    last_sync: datetime | None = None
    msgboard_principal: str | None = None
    parent: str | None = None  # spawning agent; None for a replica
    children: list[str] = Field(default_factory=list)
    depth: int = 0  # generations this agent may still spawn below itself


class WorkerStatus(BaseModel):
    id: str
    swarm: str
    type: str
    job_id: str | None
    state: WorkerState
    deadline: datetime | None
    agents: list[str]
    last_heartbeat: datetime | None
    launched_at: datetime
    crashes: int = 0


class TypeStatus(BaseModel):
    name: str
    health: TypeHealth
    replicas: int
    agents_by_state: dict[str, int]
    workers_by_state: dict[str, int]
    cost_usd: float = 0.0


class SwarmStatus(BaseModel):
    swarm: str
    namespace: str
    bucket: str
    types: dict[str, TypeStatus]
    agents_total: int
    workers_total: int
    pending_prompts: int
    cost_usd: float = 0.0


# --------------------------------------------------------------------------- constants

HEARTBEAT_INTERVAL = timedelta(seconds=30)
UNKNOWN_AFTER = timedelta(seconds=90)
DEAD_AFTER = timedelta(seconds=180)
CRASHLOOP_THRESHOLD = 5
LAUNCHES_PER_WINDOW = 400  # per 5-minute Hub rate-limit window, PRO assumption
RATE_WINDOW = timedelta(minutes=5)
JOB_LABELS = ("swarm", "type", "worker")  # every worker job carries these labels
