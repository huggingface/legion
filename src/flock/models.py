"""Shared contracts between controller, worker and CLI.

This file is the contract. Everything in docs/design.md section 5 ("Contracts")
is expressed here. Change it deliberately and update the design doc.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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
# Provider names prefix model strings ("hf:Qwen/...") and become Codex `model_providers.<name>`
# keys, so no dots and no colons. A model string whose prefix does not look like this
# (e.g. Bedrock's "anthropic.claude-...-v1:0") is a harness-native id, not a provider reference.
PROVIDER_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SELF_HOSTED_PRICING = [0.0, 0.0, 0.0]

Reasoning = Literal["off", "low", "medium", "high"]
# `reasoning` -> OpenAI `reasoning.effort` / Codex `model_reasoning_effort`.
REASONING_EFFORT = {"off": "none", "low": "low", "medium": "medium", "high": "high"}


def _pricing_row(v: list[float], where: str) -> list[float]:
    if len(v) != 3 or any(x < 0 for x in v):
        raise ValueError(f"{where}: pricing must be [input, output, cached] USD per 1M tokens, >= 0")
    return [float(x) for x in v]


class ProviderSpec(BaseModel):
    """Where a model is served (design.md section 5, 'Providers'). Field names follow swarmlab's
    provider params so a `providers:` block moves between the two projects.

    `api_key_env` names a secret / env var; the value is never in the spec. The controller
    sends that secret to the provider's jobs and uses it for preflight; the worker hands it to
    the harness."""

    type: Literal["openai_compat", "anthropic"]
    base_url: str | None = None  # openai_compat: required, ends in /v1; anthropic: API root
    api_key_env: str
    self_hosted: bool = False  # price defaults to [0, 0, 0]; real cost is the GPU job's time
    # Max agents of this swarm on this provider that may be in a turn at once. Stored and
    # reported in status; not enforced yet (design.md section 5, 'Providers').
    max_working: int | None = None
    pricing: dict[str, list[float]] = Field(default_factory=dict)  # model id ("*" = any) -> row
    timeout_s: float | None = None  # passthrough to the harness / preflight
    max_retries: int | None = None  # passthrough to the harness
    # How the key is sent. `anthropic` providers: "x-api-key" (default, Claude Code's
    # ANTHROPIC_API_KEY) or "bearer" (`Authorization: Bearer`, Claude Code's ANTHROPIC_AUTH_TOKEN),
    # for gateways and proxies that only take a bearer token. `openai_compat` is always bearer.
    auth: Literal["x-api-key", "bearer"] | None = None

    @field_validator("api_key_env")
    @classmethod
    def _env_name(cls, v: str) -> str:
        if not ENV_NAME_RE.match(v):
            raise ValueError(f"api_key_env {v!r} must be an env var name (the key itself never goes in the spec)")
        return v

    @field_validator("pricing")
    @classmethod
    def _rows(cls, v: dict[str, list[float]]) -> dict[str, list[float]]:
        return {k: _pricing_row(row, f"pricing[{k!r}]") for k, row in v.items()}

    @field_validator("max_working")
    @classmethod
    def _positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("max_working must be >= 1")
        return v

    @model_validator(mode="after")
    def _url(self) -> ProviderSpec:
        if self.type == "openai_compat" and not self.base_url:
            raise ValueError("base_url is required for type openai_compat (e.g. https://router.huggingface.co/v1)")
        if self.type == "openai_compat" and self.auth == "x-api-key":
            raise ValueError("auth: x-api-key is for anthropic providers; openai_compat always sends a bearer token")
        if self.base_url is not None:
            self.base_url = self.base_url.rstrip("/")
        return self


class ModelSpec(BaseModel):
    """An alias under `models:`: a provider-prefixed model plus per-model settings."""

    model: str  # "<provider>:<model id>"
    reasoning: Reasoning | None = None
    # Raw request-body fields (swarmlab's `extra`). Kept for spec portability and used by
    # nothing in the harness path yet: neither Codex nor Claude Code can add body fields.
    extra: dict[str, Any] | None = None
    pricing: list[float] | None = None  # overrides the provider's pricing for this alias

    @field_validator("pricing")
    @classmethod
    def _row(cls, v: list[float] | None) -> list[float] | None:
        return None if v is None else _pricing_row(v, "pricing")


class ResolvedModel(BaseModel):
    """A type's model after resolution, pinned into the stored spec at apply time."""

    provider_name: str
    provider: ProviderSpec
    model_id: str  # what the harness sends as `model`
    reasoning: Reasoning | None = None
    extra: dict[str, Any] | None = None
    pricing: list[float]  # [input, output, cached] USD per 1M tokens, pinned

    @property
    def qualified(self) -> str:
        return f"{self.provider_name}:{self.model_id}"


class TypeSpec(BaseModel):
    """One agent type inside a swarm. See design.md 'Spec'."""

    harness: Harness
    # Resolution order (design.md section 5): an alias under `models:`, else "<provider>:<id>"
    # with the prefix under `providers:`, else a harness-native id (v1 behaviour).
    model: str
    reasoning: Reasoning | None = None  # overrides the alias's reasoning
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
    # Written by SwarmSpec validation (apply time), never by hand: anything given here is
    # replaced. None means a harness-native model.
    resolved_model: ResolvedModel | None = None

    @property
    def harness_model(self) -> str:
        """The model id the harness is started with."""
        return self.resolved_model.model_id if self.resolved_model else self.model

    @property
    def provider_name(self) -> str | None:
        return self.resolved_model.provider_name if self.resolved_model else None

    @property
    def effective_reasoning(self) -> Reasoning | None:
        if self.resolved_model is not None:
            return self.resolved_model.reasoning
        return self.reasoning

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
    providers: dict[str, ProviderSpec] = Field(default_factory=dict)
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    types: dict[str, TypeSpec]

    @field_validator("providers")
    @classmethod
    def _valid_providers(cls, v: dict[str, ProviderSpec]) -> dict[str, ProviderSpec]:
        for name in v:
            if not PROVIDER_NAME_RE.match(name):
                raise ValueError(f"providers.{name}: name must be lowercase [a-z0-9_-], starting with a letter")
        return v

    @model_validator(mode="after")
    def _resolve_models(self) -> SwarmSpec:
        """Apply-time checks and pinning: every alias and provider prefix exists, every
        provider-backed model has a price, harness and provider speak the same API. The price
        used is written into the spec (`resolved_model.pricing`, and the self-hosted default
        into the provider's `pricing`), so the stored spec carries what cost is derived from."""
        for alias, m in self.models.items():
            if ":" in alias:
                raise ValueError(f"models.{alias}: an alias cannot contain ':'")
            prefix, sep, model_id = m.model.partition(":")
            if not sep or not prefix or not model_id:
                raise ValueError(f"models.{alias}.model: {m.model!r} must look like '<provider>:<model id>'")
            if prefix not in self.providers:
                raise ValueError(f"models.{alias}.model: unknown provider {prefix!r} in {m.model!r}; "
                                 f"providers: {sorted(self.providers)}")
        for type_name, ts in self.types.items():
            ts.resolved_model = resolve_model(self, ts, type_name=type_name)
            r = ts.resolved_model
            if r is None:
                continue
            want = {Harness.claude: "anthropic", Harness.codex: "openai_compat"}[ts.harness]
            if r.provider.type != want:
                raise ValueError(
                    f"types.{type_name}.model: harness {ts.harness.value!r} cannot use provider "
                    f"{r.provider_name!r} (type {r.provider.type}); {ts.harness.value} needs a "
                    f"provider of type {want}")
            p = self.providers[r.provider_name]
            if p.self_hosted and r.model_id not in p.pricing and "*" not in p.pricing:
                p.pricing[r.model_id] = list(SELF_HOSTED_PRICING)  # pin the default
        return self

    @field_validator("swarm")
    @classmethod
    def _valid_swarm(cls, v: str) -> str:
        if not TYPE_NAME_RE.match(v):
            raise ValueError("swarm name must be lowercase [a-z0-9-], starting with a letter")
        return v

    @field_validator("types")
    @classmethod
    def _valid_types(cls, v: dict[str, TypeSpec]) -> dict[str, TypeSpec]:
        for name in v:
            if not TYPE_NAME_RE.match(name):
                raise ValueError(f"type name {name!r} must be lowercase [a-z0-9-]")
        return v


def resolve_model(spec: SwarmSpec, type_spec: TypeSpec, *, type_name: str = "?") -> ResolvedModel | None:
    """Resolve a type's `model`: alias under `models:` -> "<provider>:<id>" with the prefix
    under `providers:` -> harness-native id (returns None). Raises ValueError naming the type
    and field for an unknown provider prefix or a provider-backed model with no price."""
    where = f"types.{type_name}.model"
    alias = spec.models.get(type_spec.model)
    target = alias.model if alias else type_spec.model
    prefix, sep, model_id = target.partition(":")
    if not sep or not PROVIDER_NAME_RE.match(prefix):
        return None  # harness-native id
    if prefix not in spec.providers:
        raise ValueError(f"{where}: unknown provider {prefix!r} in {target!r}; declare it under "
                         f"providers: (known: {sorted(spec.providers)}) or use a model alias")
    if not model_id:
        raise ValueError(f"{where}: {target!r} has no model id after '{prefix}:'")
    provider = spec.providers[prefix]
    pricing = (alias.pricing if alias else None) or provider.pricing.get(model_id) or provider.pricing.get("*")
    if pricing is None:
        if not provider.self_hosted:
            raise ValueError(
                f"{where}: no pricing for {target!r}; add providers.{prefix}.pricing"
                f"[{model_id!r}] = [input, output, cached] (USD per 1M tokens)"
                + (f" or models.{type_spec.model}.pricing" if alias else "")
                + f", or mark providers.{prefix} self_hosted")
        pricing = list(SELF_HOSTED_PRICING)
    return ResolvedModel(
        provider_name=prefix,
        provider=provider,
        model_id=model_id,
        reasoning=type_spec.reasoning or (alias.reasoning if alias else None),
        extra=alias.extra if alias else None,
        pricing=list(pricing),
    )


def agent_name(type_name: str, slot: int) -> str:
    return f"{type_name}-{slot:04d}"


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


class AgentStatus(BaseModel):
    name: str
    swarm: str
    type: str
    harness: Harness
    model: str
    provider: str | None = None  # provider name when the model is provider-backed
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


class ProviderStatus(BaseModel):
    name: str
    type: str
    self_hosted: bool
    types: list[str]  # agent types using this provider
    working: int  # agents of those types in a turn now (from heartbeats)
    max_working: int | None = None  # reported, not enforced yet
    # Agents with turns but zero reported tokens: the server sends no usage, cost is unknown.
    zero_usage_agents: int = 0


class SwarmStatus(BaseModel):
    swarm: str
    namespace: str
    bucket: str
    types: dict[str, TypeStatus]
    providers: dict[str, ProviderStatus] = Field(default_factory=dict)
    agents_total: int
    workers_total: int
    pending_prompts: int
    cost_usd: float = 0.0


class PreflightResult(BaseModel):
    """One provider-backed model probed with a tool call and a follow-up turn."""

    model: str  # "<provider>:<model id>"
    provider: str
    types: list[str]
    api: str  # "responses" (openai_compat, what Codex speaks) | "messages" (anthropic)
    ok: bool  # a tool call in turn one and a completed follow-up turn carrying its result
    # Not probed (e.g. the key is not in the controller's environment): neither ok nor failed.
    # `apply` warns and proceeds, or refuses with --strict.
    unverified: bool = False
    tool_call_seen: bool = False
    follow_up_ok: bool = False
    usage_reported: bool | None = None  # False: the server streamed zero tokens (cost unknown)
    latency_ms: int | None = None  # both turns
    error: str | None = None
    warning: str | None = None


class PreflightReport(BaseModel):
    swarm: str
    ok: bool  # no probe failed (unverified models do not count as failures here)
    unverified: int = 0  # models that were not probed
    results: list[PreflightResult]


# --------------------------------------------------------------------------- constants

HEARTBEAT_INTERVAL = timedelta(seconds=30)
UNKNOWN_AFTER = timedelta(seconds=90)
DEAD_AFTER = timedelta(seconds=180)
CRASHLOOP_THRESHOLD = 5
LAUNCHES_PER_WINDOW = 400  # per 5-minute Hub rate-limit window, PRO assumption
RATE_WINDOW = timedelta(minutes=5)
JOB_LABELS = ("swarm", "type", "worker")  # every worker job carries these labels
