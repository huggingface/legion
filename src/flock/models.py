"""Shared contracts between controller, worker and CLI.

This file is the contract. Everything in docs/design.md section 5 ("Contracts")
is expressed here. Change it deliberately and update the design doc.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator

# --------------------------------------------------------------------------- spec YAML


class _SpecLoader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 booleans: only true/false. YAML 1.1 (PyYAML's default) also
    reads on/off/yes/no as booleans, so the documented `on: [comms.mention]` key became `True`.
    Dropping them in the loader protects every key and value of a spec, not just `loop.on`."""


_BOOL = "tag:yaml.org,2002:bool"
_SpecLoader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for tag, rx in rs if tag != _BOOL]
    for ch, rs in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_SpecLoader.add_implicit_resolver(_BOOL, re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
                                  list("tTfF"))


def load_spec_yaml(text: str) -> Any:
    """Parse spec YAML (safe, YAML 1.2 booleans); validate with `SwarmSpec.model_validate`."""
    return yaml.load(text, Loader=_SpecLoader)  # a SafeLoader subclass

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
    pi = "pi"
    llm = "llm"  # in-process chat-completions loop (docs/llm-harness.md)


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
    launch_refused = "launch_refused"  # the backend refused the launch (4xx); waits for a new apply


# --------------------------------------------------------------------------- spec

AGENT_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*-\d{4,}$")
TYPE_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")
# Provider names prefix model strings ("hf:Qwen/...") and become Codex `model_providers.<name>`
# keys, so no dots and no colons. A model string whose prefix does not look like this
# (e.g. Bedrock's "anthropic.claude-...-v1:0") is a harness-native id, not a provider reference.
PROVIDER_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TOOLS_VERSION_RE = re.compile(r"^[0-9a-f]{16}$")
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
    # Max agents of this swarm on this provider that may be in a turn at once. Enforced at
    # prompt admission by the controller (FLOCK_MAX_WORKING_ENFORCE=0: report only).
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
    # Raw request-body fields (swarmlab's `extra`). Sent by the `llm` harness only: Codex,
    # Claude Code and Pi cannot add body fields, so for them it is kept for spec portability.
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


# --------------------------------------------------------------------------- comms (docs/comms.md)

# Built-in comms tools a type may list in `tools:`; the model sees the bare name (`post`).
COMMS_TOOLS = ("comms:post", "comms:read", "comms:search")
COMMS_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def normalize_board_url(url: str) -> str:
    """The one spelling of a board url, used wherever board urls are stored or compared (spec,
    identities, board clients, message sources): scheme and host lowercased, default port and
    trailing slashes dropped, IPv6 brackets and userinfo kept, so it is idempotent.
    `http://Board.invalid:80/` -> `http://board.invalid`; `http://[::1]:8080/` -> `http://[::1]:8080`."""
    from urllib.parse import urlsplit, urlunsplit

    u = urlsplit(url.strip())
    scheme, host = u.scheme.lower(), (u.hostname or "").lower()
    if ":" in host:  # an IPv6 literal keeps its brackets: `http://[::1]:8080`
        host = f"[{host}]"
    port = u.port
    netloc = host if port is None or (scheme, port) in (("http", 80), ("https", 443)) else f"{host}:{port}"
    if u.username or u.password:
        netloc = u.netloc.rsplit("@", 1)[0] + "@" + netloc
    return urlunsplit((scheme, netloc, u.path.rstrip("/"), u.query, ""))


def comms_name(entry: str) -> str:
    """`comms:post` -> `post`, the name the model sees."""
    return entry.split(":", 1)[1]


class CommsRules(BaseModel):
    """Enforced by the worker before it calls the board; a refused post is a tool result
    ("refused: ..."), never a board request. Live: a re-apply reaches workers by heartbeat."""

    model_config = {"extra": "forbid"}

    min_post_gap: str | None = None  # per agent, e.g. 20s
    max_posts: int | None = None  # per agent, for the run (kept in the agent's synced state)
    max_len: int | None = None  # characters per post
    threads: Literal["per_topic"] | None = None  # posts must say thread: new or a root id
    no_group_mentions: bool = False  # refuse @all / @agents / @humans (and @everyone, @people)

    @field_validator("min_post_gap")
    @classmethod
    def _gap(cls, v: str | None) -> str | None:
        if v is not None:
            parse_duration(v)
        return v

    @field_validator("max_posts", "max_len")
    @classmethod
    def _positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("must be >= 1")
        return v

    @property
    def gap_s(self) -> float:
        return parse_duration(self.min_post_gap).total_seconds() if self.min_post_gap else 0.0


class CommsSpec(BaseModel):
    """Swarm-level `comms:` (docs/comms.md). `url`, `invite_env` and `channel` are resolved
    once by the controller at apply (channel name -> `channel_id`); `kickoff_file` and
    `protocols` are files next to the spec that `flock apply` inlines (`kickoff`,
    `protocols_text`), because the controller and workers cannot read the caller's files."""

    model_config = {"extra": "forbid"}

    kind: Literal["msgboard"] = "msgboard"
    url: str
    invite_env: str  # secret holding a multi-use agent invite (the code never goes in the spec)
    channel: str  # name, with or without `#`
    kickoff_file: str | None = None
    kickoff: str | None = None  # the file's text, inlined by `flock apply` (or given inline)
    rules: CommsRules = Field(default_factory=CommsRules)
    protocols: list[str] = Field(default_factory=list)  # files appended to the instructions
    protocols_text: list[str] = Field(default_factory=list)  # inlined by `flock apply`
    wake: list[Literal["mention", "thread"]] = Field(default_factory=lambda: ["mention", "thread"])
    search: Literal["board", "index"] | None = "board"
    # Written by the controller at apply; anything given by hand is replaced.
    channel_id: int | None = None
    kickoff_id: int | None = None

    @field_validator("url")
    @classmethod
    def _url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"url {v!r} must start with http:// or https://")
        return normalize_board_url(v)

    @field_validator("invite_env")
    @classmethod
    def _env(cls, v: str) -> str:
        if not ENV_NAME_RE.match(v):
            raise ValueError(f"invite_env {v!r} must be an env var name (the invite never goes in the spec)")
        return v

    @field_validator("channel")
    @classmethod
    def _channel(cls, v: str) -> str:
        v = v.strip().lstrip("#")
        if not COMMS_NAME_RE.match(v):
            raise ValueError(f"channel {v!r}: use the channel's name, e.g. prj-eliza-v1")
        return v

    @field_validator("search")
    @classmethod
    def _search(cls, v: str | None) -> str | None:
        if v == "index":
            raise ValueError("search: index (a per-worker FTS5 index) is not built yet; use "
                             "search: board (docs/comms.md, cut 2)")
        return v

    @field_validator("wake")
    @classmethod
    def _wake(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))


def comms_tools(ts: TypeSpec, comms: CommsSpec | None) -> list[str]:
    """The built-in comms tools a type gets: the `comms:*` entries of its `tools`, else all of
    them with `comms: true` (search only when the swarm's `comms.search` is set)."""
    if comms is None or ts.comms is False:
        return []
    listed = [t for t in ts.tools if t in COMMS_TOOLS] if isinstance(ts.tools, list) else []
    if listed:
        return list(dict.fromkeys(listed))
    if not ts.comms:
        return []
    return [t for t in COMMS_TOOLS if t != "comms:search" or comms.search]


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
    # Worker slots for this type, declared only (docs/spawn.md). Required for a type that
    # `spawn:` lets others create; None keeps v1's placement.
    workers: WorkersSpec | None = None
    # Spawned agents of this type live for one session (docs/spawn.md "Ephemeral types"): the
    # controller removes one once a heartbeat reports it idle after a turn with nothing pending
    # (done, max_turns, budget or an error alike), and its worker deletes its files. Spawn-only.
    ephemeral: bool = False
    flavor: str = "cpu-basic"
    worker_lifetime: str = "24h"
    lifetime_jitter: str = "4h"
    drain_window: str = "10m"
    idle_grace: str = "120s"
    turn_timeout: str = "30m"
    max_turns: int | None = None  # per prompt, passed to the harness
    max_budget_usd: float | None = None  # per prompt, passed to the harness
    # harness llm only: transcript size (estimated tokens) that triggers compaction; None: 32k
    context_tokens: int | None = None
    # harness llm only: chat-completions `tool_choice` on every call: auto (default) | required |
    # none | bash (force that tool). A forcing value makes each prompt one action.
    tool_choice: Literal["auto", "required", "none", "bash"] | None = None
    # harness llm only: offer a `done` tool that ends the turn; None: on when tool_choice is
    # required. With it a forced turn runs several steps; without it, one action per prompt.
    done_tool: bool | None = None
    # harness llm only: starting chars/token for the size estimate (calibrated online from the
    # server's prompt_tokens); None: 2.5. And the cap on stored tool output; None: 8000 chars.
    chars_per_token: float | None = None
    tool_output_chars: int | None = None
    persist_workdir: bool = True
    resume_prompt: str | None = None
    msgboard_invite: str | None = None  # name of the secret holding the invite code
    # Tools from the swarm's tools folder (docs/tools.md): a list of function names, or "all".
    # `flock apply` refuses a name the folder does not define. None: no folder tools.
    tools: list[str] | Literal["all"] | None = None
    # Re-prompt this type's agents on a cadence, then a final round (docs/loop.md). None: off.
    loop: LoopSpec | None = None
    # Built-in comms tools (docs/comms.md): true gives all of them (`comms:post`, `comms:read`,
    # `comms:search`), or list the ones wanted in `tools`. Needs the swarm-level `comms:`.
    comms: bool | None = None
    # Written by SwarmSpec validation (apply time), never by hand: anything given here is
    # replaced. None means a harness-native model.
    resolved_model: ResolvedModel | None = None

    @property
    def folder_tools(self) -> list[str] | Literal["all"] | None:
        """`tools` without the built-in `comms:*` entries: what the tools folder must define
        (None when nothing is left)."""
        if not isinstance(self.tools, list):
            return self.tools
        names = [t for t in self.tools if not t.startswith("comms:")]
        return names or None

    @field_validator("tools")
    @classmethod
    def _known_comms(cls, v: list[str] | str | None) -> list[str] | str | None:
        if isinstance(v, list):
            bad = [t for t in v if t.startswith("comms:") and t not in COMMS_TOOLS]
            if bad:
                raise ValueError(f"unknown comms tool(s) {', '.join(bad)}; comms tools: "
                                 f"{', '.join(COMMS_TOOLS)}")
        return v

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

    @field_validator("chars_per_token")
    @classmethod
    def _cpt(cls, v: float | None) -> float | None:
        if v is not None and not 0.5 <= v <= 8:
            raise ValueError("chars_per_token must be between 0.5 and 8")
        return v

    @field_validator("tool_output_chars")
    @classmethod
    def _toc(cls, v: int | None) -> int | None:
        if v is not None and v < 200:
            raise ValueError("tool_output_chars must be >= 200")
        return v

    @field_validator("context_tokens")
    @classmethod
    def _context_tokens(cls, v: int | None) -> int | None:
        if v is not None and v < 256:
            raise ValueError("context_tokens must be >= 256")
        return v

    @model_validator(mode="after")
    def _ephemeral_spawn_only(self) -> TypeSpec:
        if self.ephemeral and self.replicas:
            raise ValueError("ephemeral: true is for spawned agents only; set replicas: 0")
        return self

    @field_validator("workers")
    @classmethod
    def _replicas_fit(cls, v: WorkersSpec | None, info) -> WorkersSpec | None:
        """Declared workers are a cap (docs/spawn.md): the replicas must fit in them."""
        replicas, per = info.data.get("replicas"), info.data.get("agents_per_worker")
        if v is None or v.initial is None or replicas is None or per is None:
            return v
        room = (v.initial + v.max_extra) * max(1, per)
        if replicas > room:
            raise ValueError(
                f"replicas {replicas} do not fit in workers.initial + workers.max_extra = "
                f"{v.initial + v.max_extra} workers x agents_per_worker {per} = {room} agents"
            )
        return v


class ComputeSpec(BaseModel):
    """Where workers run (docs/local.md). `hf-jobs` (the default): one HF Job per worker.
    `local`: one `python -m flock.worker.main` subprocess per worker on the controller's machine
    (`flock run`). The spec also accepts the short form `compute: local`."""

    model_config = {"extra": "forbid"}

    kind: Literal["hf-jobs", "local"] = "hf-jobs"


class SwarmSpec(BaseModel):
    swarm: str
    # HF namespace jobs are launched under, and the bucket agents sync to (e.g.
    # hf://buckets/cmpatino/flock). Required for compute hf-jobs; optional for compute local.
    namespace: str = ""
    bucket: str = ""
    providers: dict[str, ProviderSpec] = Field(default_factory=dict)
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    types: dict[str, TypeSpec]
    # The tools folder (docs/tools.md), relative to the spec file; None: `tools/` next to it.
    # Read by `flock apply` only.
    tools_dir: str | None = None
    # Content hash of the tools bundle, written by `flock apply` after uploading the folder
    # (anything given by hand is replaced). Workers fetch that version; None: no tools.
    tools_version: str | None = None
    # The team's message board (docs/comms.md); None: no comms.
    comms: CommsSpec | None = None

    @field_validator("tools_version")
    @classmethod
    def _valid_tools_version(cls, v: str | None) -> str | None:
        if v is not None and not TOOLS_VERSION_RE.match(v):
            raise ValueError("tools_version is written by `flock apply` (16 hex digits)")
        return v

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
            llm_only = ("tool_choice", "done_tool", "context_tokens", "chars_per_token",
                        "tool_output_chars")
            if ts.harness != Harness.llm and any(getattr(ts, f) is not None for f in llm_only):
                raise ValueError(f"types.{type_name}: {', '.join(llm_only)} are for harness llm only")
            if r is None:
                if ts.harness == Harness.llm:
                    raise ValueError(
                        f"types.{type_name}.model: harness 'llm' needs a provider-backed model "
                        f"('<provider>:<model id>' or an alias); {ts.model!r} names no provider")
                continue
            want = {Harness.claude: "anthropic", Harness.codex: "openai_compat",
                    Harness.pi: "openai_compat", Harness.llm: "openai_compat"}[ts.harness]
            if r.provider.type != want:
                raise ValueError(
                    f"types.{type_name}.model: harness {ts.harness.value!r} cannot use provider "
                    f"{r.provider_name!r} (type {r.provider.type}); {ts.harness.value} needs a "
                    f"provider of type {want}")
            p = self.providers[r.provider_name]
            if p.self_hosted and r.model_id not in p.pricing and "*" not in p.pricing:
                p.pricing[r.model_id] = list(SELF_HOSTED_PRICING)  # pin the default
        for type_name, ts in self.types.items():
            cap = ts.loop.while_.agent_usd if ts.loop else None
            if isinstance(cap, dict) and agent_cap(ts) is None:
                raise ValueError(
                    f"types.{type_name}.loop.while.agent_usd: no entry for model {ts.model!r} "
                    f"(keys: {sorted(cap)}); key it by the type's model alias or model id")
        return self

    @model_validator(mode="after")
    def _check_comms(self) -> SwarmSpec:
        """A type that wants comms tools needs the swarm's `comms:` block; a listed
        `comms:search` needs `comms.search`; `comms: false` with listed comms tools is a
        contradiction."""
        for name, ts in self.types.items():
            listed = [t for t in ts.tools if t in COMMS_TOOLS] if isinstance(ts.tools, list) else []
            if (ts.comms or listed) and self.comms is None:
                raise ValueError(f"types.{name}: comms tools need the swarm-level `comms:` block "
                                 "(kind, url, invite_env, channel)")
            if ts.comms is False and listed:
                raise ValueError(f"types.{name}: comms: false but tools lists {', '.join(listed)}")
            if "comms:search" in listed and self.comms is not None and not self.comms.search:
                raise ValueError(f"types.{name}.tools: comms:search needs comms.search (board)")
        c = self.comms
        if c is not None and c.protocols_text and len(c.protocols_text) != len(c.protocols):
            raise ValueError("comms.protocols_text is written by `flock apply` (one per protocols file)")
        return self

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
    compute: ComputeSpec = Field(default_factory=ComputeSpec)

    @field_validator("compute", mode="before")
    @classmethod
    def _compute_short(cls, v: Any) -> Any:
        return {"kind": v} if isinstance(v, str) else v

    @model_validator(mode="after")
    def _job_fields(self) -> SwarmSpec:
        if self.compute.kind == "hf-jobs":
            missing = [f for f in ("namespace", "bucket") if not getattr(self, f)]
            if missing:
                raise ValueError(f"{' and '.join(missing)} required for compute hf-jobs "
                                 "(or set `compute: local`, docs/local.md)")
        return self

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
    of every agent in the swarm, removed ones included. `budget_usd` is also a hard stop
    (docs/loop.md): at `budget_usd` minus the types' `loop.final.reserve_usd` the controller
    stops loops, drops pending prompts and runs the final rounds; at `budget_usd` it stops
    every agent and refuses new prompts."""

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


# --------------------------------------------------------------------------- loop (docs/loop.md)

# Placeholders the worker fills in loop prompts; any other `{name}` is left as written.
LOOP_PLACEHOLDERS = ("round", "elapsed", "wall", "spend", "cap", "why", "last_summary")
NEEDS_WORLD = "needs a world (not in this version)"
_LOOP_SOURCE_RE = re.compile(r"^(comms|world|artifacts)\.[A-Za-z0-9_.:*-]+$")


class LoopWhile(BaseModel):
    """While every set condition holds, the loop goes on. Type-wide: `until` (wall time since
    the loop started), `budget_usd` (the type's cost). Per agent: `max_rounds` (loop prompts),
    `agent_usd` (the agent's cost; a map is keyed by model alias or model id), `idle_rounds`
    (that many loop turns in a row without a tool call). `score_<op>: {metric: value}`
    conditions are kept under `score` and need a world (refused at apply in this version)."""

    model_config = {"extra": "forbid"}

    until: str | None = None
    max_rounds: int | None = None
    budget_usd: float | None = None
    agent_usd: float | dict[str, float] | None = None
    idle_rounds: int | None = None
    score: dict[str, dict[str, float]] = Field(default_factory=dict)  # op -> {metric: value}

    @model_validator(mode="before")
    @classmethod
    def _score_keys(cls, data: Any) -> Any:
        if isinstance(data, dict) and any(str(k).startswith("score_") for k in data):
            data = dict(data)
            score = dict(data.pop("score", None) or {})
            for k in [k for k in data if str(k).startswith("score_")]:
                score[k.removeprefix("score_")] = data.pop(k)
            data["score"] = score
        return data

    @field_validator("until")
    @classmethod
    def _until(cls, v: str | None) -> str | None:
        if v is not None:
            parse_duration(v)
        return v

    @field_validator("max_rounds", "idle_rounds")
    @classmethod
    def _positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("budget_usd")
    @classmethod
    def _usd(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("must be >= 0")
        return v

    @field_validator("agent_usd")
    @classmethod
    def _agent_usd(cls, v: float | dict[str, float] | None) -> float | dict[str, float] | None:
        values = v.values() if isinstance(v, dict) else [] if v is None else [v]
        if any(x < 0 for x in values):
            raise ValueError("must be >= 0")
        return v


class LoopFinal(BaseModel):
    """One last prompt per live agent when the loop stops (any reason but `--no-final`).
    `reserve_usd` is held back from `limits.budget_usd` for it. `add_tools` are tool names the
    final prompt unlocks; carried in the prompt and its events for the tools layer."""

    model_config = {"extra": "forbid"}

    prompt: str | None = None
    prompt_file: str | None = None  # inlined into `prompt` by `flock apply`
    reserve_usd: float = 0.0
    add_tools: list[str] = Field(default_factory=list)

    @field_validator("reserve_usd")
    @classmethod
    def _usd(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    @model_validator(mode="after")
    def _has_prompt(self) -> LoopFinal:
        if self.prompt is None and self.prompt_file is None:
            raise ValueError("final needs prompt or prompt_file")
        return self


class LoopSpec(BaseModel):
    """A type's prompting loop (docs/loop.md). `first` goes alone to each agent; the loop
    prompt follows once that turn ended, then on every `every` (`turn_end`: re-sent by the
    worker when a turn completes; a duration: the controller prompts idle agents that long
    after their last loop prompt; `rounds` needs a world) and on each `on` wake-up, while
    `while` holds. Then `final`, once per live agent, then idle."""

    model_config = {"extra": "forbid", "populate_by_name": True}

    prompt: str | None = None
    prompt_file: str | None = None  # inlined into `prompt` by `flock apply`
    first: str | None = None
    every: str = "turn_end"
    on: list[str] = Field(default_factory=list)  # comms.mention, comms.thread, world.event:*
    context: list[str] = Field(default_factory=list)  # needs comms / artifacts / a world
    stagger: str | None = None  # agent i gets its first prompt i * stagger after the start
    while_: LoopWhile = Field(default_factory=LoopWhile, alias="while")
    final: LoopFinal | None = None

    @model_serializer(mode="wrap")
    def _dump(self, handler):  # `while` on the wire, as written in specs
        out = handler(self)
        if isinstance(out, dict) and "while_" in out:
            out["while"] = out.pop("while_")
        return out

    @field_validator("every")
    @classmethod
    def _every(cls, v: str) -> str:
        if v not in ("turn_end", "rounds") and parse_duration(v).total_seconds() <= 0:
            raise ValueError("every must be turn_end, rounds or a duration > 0")
        return v

    @field_validator("stagger")
    @classmethod
    def _stagger(cls, v: str | None) -> str | None:
        if v is not None:
            parse_duration(v)
        return v

    @field_validator("on", "context")
    @classmethod
    def _sources(cls, v: list[str]) -> list[str]:
        for x in v:
            if not _LOOP_SOURCE_RE.match(x):
                raise ValueError(f"{x!r}: expected comms.<name>, world.<name> or artifacts.<name>")
        return v

    @model_validator(mode="after")
    def _has_prompt(self) -> LoopSpec:
        if self.prompt is None and self.prompt_file is None:
            raise ValueError("loop needs prompt or prompt_file")
        return self

    @property
    def every_s(self) -> float | None:
        """Seconds between loop prompts for a duration `every`, else None."""
        if self.every in ("turn_end", "rounds"):
            return None
        return parse_duration(self.every).total_seconds()


def hit_max_turns(turn_ended: dict) -> bool:
    """A `turn_ended` that stopped at the type's `max_turns` (llm, Pi, Codex: `reason`; Claude:
    `subtype`). Not a failed turn for a loop: with `tool_choice: required` it is how turns end."""
    return turn_ended.get("reason") == "max_turns" or turn_ended.get("subtype") == "error_max_turns"


def agent_cap(ts: TypeSpec) -> float | None:
    """The type's per-agent loop cap (`loop.while.agent_usd`), resolved for its model."""
    cap = ts.loop.while_.agent_usd if ts.loop else None
    if not isinstance(cap, dict):
        return cap
    r = ts.resolved_model
    for key in (ts.model, r.model_id if r else None, r.qualified if r else None):
        if key is not None and key in cap:
            return cap[key]
    return None


def loop_refusals(spec: SwarmSpec) -> list[str]:
    """Loop settings this version accepts in the model but cannot run: world-backed ones, and
    context sources whose feeders are not built yet. `flock apply` and the controller refuse a
    spec with any of them."""
    out: list[str] = []
    for name, ts in spec.types.items():
        lp = ts.loop
        if lp is None:
            continue
        where = f"types.{name}.loop"
        if lp.every == "rounds":
            out.append(f"{where}.every: rounds {NEEDS_WORLD}")
        for op in lp.while_.score:
            out.append(f"{where}.while.score_{op} {NEEDS_WORLD}")
        out += [f"{where}.on: {x} {NEEDS_WORLD}" for x in lp.on if x.startswith("world.")]
        for x in lp.context:
            why = NEEDS_WORLD if x.startswith("world.") else "is not in this version"
            out.append(f"{where}.context: {x} {why}")
        if lp.prompt is None:
            out.append(f"{where}.prompt_file: not inlined (use `flock apply`, which reads it)")
        if lp.final is not None and lp.final.prompt is None:
            out.append(f"{where}.final.prompt_file: not inlined (use `flock apply`, which reads it)")
    return out


CommsSpec.model_rebuild()
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


class LoopMeta(BaseModel):
    """What the worker needs to render a loop prompt (docs/loop.md): it fills the
    LOOP_PLACEHOLDERS from these plus its own spend and last `done` summary."""

    kind: Literal["first", "loop", "final"]
    type: str
    round: int | None = None  # None: the worker numbers it (its own next round)
    started_at: datetime | None = None  # when the loop started, for {elapsed}
    wall_s: float | None = None  # `while.until` in seconds, for {wall}
    cap_usd: float | None = None  # `while.agent_usd` for this agent, for {cap}
    why: str = ""
    add_tools: list[str] = Field(default_factory=list)  # final round: tools it unlocks


class PromptCommand(BaseModel):
    kind: Literal["prompt"] = "prompt"
    agent: str
    prompt_id: str
    text: str
    timeout: str | None = None
    reply_to: str | None = None  # opaque, for the caller's bookkeeping
    loop: LoopMeta | None = None  # a loop prompt: `text` is a template the worker renders

    @model_serializer(mode="wrap")
    def _dump(self, handler):  # plain prompts keep their wire shape
        out = handler(self)
        if isinstance(out, dict) and out.get("loop", 0) is None:
            out.pop("loop")
        return out


class LoopStartCommand(BaseModel):
    """`every: turn_end` on the worker: send `text` (rendered) now if the agent is idle and
    again whenever a turn ends completed, until `max_rounds`, `deadline`, a `loop_stop`, or 3
    failed turns in a row. Idempotent: a repeat updates the settings."""

    kind: Literal["loop_start"] = "loop_start"
    agent: str
    text: str
    meta: LoopMeta  # kind "loop"; `round` is the rounds done so far
    max_rounds: int | None = None
    deadline: datetime | None = None
    # One run of the loop (`flock loop --start` begins a new one): a new run resets the
    # worker's round counter; a repeat of the same run (lost command, live update) does not.
    run: str | None = None


class LoopStopCommand(BaseModel):
    kind: Literal["loop_stop"] = "loop_stop"
    agent: str  # end the worker-side loop; the turn in flight finishes
    reason: str = "stop"


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
    tools_version: str | None = None  # the swarm's tools bundle (docs/tools.md); None: no tools
    comms: CommsSpec | None = None  # the swarm's comms block (docs/comms.md), channel resolved


class UnassignCommand(BaseModel):
    kind: Literal["unassign"] = "unassign"
    agent: str  # stop, final sync, forget


Command = (PromptCommand | StopCommand | DrainCommand | AssignCommand | UnassignCommand
           | LoopStartCommand | LoopStopCommand)


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


class BulkPromptRequest(BaseModel):
    """One prompt to every agent of a type (`POST /swarms/{swarm}/prompts`), optionally only
    those in one state. Each agent gets its own pending prompt, admitted as a single one is."""

    type: str
    state: AgentState | None = None
    text: str
    timeout: str | None = None
    reply_to: str | None = None
    # Skip agents that already have a pending (undelivered) prompt. Default: on with
    # state=idle ("only if idle and empty": repeated calls never stack prompts), off otherwise.
    skip_pending: bool | None = None


class ChildrenPromptRequest(BaseModel):
    """One prompt to an agent's direct children (`POST /agents/{name}/children/prompts`),
    optionally only those of one type and/or state; same rules as BulkPromptRequest. Allowed
    for that agent's own scoped token (and its ancestors'), so a leader prompts its idle
    children in one request."""

    type: str | None = None
    state: AgentState | None = None
    text: str
    timeout: str | None = None
    reply_to: str | None = None
    skip_pending: bool | None = None  # default: on with state=idle


class BulkPromptAck(BaseModel):
    swarm: str
    type: str | None  # None: any type (children route without `type`)
    parent: str | None = None  # set by the children route
    matched: int  # agents of the type
    queued: int  # prompts for agents whose worker is up
    pending: int  # prompts held until the agent's worker is up
    skipped_state: int = 0  # not in the requested state
    skipped_removing: int = 0  # being removed (spawn cascade)
    skipped_pending: int = 0  # already had an undelivered prompt (skip_pending)


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
    last_error: str | None = None  # last launch failure or crash: job status, log tail


class LoopStatus(BaseModel):
    """A type's loop (docs/loop.md). `state`: running, final (final round in flight), done
    (final round over, or stopped without one)."""

    type: str
    state: Literal["running", "final", "done"]
    mode: Literal["worker", "controller"]  # who sends `every: turn_end` prompts
    every: str
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    reason: str | None = None  # why it stopped
    agents: int = 0
    agents_done: int = 0  # out of the loop (max_rounds, agent_usd, idle_rounds, errors)
    rounds_min: int = 0
    rounds_max: int = 0
    rounds_total: int = 0
    final_sent: int = 0
    final_done: int = 0
    per_agent: dict[str, dict[str, Any]] | None = None  # GET /swarms/{s}/loop only


class TypeStatus(BaseModel):
    name: str
    health: TypeHealth
    replicas: int
    agents_by_state: dict[str, int]
    workers_by_state: dict[str, int]
    cost_usd: float = 0.0
    # Last worker-launch error for this type (the backend's message), cleared by a successful
    # launch or a new apply. With health launch_refused it is why launches stopped.
    launch_error: str | None = None
    loop: LoopStatus | None = None


class ProviderStatus(BaseModel):
    name: str
    type: str
    self_hosted: bool
    types: list[str]  # agent types using this provider
    working: int  # agents of those types in a turn now (from heartbeats)
    max_working: int | None = None  # enforced at prompt admission unless FLOCK_MAX_WORKING_ENFORCE=0
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
    # limits.budget_usd as a stop (docs/loop.md): None (under the threshold or no budget),
    # "final_reserve" (loops stopped, final rounds) or "hard" (every agent stopped).
    budget_stop: Literal["final_reserve", "hard"] | None = None


class LoopRequest(BaseModel):
    """`POST /swarms/{swarm}/loop`: stop or (re)start one type's loop."""

    type: str
    action: Literal["start", "stop"]
    final: bool = True  # stop: run the final round (if the type has one)


class PreflightResult(BaseModel):
    """One provider-backed model probed with a tool call and a follow-up turn."""

    model: str  # "<provider>:<model id>"
    provider: str
    types: list[str]
    # "responses" (openai_compat, what Codex speaks) | "messages" (anthropic) |
    # "chat_completions" (openai_compat for llm types)
    api: str
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
