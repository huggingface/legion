"""Shared contracts between controller, worker and CLI.

This file is the contract. Everything in docs/design.md "Contracts"
is expressed here. Change it deliberately and update the design doc.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal, NamedTuple
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator

from flock.registry import COMPUTE, HARNESSES
from flock.world.protocol import TOOL_NAME_RE as WORLD_TOOL_RE
from flock.world.protocol import WorldManifest

# --------------------------------------------------------------------------- spec YAML


class _SpecLoader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 booleans: only true/false. YAML 1.1 (PyYAML's default) also
    reads on/off/yes/no as booleans, so an `on:` key became `True`.
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


def until_deadline(until: str, started_at: float) -> float:
    """Epoch deadline of `while.until`: started_at + duration, or the ISO-8601 time (naive = UTC)."""
    try:
        return started_at + parse_duration(until).total_seconds()
    except ValueError:
        try:
            t = datetime.fromisoformat(until)
        except ValueError:
            raise ValueError(f"bad until {until!r}; use a duration (2h) or a time (2026-10-28T20:00Z)") from None
        return (t if t.tzinfo else t.replace(tzinfo=UTC)).timestamp()


# --------------------------------------------------------------------------- enums


# the names in flock.registry.HARNESSES; an enum so spec errors and the JSON schema list them
Harness = StrEnum("Harness", [(name, name) for name in HARNESSES], module=__name__)


class AgentState(StrEnum):
    working = "working"  # a harness process is running a turn
    idle = "idle"  # no process; idle_since set
    stuck = "stuck"  # a turn exceeded its timeout and was killed; cleared on next turn
    unknown = "unknown"  # heartbeat older than compute.worker_unknown_after (per-backend default)
    dead = "dead"  # heartbeat older than compute.worker_dead_after, or job terminal


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
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TOOLS_VERSION_RE = re.compile(r"^[0-9a-f]{16}$")
ZERO_PRICING = [0.0, 0.0, 0.0]  # a type without `pricing`: its usage costs $0

Reasoning = Literal["off", "low", "medium", "high"]
# `reasoning` -> OpenAI `reasoning.effort`.
REASONING_EFFORT = {"off": "none", "low": "low", "medium": "medium", "high": "high"}


def _pricing_row(v: list[float], where: str) -> list[float]:
    if len(v) != 3 or any(x < 0 for x in v):
        raise ValueError(f"{where}: pricing must be [input, output, cached] USD per 1M tokens, >= 0")
    return [float(x) for x in v]


# --------------------------------------------------------------------------- inference (docs/open-models.md)

HF_ROUTER_URL = "https://router.huggingface.co/v1"
# `hf_router.provider` values that let the router pick the provider per request
ROUTER_POLICIES = ("fastest", "cheapest", "preferred")
# What a Trial gets for servers that take no key: Pi needs a key reference all the same
NO_KEY_ENV, NO_KEY = "FLOCK_INFERENCE_KEY", "none"
_ROUTER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class HFRouter(BaseModel):
    """`inference.hf_router`: the HF router, sent `<model>:<provider>`."""

    model_config = {"extra": "forbid"}

    # A provider (`deepinfra`), or a selection policy: `fastest` (the router's default),
    # `cheapest` or `preferred` (your order in the Hub settings). None: the router's default.
    provider: str | None = None
    bill_to: str | None = None  # an org the router bills instead of the key's owner (X-HF-Bill-To)

    @field_validator("provider", "bill_to")
    @classmethod
    def _name(cls, v: str | None) -> str | None:
        if v is not None and not _ROUTER_NAME_RE.match(v):
            raise ValueError(f"{v!r} must be a name, e.g. deepinfra or huggingface")
        return v

    @property
    def policy(self) -> bool:
        """The router picks the provider per request: no provider is named."""
        return self.provider is None or self.provider in ROUTER_POLICIES


# `inference.hf_jobs` (docs/open-models.md "Servers flock starts"): the vLLM image flock starts
# (it serves Qwen3_5ForConditionalGeneration), the port the jobs proxy exposes, and the job
# lifetime when neither the spec nor its compile step set one
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:v0.33.0"
VLLM_PORT = 8000
DEFAULT_SERVER_TIMEOUT = "2h"
# `vllm_args` keys flock sets itself: the server's address and name, the model, and the
# tool-calling flags, which are named fields. `api_key`: the jobs proxy drops `Authorization`,
# so vLLM would refuse every request (the proxy's HF-token check is the authentication).
VLLM_OWNED = ("host", "port", "served_model_name", "model", "api_key", "enable_auto_tool_choice",
              "tool_call_parser", "reasoning_parser")
_VLLM_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def vllm_arg_key(entry: str) -> str:
    """`max-model-len=131072` -> `max_model_len`: one spelling per flag."""
    return entry.split("=", 1)[0].strip().replace("-", "_")


class HFJobs(BaseModel):
    """`inference.hf_jobs`: vLLM servers flock starts on HF Jobs, one group per model, and stops
    when the run ends (docs/open-models.md). The servers' URLs are pinned into the spec as
    `urls` (`resolved_model.inference.urls`)."""

    model_config = {"extra": "forbid"}

    flavor: str = Field(min_length=1)  # a GPU flavor (`hf jobs hardware`), e.g. a100-large
    tool_call_parser: str = Field(min_length=1)  # Pi needs tool calls: --enable-auto-tool-choice
    reasoning_parser: str | None = None
    # Every other `vllm serve` flag as KEY=VALUE (`max_model_len=131072` -> --max-model-len
    # 131072; `true`/`false` -> --flag/--no-flag). Fills in max_num_seqs (the block's
    # max_working) and tensor_parallel_size (the flavor's GPUs) unless set here.
    vllm_args: list[str] = Field(default_factory=list)
    # The job's lifetime, the backstop if the run never stops it; None: the run's
    # run_timeout_seconds + startup_timeout + a margin (the compile step writes it)
    timeout: str | None = None
    startup_timeout: str = "20m"  # how long flock waits for every replica to answer
    replicas: int = Field(default=1, ge=1)  # each agent calls one (rendezvous hashing)
    image: str = DEFAULT_VLLM_IMAGE
    namespace: str | None = None  # who pays; None: the spec's namespace, else the token's account

    @field_validator("timeout", "startup_timeout")
    @classmethod
    def _duration(cls, v: str | None) -> str | None:
        return None if v is None else _positive_duration(v)

    @field_validator("vllm_args")
    @classmethod
    def _vllm_args(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        for entry in v:
            key, _, value = entry.partition("=")
            name = vllm_arg_key(entry)
            if not _VLLM_KEY_RE.match(key.strip()) or not value.strip():
                raise ValueError(f"vllm_args {entry!r}: use KEY=VALUE (a flag: KEY=true), e.g. "
                                 "max_model_len=131072")
            if name in VLLM_OWNED:
                named = name in HFJobs.model_fields
                raise ValueError(f"vllm_args {entry!r}: " + (
                    f"set hf_jobs.{name} instead" if named else
                    "the jobs proxy drops Authorization, so vLLM would refuse every request; "
                    "the proxy's HF-token check authenticates" if name == "api_key" else
                    "flock sets it"))
            if name in seen:
                raise ValueError(f"vllm_args: {name} is set twice")
            seen.add(name)
        return v


class Inference(BaseModel):
    """Where a model runs (design.md 'Inference'): the HF router (`hf_router`, the default),
    OpenAI-compatible servers you run (`urls`) or vLLM servers flock starts on HF Jobs
    (`hf_jobs`), plus settings for any of them. The same block in a Harbor spec (the run's, a
    role's) and a swarm spec (the swarm's, a type's).

    `api_key_env` names a secret / env var; the value is never in the spec. The controller
    sends that secret to the workers and uses it for its checks; the worker hands it to the
    harness.

    With `hf_jobs`, `urls` is what the controller pins once the servers answer (only in
    `resolved_model.inference`; a spec that sets both is refused)."""

    model_config = {"extra": "forbid"}

    hf_router: HFRouter | None = None
    # Servers (`.../v1`) that all serve the type's `model`; each agent calls one of them
    urls: list[str] | None = None
    hf_jobs: HFJobs | None = None
    api_key_env: str | None = None  # default: HF_TOKEN for hf_router and hf_jobs; no key for urls
    # Max agents of this swarm on these endpoints in a turn at once. Enforced at prompt
    # admission by the controller (FLOCK_MAX_WORKING_ENFORCE=0: report only).
    max_working: int | None = Field(default=None, ge=1)
    context_window: int | None = Field(default=None, ge=1)  # tokens, when the server reports none
    # urls only: scrape each server's vLLM `/metrics` into the health snapshots (docs/health.md)
    metrics: bool = False
    # Pi's `--thinking` level; None: Pi's default
    reasoning_effort: Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None

    @field_validator("urls", mode="before")
    @classmethod
    def _url_list(cls, v: Any) -> Any:
        """A comma-separated string is a list (`urls: ${VLLM_URLS}`)."""
        if isinstance(v, str):
            return [u.strip() for u in v.split(",") if u.strip()]
        return v

    @field_validator("urls")
    @classmethod
    def _urls(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        out = []
        for u in v:
            url = urlsplit(u.strip())
            if (url.scheme not in ("http", "https") or not url.hostname or url.username
                    or url.password or url.query or url.fragment):
                raise ValueError(f"{u!r} must be an HTTP(S) URL without credentials, e.g. "
                                 "http://10.0.0.5:8000/v1")
            out.append(u.strip().rstrip("/"))
        if not out:
            raise ValueError("urls must list at least one URL")
        if len(set(out)) != len(out):
            raise ValueError("urls must not repeat a URL")
        return out

    @field_validator("api_key_env")
    @classmethod
    def _env_name(cls, v: str | None) -> str | None:
        if v is not None and not ENV_NAME_RE.match(v):
            raise ValueError(f"api_key_env {v!r} must be an env var name (the key itself never "
                             "goes in the spec)")
        return v

    @model_validator(mode="after")
    def _one(self) -> Inference:
        given = [k for k in ("hf_router", "urls", "hf_jobs") if getattr(self, k) is not None]
        if self.hf_jobs is not None and self.urls is not None:
            given.remove("urls")  # the started servers', pinned (refused in a spec's own block)
        if len(given) > 1:
            raise ValueError(f"set one of hf_router, urls or hf_jobs, not {' and '.join(given)}")
        if not given:
            self.hf_router = HFRouter()
        if self.metrics and self.urls is None:
            raise ValueError("metrics scrapes your servers' vLLM metrics: it needs urls")
        return self

    def held(self) -> Inference:
        """The block as workers (its key) and Trials (endpoints, Pi's config) hold it: without
        the controller's `max_working`."""
        return self.model_copy(update={"max_working": None})

    @property
    def endpoints(self) -> list[str]:
        """The servers' base URLs, as written: `urls` (with `hf_jobs`, the started servers'
        once pinned), or the router's. Empty: hf_jobs servers not started yet."""
        if self.urls:
            return list(self.urls)
        return [HF_ROUTER_URL] if self.hf_router is not None else []

    @property
    def key_env(self) -> str | None:
        """The key's env name: `api_key_env`, else HF_TOKEN for the router and the HF Jobs
        proxy; None: no key."""
        hf = self.hf_router is not None or self.hf_jobs is not None
        return self.api_key_env or ("HF_TOKEN" if hf else None)

    def unpinned(self) -> Inference:
        """The block as a spec writes it: without the started servers' `urls`."""
        return self.model_copy(update={"urls": None}) if self.hf_jobs is not None else self

    @property
    def headers(self) -> dict[str, str]:
        """Request headers besides the key: the router's bill-to org."""
        r = self.hf_router
        return {"X-HF-Bill-To": r.bill_to} if r is not None and r.bill_to else {}

    def request_model(self, model: str) -> str:
        """What the servers are sent as `model`: `<model>:<provider>` for the router with a
        provider or policy set, else `model`."""
        r = self.hf_router
        return f"{model}:{r.provider}" if r is not None and r.provider else model

    def endpoint_for(self, agent: str) -> str:
        """The server `agent` calls: rendezvous hashing over `endpoints`. Stable for a given list,
        spreads agents roughly evenly, and a URL added or removed moves only the agents that
        must move."""
        return max(self.endpoints, key=lambda u: hashlib.sha256(f"{agent}\n{u}".encode()).digest())

    def describe(self) -> str:
        """One line for status: `hf_router deepinfra`, the URLs, or `hf_jobs <flavor>` and the
        started servers' URLs."""
        if self.hf_router is not None:
            return f"hf_router {self.hf_router.provider or 'fastest'}" + (
                f", billed to {self.hf_router.bill_to}" if self.hf_router.bill_to else "")
        if self.hf_jobs is not None:
            return f"hf_jobs {self.hf_jobs.flavor}" + (
                f", urls {', '.join(self.endpoints)}" if self.urls else ", not started")
        return "urls " + ", ".join(self.endpoints)


def loopback_url(url: str) -> bool:
    """The URL names the machine it is used on: `localhost` or a loopback address."""
    host = urlsplit(url).hostname or ""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def private_url(url: str) -> bool:
    """The URL names this machine or a private network (an RFC 1918 or link-local address)."""
    try:
        return loopback_url(url) or ipaddress.ip_address(urlsplit(url).hostname or "").is_private
    except ValueError:
        return False


class ResolvedModel(BaseModel):
    """A type's model after resolution, pinned into the stored spec at apply time."""

    # The inference block it runs on, which admission (`max_working`) and status count by:
    # `inference` (the swarm's) or `types.<type>.inference` (the type's own)
    provider_name: str
    # The block; with `hf_jobs`, plus the started servers' `urls`, pinned by the controller
    # once they answer (controller/servers.py) and kept while the block and model are the same
    inference: Inference
    model_id: str  # the type's `model`: what flock prices the usage under
    reasoning: Reasoning | None = None
    pricing: list[float]  # [input, output, cached] USD per 1M tokens, pinned
    # The window the servers serve the model with, from GET /v1/models (vLLM's max_model_len,
    # the router provider's context_length; under a router policy the smallest among the
    # providers with tools), else `inference.context_window`. Pinned by the controller's apply
    # (controller/inference.py); None before that.
    context_window: int | None = None

    @property
    def qualified(self) -> str:
        return f"{self.provider_name}:{self.model_id}"

    @property
    def request_model(self) -> str:
        """What the harness sends as `model`."""
        return self.inference.request_model(self.model_id)


# --------------------------------------------------------------------------- comms (docs/comms.md)

# Built-in comms tools a type may list in `tools:`; the model sees the bare name (`post`).
COMMS_TOOLS = ("comms:post", "comms:read", "comms:search")
EXTENDED_COMMS_TOOLS = ("comms:channels", "comms:create_channel", "comms:join_channel",
                        "comms:dm", "comms:read_dms", "comms:react", "comms:votes", "comms:doc",
                        "comms:edit_doc")
ALL_COMMS_TOOLS = COMMS_TOOLS + EXTENDED_COMMS_TOOLS
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
    max_channels_per_agent: int = Field(default=3, ge=1, le=100)
    max_channels: int = Field(default=60, ge=1, le=100)
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
    `protocols` are files next to the spec that `flock launch` inlines (`kickoff`,
    `protocols_text`), because the controller and workers cannot read the caller's files."""

    model_config = {"extra": "forbid"}

    kind: Literal["msgboard"] = "msgboard"
    extended: bool = False  # multi-channel tools and serialized per-identity activity polling
    # extended: each agent identity's feed is polled about once per this many seconds; the
    # worker's poll interval is max(2 s, poll_cycle_s / its agents).
    poll_cycle_s: float | None = Field(default=None, ge=10, le=3600)  # unset: 300
    url: str
    # `space`: `flock init` duplicates `template` into a private per-run Space (`space`, default
    # `<template namespace>/board-<swarm>`) and fills BOARD_URL and the invite (docs/comms.md).
    # A url: the same as `url:`. Unset: `url` names an existing board.
    board: str | None = None
    template: str | None = None  # a msgboard Space (one board at its root); required with `space`
    space: str | None = None
    # Space secret -> the run.env (or environment) name holding its value, e.g. {HF_TOKEN:
    # BOARD_BUCKET_TOKEN}: a token scoped to the board's bucket, never the operator's own.
    space_secrets: dict[str, str] = Field(default_factory=dict)
    # Space variables over flock's (MSGBOARD_REPLICA: `<bucket>/board/<swarm>`); "" blanks one.
    space_variables: dict[str, str] = Field(default_factory=dict)
    space_hardware: str = "cpu-basic"  # the Hub requires one to duplicate a Space
    invite_env: str  # secret holding a multi-use agent invite (the code never goes in the spec)
    channel: str  # name, with or without `#`
    kickoff_file: str | None = None
    kickoff: str | None = None  # the file's text, inlined by `flock launch` (or given inline)
    rules: CommsRules = Field(default_factory=CommsRules)
    protocols: list[str] = Field(default_factory=list)  # files appended to the instructions
    protocols_text: list[str] = Field(default_factory=list)  # inlined by `flock launch`
    wake: list[Literal["mention", "thread", "dm", "channel"]] = Field(default_factory=lambda: ["mention", "thread"])
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

    @field_validator("board")
    @classmethod
    def _board(cls, v: str | None) -> str | None:
        if v not in (None, "space") and not v.startswith(("http://", "https://")):
            raise ValueError(f"board {v!r}: use `space` or the board's url")
        return v

    @model_validator(mode="after")
    def _template(self) -> CommsSpec:
        if self.board == "space" and not self.template:
            raise ValueError("board: space needs `template`: the msgboard Space to duplicate")
        return self

    @field_validator("channel")
    @classmethod
    def _channel(cls, v: str) -> str:
        v = v.strip().lstrip("#")
        if not COMMS_NAME_RE.match(v):
            raise ValueError(f"channel {v!r}: use the channel's name, e.g. general")
        return v

    @field_validator("search")
    @classmethod
    def _search(cls, v: str | None) -> str | None:
        if v == "index":
            raise ValueError("search: index is not supported; use search: board (docs/comms.md)")
        return v

    @field_validator("wake")
    @classmethod
    def _wake(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))


def board_url_default(data: Any) -> Any:
    """`comms.board` -> `comms.url` in raw spec data, before ${VAR} substitution: `space` reads
    BOARD_URL (written by `flock init`), a url is the url. An explicit `url` wins."""
    c = data.get("comms") if isinstance(data, dict) else None
    if isinstance(c, dict) and c.get("board") and "url" not in c:
        c["url"] = "${BOARD_URL}" if c["board"] == "space" else c["board"]
    return data


def comms_tools(ts: TypeSpec, comms: CommsSpec | None) -> list[str]:
    """The built-in comms tools a type gets: the `comms:*` entries of its `tools`, else all of
    them with `comms: true` (search only when the swarm's `comms.search` is set)."""
    if comms is None or ts.comms is False:
        return []
    listed = [t for t in ts.tools if t in ALL_COMMS_TOOLS] if isinstance(ts.tools, list) else []
    if listed:
        return list(dict.fromkeys(listed))
    if not ts.comms:
        return []
    return [t for t in (ALL_COMMS_TOOLS if comms.extended else COMMS_TOOLS)
            if t != "comms:search" or comms.search]


class DashboardSpec(BaseModel):
    """Swarm-level `dashboard:` (docs/dashboard.md): `flock init` writes the run's flock-overview
    row to `<registry>/runs/<id>.json` (id: default the swarm). The other row keys are derived from the spec."""

    model_config = {"extra": "forbid"}

    registry: str  # hf://buckets/<ns>/<bucket>, the bucket the overview reads runs/ from
    id: str | None = None  # the overview row's id (runs/<id>.json); default: the swarm name
    name: str | None = None  # default: the swarm name
    title: str | None = None
    setup: str | None = None  # one line
    viewer: str  # the run viewer's url (only named, never called)
    viewer_public: str | None = None
    dataset: str | None = None
    code_repo: str | None = None
    spec_url: str | None = None

    @field_validator("registry")
    @classmethod
    def _registry(cls, v: str) -> str:
        if not v.startswith("hf://buckets/"):
            raise ValueError(f"registry {v!r} must be an hf://buckets/<ns>/<bucket> url")
        return v.rstrip("/")


# --------------------------------------------------------------------------- world (docs/world.md)


class WorldSpec(BaseModel):
    """Swarm-level `world:` (docs/world.md): an HTTP service speaking World API v1 whose tools
    the agents get. `command` (run in `dir`) is started by the controller as a child process
    with `compute: local`; `url` names a world already running (required with `compute:
    hf-jobs`, where starting one is not in this version). `token_env` names the secret sent as
    `X-World-Token`; a started world without it gets a generated one. `endpoint` and `manifest`
    are written by the controller at apply."""

    model_config = {"extra": "forbid"}

    dir: str | None = None  # relative to the spec file; `flock launch` resolves it
    command: list[str] | None = None  # `{port}` and `{python}` are filled in by flock
    url: str | None = None
    token_env: str | None = None
    # Written by the controller at apply; anything given by hand is replaced.
    endpoint: str | None = None
    manifest: WorldManifest | None = None
    # Which credential this configuration belongs to, without the credential: the origin it is
    # bound to (`endpoint`, or `local:<swarm>` for a world flock started) and
    # HMAC(secret, origin). A worker sends its secret only when its own HMAC matches.
    credential_origin: str | None = None
    credential_id: str | None = None
    # The world's Slurm job name, when it runs as one: health shows its squeue state, so a
    # world waiting for a node (PENDING) is told apart from a crashed one (docs/health.md).
    slurm_job_name: str | None = None
    # Seconds the world may be down (health probe) before the controller pauses the types that
    # use it; resumed when it is back (docs/health.md). None: FLOCK_WORLD_DOWN_PAUSE_S or 1800;
    # 0: never. Controller only (not sent to workers).
    down_pause_s: float | None = Field(default=None, ge=0)

    @field_validator("command", mode="before")
    @classmethod
    def _command(cls, v: Any) -> Any:
        if isinstance(v, str):
            import shlex

            v = shlex.split(v)
        if v is not None and (not isinstance(v, list) or not v):
            raise ValueError("command must be a non-empty list (or a string), e.g. "
                             "[flock, world, serve, \"taskboard:TaskBoard\"]")
        return [str(x) for x in v] if v is not None else None

    @field_validator("url", "endpoint")
    @classmethod
    def _url(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError(f"url {v!r} must start with http:// or https://")
        return v.rstrip("/") if v else v

    @field_validator("token_env")
    @classmethod
    def _env(cls, v: str | None) -> str | None:
        if v is not None and not ENV_NAME_RE.match(v):
            raise ValueError(f"token_env {v!r} must be an env var name (the token never goes in "
                             "the spec)")
        return v

    @model_validator(mode="after")
    def _how(self) -> WorldSpec:
        if self.command is None and self.url is None:
            raise ValueError("world needs `command` (started by flock, compute: local) or `url` "
                             "(a world already running)")
        if self.dir is not None and self.command is None:
            raise ValueError("world.dir is where `command` runs; give the command too")
        return self


def world_tools(ts: TypeSpec, world: WorldSpec | None) -> list[str]:
    """The world tools a type gets (model-visible names): the `world:*` entries of its `tools`,
    else every tool of the world's manifest; none with `world: false`."""
    if world is None or ts.world is False:
        return []
    listed = ([t.split(":", 1)[1] for t in ts.tools if t.startswith("world:")]
              if isinstance(ts.tools, list) else [])
    if listed:
        return list(dict.fromkeys(listed))
    return world.manifest.names() if world.manifest is not None else []


class WorkdirSpec(BaseModel):
    """Workdir size feedback and throwaway dirs (docs/design.md "Workdir"). Sizes count what the
    bucket mirror keeps (sync excludes and `ephemeral` left out). Above `soft_*` the agent gets
    a note naming its largest dirs (at most hourly); above `max_*` (default 5x soft) a
    `workdir_over_quota` event too. Nothing is ever refused: the workdir still syncs."""

    model_config = {"extra": "forbid"}

    soft_files: int = Field(default=50_000, ge=1)
    soft_mb: float = Field(default=2048.0, gt=0)
    max_files: int | None = Field(default=None, ge=1)  # None: 5 x soft_files
    max_mb: float | None = Field(default=None, gt=0)  # None: 5 x soft_mb
    # dirs relative to the workdir: never synced, created at setup, wiped when the agent moves
    ephemeral: list[str] = Field(default_factory=list)
    # bucket layout (worker.sync "Base tar"): a tar of the workdir plus per-file deltas, rebuilt
    # every base_interval_s or when the delta passes base_delta_files (changed + deleted)
    base: bool = False
    base_interval_s: float = Field(default=3600.0, gt=0)
    base_delta_files: int = Field(default=5000, ge=1)

    @field_validator("ephemeral")
    @classmethod
    def _relative_dirs(cls, v: list[str]) -> list[str]:
        out = []
        for d in v:
            parts = [p for p in d.strip().split("/") if p not in ("", ".")]
            if not parts or d.strip().startswith("/") or ".." in parts:
                raise ValueError(f"ephemeral {d!r}: a directory relative to the workdir")
            out.append("/".join(parts) + "/")
        return out


class MemoryKnobs(BaseModel):
    """Worker memory thresholds, passed through as the FLOCK_MEM_* settings (worker.main)."""

    model_config = {"extra": "forbid"}

    rotate: float | None = None
    drain: float | None = None
    preempt: float | None = None


class WorkerKnobs(BaseModel):
    """A type's worker settings, applied live (heartbeats) by its workers. Each one unset
    falls back to the worker's env variable, then to the built-in default (docs/slurm.md)."""

    model_config = {"extra": "forbid"}

    drain_budget_s: float | None = Field(default=None, gt=0)  # FLOCK_DRAIN_BUDGET_S
    sync_interval_s: float | None = Field(default=None, ge=10)  # SYNC_INTERVAL
    sync_exclude: list[str] | None = None  # FLOCK_SYNC_EXCLUDE (added to the defaults)
    restore_batch: int | None = Field(default=None, ge=1)  # FLOCK_RESTORE_BATCH
    restore_threads: int | None = Field(default=None, ge=1, le=32)  # FLOCK_RESTORE_THREADS
    memory: MemoryKnobs | None = None


def _duration(v: str) -> str:
    if parse_duration(v).total_seconds() < 0:
        raise ValueError("must be >= 0")
    return v


def _positive_duration(v: str) -> str:
    if parse_duration(v).total_seconds() <= 0:
        raise ValueError("must be > 0")
    return v


class RampPhase(BaseModel):
    """One phase of `replicas_ramp.phases`: `add` every `every` until `until` (default: the
    ramp's `max`). The next phase starts where this one reached `until`. `stagger`: the
    agents of each of its raises join this far apart."""

    model_config = {"extra": "forbid"}

    every: str
    add: int = Field(ge=1)
    until: int | None = Field(default=None, ge=1)
    stagger: str = "0s"

    _every = field_validator("every")(_positive_duration)
    _stagger = field_validator("stagger")(_duration)


class RampHold(BaseModel):
    """`replicas_ramp.hold_if`: a step that takes the swarm's replicas above `above` waits while any vLLM server of a `metrics: true` provider
    had `waiting` or more requests queued in every sample of the last 2 minutes, or takes
    more than `request_s` per request (running / completions per second), or its metrics
    are missing, short or reset."""

    model_config = {"extra": "forbid"}

    waiting: float = Field(default=1, gt=0)
    request_s: float = Field(default=15, gt=0, allow_inf_nan=False)
    above: int = Field(default=100, ge=0)


class RampSpec(BaseModel):
    """`types.T.replicas_ramp` (docs/live.md "Ramp"): the controller raises the type's
    replicas to `start`, then by `add` every `every` (or through `phases`), up to `max`. It
    never lowers them: a lowered `max` only caps later raises. `stagger`: the `start` raise
    adds one agent at a time this far apart (default 10 s)."""

    model_config = {"extra": "forbid"}

    start: int = Field(ge=0)
    add: int | None = Field(default=None, ge=1)
    every: str | None = None
    max: int = Field(ge=1)
    phases: list[RampPhase] = Field(default_factory=list)
    stagger: str = "10s"
    hold_if: RampHold | None = None

    @field_validator("every")
    @classmethod
    def _every(cls, v: str | None) -> str | None:
        return None if v is None else _positive_duration(v)

    _stagger = field_validator("stagger")(_duration)

    @model_validator(mode="after")
    def _one_form(self) -> RampSpec:
        if self.phases and (self.add is not None or self.every is not None):
            raise ValueError("give `phases` or `add` and `every`, not both")
        if not self.phases and (self.add is None or self.every is None):
            raise ValueError("needs `add` and `every` (or `phases`)")
        return self

    def schedule(self) -> list[tuple[float, int, float]]:
        """(seconds from the ramp's start, replicas, stagger seconds) per step; step 0 is
        `start` at 0."""
        sec = lambda d: parse_duration(d).total_seconds()
        t, n = 0.0, min(self.start, self.max)
        out = [(t, n, sec(self.stagger))]
        for ph in self.phases or [RampPhase(every=self.every, add=self.add)]:
            until = min(ph.until or self.max, self.max)
            while n < until:
                t, n = t + sec(ph.every), min(n + ph.add, until)
                out.append((t, n, sec(ph.stagger)))
        return out


class HarborSpec(BaseModel):
    """A `harness: harbor` type's Trial inputs, written by the compile step
    (`flock.harbor.spec.compile_spec`), never by hand: each agent is one long-lived Harbor Trial
    on the upstream Harbor task path. The worker composes the agent's instructions from `team`,
    `role_text`, `roster` and the tools with the upstream `build_agent_instructions`."""

    model_config = {"extra": "forbid"}

    task: dict[str, Any]  # ResolvedTask.manifest(): registry commit, dataset, the pinned task
    # a partial flock.harbor Spec (model, inference, task, environment, limits, tools_dir,
    # tools, resources_dir) for the upstream build_trial_config and the worker's HarborHost
    config: dict[str, Any]
    team: str = ""  # the team_file's text
    role: str | None = None  # the role named in the instructions (None: agents.count)
    role_text: str = ""  # the role_file's text
    # every replica's name -> role: with `naming: consecutive-id`, consecutive names compiled
    # from the counts; with words empty here, and filled by the controller in each assign
    roster: dict[str, str | None] = Field(default_factory=dict)
    tools: dict[str, Any] = Field(default_factory=dict)  # the tool bundle's manifest
    tools_sha256: str | None = None
    resources: dict[str, dict[str, Any]] = Field(default_factory=dict)  # resource options
    # the content hash of the run's tools and resources folders, packed by `flock launch` and
    # uploaded as the swarm's tools bundle of that version: a worker fetches and unpacks it in
    # its own run dir (None: config's tools_dir and resources_dir, read where they are)
    folders: str | None = None

    def names(self) -> list[str]:
        """This type's replica names by slot, with `naming: consecutive-id` (its roster's)."""
        return [a for a, r in self.roster.items() if r == self.role]


class TypeSpec(BaseModel):
    """One agent type inside a swarm. See design.md 'Spec'."""

    harness: Harness
    # A Hub id; with `urls`, the name the servers serve it under (design.md 'Inference')
    model: str = Field(min_length=1)
    # Its own endpoints (a whole block); None: the swarm's `inference`
    inference: Inference | None = None
    # [input, output, cached] USD per 1M tokens of `model`; None: $0
    pricing: list[float] | None = None
    reasoning: Reasoning | None = None
    image: str | None = None  # None -> default worker image
    system_prompt_file: str | None = None  # written into the workdir as AGENTS.md
    # The file's contents, inlined by `flock launch` (the controller and workers cannot
    # read the caller's files). Wins over system_prompt_file on the worker.
    system_prompt: str | None = None
    # A second file appended to the system prompt by `flock launch` (after a blank line): a
    # per-type section on top of a prompt several types share (e.g. a spawner and its
    # child types). The worker only sees the combined `system_prompt`.
    system_prompt_append_file: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    secrets: list[str] = Field(default_factory=list)  # names of secrets the job receives
    labels: dict[str, str] = Field(default_factory=dict)
    replicas: int = 0
    # measured defaults
    agents_per_worker: int = 50
    max_working_per_worker: int = 20
    # Agents of this type in a turn at once, swarm-wide, admitted like a provider's
    # `max_working` (both apply). Controller only.
    max_working: int | None = Field(default=None, ge=1)
    replicas_ramp: RampSpec | None = None  # controller only (docs/live.md "Ramp")
    # Worker slots for this type, declared only (docs/spawn.md). Required for a type that
    # `spawn:` lets others create; None: unbounded.
    workers: WorkersSpec | None = None
    # Spawned agents of this type live for one session (docs/spawn.md "Ephemeral types"): the
    # controller removes one once a heartbeat reports it idle after a turn with nothing pending
    # (done, max_turns, budget or an error alike), and its worker deletes its files. Spawn-only.
    ephemeral: bool = False
    # With the swarm's `naming: words`: put before every name of this type (e.g. "iv-" for
    # interviewers), so roles stay recognizable. Lowercase letters and hyphens, ending in "-".
    name_prefix: str | None = None
    flavor: str = "cpu-basic"
    worker_lifetime: str = "24h"
    lifetime_jitter: str = "4h"
    drain_window: str = "10m"
    idle_grace: str = "120s"
    turn_timeout: str = "30m"
    max_turns: int | None = None  # per prompt, passed to the harness
    max_budget_usd: float | None = None  # per prompt, passed to the harness
    persist_workdir: bool = True
    workdir: WorkdirSpec | None = None  # None: WorkdirSpec() defaults
    worker: WorkerKnobs | None = None  # None: the worker's env, then built-in defaults
    resume_prompt: str | None = None
    # Tools from the swarm's tools folder: a list of function names, or "all".
    # `flock launch` refuses a name the folder does not define. None: no folder tools.
    tools: list[str] | Literal["all"] | None = None
    # Re-prompt this type's agents on a cadence, then a final round (docs/loop.md). None: off.
    loop: LoopSpec | None = None
    # Built-in comms tools (docs/comms.md): true gives all of them (`comms:post`, `comms:read`,
    # `comms:search`), or list the ones wanted in `tools`. Needs the swarm-level `comms:`.
    comms: bool | None = None
    # World tools (docs/world.md): every tool of the swarm's `world:` unless `tools` lists
    # `world:<name>` entries (then only those); false: none.
    world: bool | None = None
    # harness harbor only (and required there): written by the compile step
    harbor: HarborSpec | None = None
    # Written by SwarmSpec validation (apply time), never by hand: anything given here is
    # replaced. None until the type is validated inside a SwarmSpec.
    resolved_model: ResolvedModel | None = None

    @property
    def folder_tools(self) -> list[str] | Literal["all"] | None:
        """`tools` without the built-in `comms:*` and the `world:*` entries: what the tools
        folder must define (None when nothing is left)."""
        if not isinstance(self.tools, list):
            return self.tools
        names = [t for t in self.tools if not t.startswith(("comms:", "world:"))]
        return names or None

    @field_validator("tools")
    @classmethod
    def _known_comms(cls, v: list[str] | str | None) -> list[str] | str | None:
        if isinstance(v, list):
            bad = [t for t in v if t.startswith("comms:") and t not in ALL_COMMS_TOOLS]
            if bad:
                raise ValueError(f"unknown comms tool(s) {', '.join(bad)}; comms tools: "
                                 f"{', '.join(COMMS_TOOLS)}")
            bad = [t for t in v if t.startswith("world:") and not WORLD_TOOL_RE.match(t[6:])]
            if bad:
                raise ValueError(f"bad world tool name(s) {', '.join(bad)}: world:<tool name>")
        return v

    @property
    def harness_model(self) -> str:
        """The model id the harness reports usage under (what flock prices)."""
        return self.resolved_model.model_id if self.resolved_model else self.model

    @field_validator("pricing")
    @classmethod
    def _price_row(cls, v: list[float] | None) -> list[float] | None:
        return None if v is None else _pricing_row(v, "pricing")

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

    @field_validator("name_prefix")
    @classmethod
    def _name_prefix(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"[a-z]+(-[a-z]+)*-", v):
            raise ValueError(f"name_prefix {v!r}: lowercase letters and hyphens, ending in '-' "
                             "(e.g. 'iv-')")
        return v

    @field_validator("replicas", "agents_per_worker", "max_working_per_worker")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    @model_validator(mode="after")
    def _checks(self) -> TypeSpec:
        if self.ephemeral and self.replicas:
            raise ValueError("ephemeral: true is for spawned agents only; set replicas: 0")
        if self.harbor is None:
            raise ValueError("harness harbor needs the `harbor:` block (written by the compile "
                             "step of a flock.harbor spec)")
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


_MEM = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*$", re.IGNORECASE)


def mem_mb(value: str) -> float:
    """sbatch --mem as MB: a bare number is MB; K/M/G/T suffixes (1024-based)."""
    m = _MEM.match(str(value))
    if not m:
        raise ValueError(f"memory {value!r}: expected a number with K, M, G or T")
    scale = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[m.group(2).upper()]
    return float(m.group(1)) * scale


def size_problem(cpus: int | str | None, mem: str | None, max_cpus: int | str | None,
                 max_mem: str | None) -> str | None:
    """Why a job of this size can never start on the partition (None: it fits, or no limit)."""
    try:
        if cpus is not None and max_cpus and int(cpus) > int(max_cpus):
            return f"cpus {cpus} is over max_cpus {max_cpus}: the job would pend forever"
        if mem is not None and mem_mb(mem) > (mem_mb(max_mem) if max_mem else float("inf")):
            return f"mem {mem} is over max_mem {max_mem}: the job would pend forever"
    except ValueError as e:
        return str(e)
    return None


class SlurmResources(BaseModel):
    """Per-type overrides of a `compute: slurm` spec."""

    model_config = {"extra": "forbid"}

    partition: str | None = None
    cpus: int | None = Field(default=None, ge=1)
    mem: str | None = None
    time: str | None = None
    container_root: str | None = None


class ComputeSpec(BaseModel):
    """Where workers run (docs/local.md). `hf-jobs` (the default): one HF Job per worker.
    `local`: one `python -m flock.worker.main` subprocess per worker on the controller's machine
    (`flock launch` runs them on this machine). `slurm`: one `sbatch` job per worker (docs/slurm.md).
    The spec also accepts the short form `compute: local`.

    `worker_unknown_after`, `worker_dead_after` and `crash_backoff` are for every backend
    (docs/live.md "Worker liveness"). The other keys are for `slurm` and win over the
    controller's FLOCK_SLURM_* env, which only fills unset keys (then built-in defaults).
    `types` sets per-type `cpus`/`mem`/`time`/`partition` (an interviewer type needs less
    than an agent)."""

    model_config = {"extra": "forbid"}

    kind: Literal[tuple(COMPUTE)] = "hf-jobs"  # type: ignore[valid-type]
    partition: str | None = None
    account: str | None = None
    cpus: int | None = Field(default=None, ge=1)
    mem: str | None = None  # sbatch --mem, e.g. 7G
    time: str | None = None  # sbatch --time, e.g. 29:00:00
    worker_script: str | None = None  # the cluster's worker sbatch script; $1 = the env file
    log_dir: str | None = None  # shared dir: <log_dir>/<worker id>-<job id>.out
    launch_dir: str | None = None  # shared dir: <worker id>-<incarnation>.env, no secrets
    host_file: str | None = None  # shared file holding the controller's hostname
    run: str | None = None  # RUN= in the env file; default: the launch_dir's parent dir name
    venv: str | None = None  # written to the env file as FLOCK_VENV for the script
    secrets_file: str | None = None  # path only, as FLOCK_SECRETS_FILE; never read by flock
    # workers run in a container whose /tmp is RAM: every agent/harness path goes under this
    # root instead: <root>/flock/<worker id>/{work,home,hf,tmp}
    container_root: str | None = None
    container_image: str | None = None  # an Enroot image (.sqsh): FLOCK_CONTAINER_IMAGE

    @model_serializer(mode="wrap")
    def _compact(self, handler: Any) -> Any:
        """Only the keys that are set: `{kind: local}` stays `{kind: local}` in dumps."""
        out = handler(self)
        if not isinstance(out, dict):
            return out
        return {k: v for k, v in out.items() if k == "kind" or (v is not None and v != {})}
    types: dict[str, SlurmResources] = Field(default_factory=dict)
    # the largest job the partition can ever run: sbatch accepts bigger ones, which then pend
    # forever; a launch over these is refused (also checked at apply). None: not checked (the
    # controller's FLOCK_SLURM_MAX_CPUS/_MEM, from the cluster profile, still apply at launch)
    max_cpus: int | None = Field(default=None, ge=1)
    max_mem: str | None = None
    # worker liveness (docs/live.md "Worker liveness"); None: the backend's default (`liveness`)
    worker_unknown_after: str | None = None
    worker_dead_after: str | None = None
    crash_backoff: str | None = None

    @field_validator("worker_unknown_after", "worker_dead_after", "crash_backoff")
    @classmethod
    def _timing(cls, v: str | None) -> str | None:
        return None if v is None else _positive_duration(v)

    @model_validator(mode="after")
    def _check_liveness(self) -> ComputeSpec:
        lv = self.liveness()
        hb = (LOCAL_HEARTBEAT_INTERVAL if self.kind == "local"
              else HEARTBEAT_INTERVAL).total_seconds()
        if lv.unknown_after < 2 * hb:  # a healthy worker would look silent and be relaunched
            raise ValueError(f"worker_unknown_after ({lv.unknown_after:g}s) must be at least "
                             f"{2 * hb:g}s: two heartbeats at the {self.kind} interval ({hb:g}s)")
        if lv.unknown_after >= lv.dead_after:
            dflt = lambda v: "" if v else " (default)"
            raise ValueError(
                f"worker_unknown_after ({lv.unknown_after:g}s{dflt(self.worker_unknown_after)}) "
                f"must be shorter than worker_dead_after "
                f"({lv.dead_after:g}s{dflt(self.worker_dead_after)})")
        return self

    def liveness(self) -> Liveness:
        """Worker liveness timings in seconds: the keys set here, else the backend's defaults."""
        dflt = (LOCAL_LIVENESS if self.kind == "local"
                else (UNKNOWN_AFTER, DEAD_AFTER, CRASH_BACKOFF))
        keys = (self.worker_unknown_after, self.worker_dead_after, self.crash_backoff)
        return Liveness(*(parse_duration(v or d).total_seconds() for v, d in zip(keys, dflt)))


class SwarmSpec(BaseModel):
    swarm: str
    # HF namespace jobs are launched under, and the bucket agents sync to (e.g.
    # hf://buckets/<namespace>/<bucket>). Required for compute hf-jobs; optional for compute local.
    namespace: str = ""
    bucket: str = ""
    # Where the types' models run (design.md 'Inference'), unless a type sets its own
    inference: Inference = Field(default_factory=Inference)
    stop_on_success: bool = False
    types: dict[str, TypeSpec]
    # The tools folder, relative to the spec file; None: `tools/` next to it.
    # Read by `flock launch` only.
    tools_dir: str | None = None
    # Content hash of the tools bundle, written by `flock launch` after uploading the folder
    # (anything given by hand is replaced). Workers fetch that version; None: no tools.
    tools_version: str | None = None
    # The team's message board (docs/comms.md); None: no comms.
    comms: CommsSpec | None = None
    # The swarm's world (docs/world.md); None: no world.
    world: WorldSpec | None = None
    # The run's flock-overview row (docs/dashboard.md); None: not registered.
    dashboard: DashboardSpec | None = None
    # How agents are named (docs/spawn.md "Names"): `numeric` (`<type>-NNNN`, children
    # `<parent>-<type>-NNNN`), `words` (`<adjective>-<noun>`, unique on the controller,
    # reserved and never reused while the swarm exists; a Harbor spec's default) or
    # `consecutive-id` (Harbor types: their roster's names, consecutive `agent-001`, ...;
    # children as numeric).
    naming: Literal["numeric", "words", "consecutive-id"] = "numeric"

    @field_validator("tools_version")
    @classmethod
    def _valid_tools_version(cls, v: str | None) -> str | None:
        if v is not None and not TOOLS_VERSION_RE.match(v):
            raise ValueError("tools_version is written by `flock launch` (16 hex digits)")
        return v

    @model_validator(mode="after")
    def _resolve_models(self) -> SwarmSpec:
        """Apply-time pinning: each type's inference block and price go into the spec
        (`resolved_model`), so the stored spec carries what admission and cost derive from. The
        context window the controller pinned stays while the block (but `max_working`, the
        controller's) and the model are the same, and so do the hf_jobs servers' `urls`."""
        if self.stop_on_success and not any(
                ts.harbor and ts.harbor.config.get("success_check") for ts in self.types.values()):
            raise ValueError("stop_on_success requires a Harbor success_check")
        for type_name, ts in self.types.items():
            old, ts.resolved_model = ts.resolved_model, resolve_model(self, ts, type_name=type_name)
            new = ts.resolved_model
            if old is not None and (old.inference.unpinned().held(), old.model_id) == (
                    new.inference.held(), new.model_id):
                new.context_window = old.context_window
                if new.inference.hf_jobs is not None:
                    new.inference = new.inference.model_copy(update={"urls": old.inference.urls})
        for type_name, ts in self.types.items():
            cap = ts.loop.while_.agent_usd if ts.loop else None
            if isinstance(cap, dict) and agent_cap(ts) is None:
                raise ValueError(
                    f"types.{type_name}.loop.while.agent_usd: no entry for model {ts.model!r} "
                    f"(keys: {sorted(cap)}); key it by the type's model id")
        return self

    @model_validator(mode="after")
    def _check_comms(self) -> SwarmSpec:
        """A type that wants comms tools needs the swarm's `comms:` block; a listed
        `comms:search` needs `comms.search`; `comms: false` with listed comms tools is a
        contradiction."""
        for name, ts in self.types.items():
            listed = [t for t in ts.tools if t in ALL_COMMS_TOOLS] if isinstance(ts.tools, list) else []
            if (ts.comms or listed) and self.comms is None:
                raise ValueError(f"types.{name}: comms tools need the swarm-level `comms:` block "
                                 "(kind, url, invite_env, channel)")
            if ts.comms is False and listed:
                raise ValueError(f"types.{name}: comms: false but tools lists {', '.join(listed)}")
            if "comms:search" in listed and self.comms is not None and not self.comms.search:
                raise ValueError(f"types.{name}.tools: comms:search needs comms.search (board)")
        for name, ts in self.types.items():  # world tools (docs/world.md)
            listed = ([t for t in ts.tools if t.startswith("world:")]
                      if isinstance(ts.tools, list) else [])
            if (ts.world or listed) and self.world is None:
                raise ValueError(f"types.{name}: world tools need the swarm-level `world:` block "
                                 "(command or url)")
            if ts.world is False and listed:
                raise ValueError(f"types.{name}: world: false but tools lists {', '.join(listed)}")
        if self.dashboard is not None and (self.compute.kind == "local" or not self.bucket):
            raise ValueError("dashboard: needs the run's state in a bucket (`bucket:`, compute "
                             "hf-jobs or slurm); compute: local keeps it on this machine")
        c = self.comms
        if c is not None and c.protocols_text and len(c.protocols_text) != len(c.protocols):
            raise ValueError("comms.protocols_text is written by `flock launch` (one per protocols file)")
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
        kind = self.compute.kind
        unknown = self.compute.liveness().unknown_after
        for name, ts in self.types.items():  # a type's own HEARTBEAT_INTERVAL slows its heartbeats
            try:
                hb = float(ts.env["HEARTBEAT_INTERVAL"])
            except (KeyError, ValueError):
                continue
            if 2 * hb > unknown:
                raise ValueError(f"types.{name}.env.HEARTBEAT_INTERVAL ({hb:g}s): two heartbeats "
                                 f"({2 * hb:g}s) exceed worker_unknown_after ({unknown:g}s); "
                                 f"raise worker_unknown_after or lower the interval")
        if kind == "hf-jobs":
            missing = [f for f in ("namespace", "bucket") if not getattr(self, f)]
            if missing:
                raise ValueError(f"{' and '.join(missing)} required for compute hf-jobs "
                                 "(or set `compute: local`, docs/local.md)")
        elif kind != "local" and not self.bucket:
            # agent state goes to an HF bucket on every remote backend, never to a shared disk
            raise ValueError(f"bucket is required for compute {kind}")
        if kind != "local" and self.world is not None and self.world.url is None:
            raise ValueError(f"world.url is required for compute {kind}: run the world as a "
                             "Job, Space or cluster job and give its url (starting it from "
                             "`command` is compute: local only, docs/world.md)")
        if kind == "slurm":
            sizes = [("compute", self.compute.cpus, self.compute.mem)] + [
                (f"compute.types.{n}", r.cpus, r.mem) for n, r in self.compute.types.items()]
            for where, cpus, mem in sizes:
                if problem := size_problem(cpus, mem, self.compute.max_cpus,
                                           self.compute.max_mem):
                    raise ValueError(f"{where}: {problem}")
        unknown = set(self.compute.types) - set(self.types)
        if unknown:
            raise ValueError(f"compute.types names unknown types: {sorted(unknown)}")
        for ts in self.types.values():  # who must reach the servers: the agents (docs/open-models.md)
            r, backend = ts.resolved_model, ts.harbor and ts.harbor.config.get("environment", {})
            backend = (backend or {}).get("backend")
            for url in (r.inference.urls or []) if r else []:
                if (kind == "hf-jobs" or backend in ("hf-sandbox", "e2b")) and private_url(url):
                    raise ValueError(
                        f"{r.provider_name}.urls: {url} is this machine or a private network, "
                        "which sandboxes and HF Jobs workers cannot reach: serve the model at "
                        "a public URL")
                if kind == "slurm" and loopback_url(url):
                    raise ValueError(
                        f"{r.provider_name}.urls: {url} would be each worker's own node under "
                        "compute slurm: list the server node's address")
        return self

    def inference_blocks(self) -> dict[str, Inference]:
        """The inference blocks the types run on, by name (`inference`, `types.<t>.inference`)."""
        return {ts.resolved_model.provider_name: ts.resolved_model.inference
                for ts in self.types.values() if ts.resolved_model is not None}

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


def resolve_model(spec: SwarmSpec, type_spec: TypeSpec, *, type_name: str = "?") -> ResolvedModel:
    """A type's `model` on its inference block (its own, else the swarm's), with its price
    (`pricing`, else $0). Raises ValueError for a model id the HF router cannot take."""
    inference = type_spec.inference or spec.inference
    model = type_spec.model
    if inference.hf_jobs is not None and inference.urls is not None:
        where = f"types.{type_name}.inference" if type_spec.inference else "inference"
        raise ValueError(f"{where}: set one of hf_router, urls or hf_jobs, not urls and hf_jobs "
                         "(flock pins the started servers' urls itself)")
    if inference.hf_router is not None and (model.startswith("huggingface/") or ":" in model):
        raise ValueError(f"types.{type_name}.model: {model!r}: the HF router takes a plain Hub id "
                         "(e.g. Qwen/Qwen3.8-27B); name the provider in "
                         "inference.hf_router.provider")
    return ResolvedModel(
        provider_name=f"types.{type_name}.inference" if type_spec.inference else "inference",
        inference=inference,
        model_id=model,
        reasoning=type_spec.reasoning,
        pricing=list(type_spec.pricing or ZERO_PRICING),
    )


def agent_name(type_name: str, slot: int) -> str:
    return f"{type_name}-{slot:04d}"


# --------------------------------------------------------------------------- spawn


class WorkersSpec(BaseModel):
    """Worker slots a type may use: `initial` (default: `ceil(replicas /
    agents_per_worker)`) plus up to `max_extra` opened on demand for spawned agents."""

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
    """`ceil(replicas / agents_per_worker)` for a type, or the declared `workers.initial`."""
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
# message_board.*: the message_board resource's events (harness harbor); world.* and artifacts.*
# have no feeder now: a warning at launch
_LOOP_SOURCE_RE = re.compile(r"^(message_board|world|artifacts)\.[A-Za-z0-9_.:*-]+$")


class LoopWhile(BaseModel):
    """While every set condition holds, the loop goes on. Type-wide: `until` (wall time since
    the loop started), `budget_usd` (the type's cost). Per agent: `max_rounds` (loop prompts),
    `agent_usd` (the agent's cost; a map is keyed by model alias or model id)."""

    model_config = {"extra": "forbid"}

    until: str | None = None
    max_rounds: int | None = None
    budget_usd: float | None = None
    agent_usd: float | dict[str, float] | None = None

    @field_validator("until", mode="before")
    @classmethod
    def _until_from_yaml(cls, v: Any) -> Any:
        """Unquoted YAML times arrive as date/datetime; a date means midnight UTC."""
        return v.isoformat() if isinstance(v, date) else v

    @field_validator("until")
    @classmethod
    def _until(cls, v: str | None) -> str | None:
        if v is not None:
            until_deadline(v, 0.0)
        return v

    @field_validator("max_rounds")
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
    `reserve_usd` is held back from `limits.budget_usd` for it."""

    model_config = {"extra": "forbid"}

    prompt: str | None = None
    prompt_file: str | None = None  # inlined into `prompt` by `flock launch`
    reserve_usd: float = 0.0

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


class LoopWake(BaseModel):
    """How `on:` wake-ups spend rounds (docs/loop.md, "Wake queue"). Per agent: at least
    `min_gap` between two wake prompts (wakes in between are coalesced into the next one), at
    most `max` wake prompts per run of the loop, or per `window` when set (None: no cap).
    `counts_as_round: false` keeps wake prompts out of `max_rounds` (they carry the current
    round number and do not advance it); `true`: a wake prompt is a round."""

    model_config = {"extra": "forbid"}

    min_gap: str = "30s"
    max: int | None = 10
    window: str | None = None
    counts_as_round: bool = False

    @field_validator("min_gap", "window")
    @classmethod
    def _duration(cls, v: str | None, info) -> str | None:
        if v is not None and parse_duration(v).total_seconds() <= 0 and info.field_name == "window":
            raise ValueError("window must be a duration > 0")
        return v

    @field_validator("max")
    @classmethod
    def _max(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("must be >= 0")
        return v


class LoopSpec(BaseModel):
    """A type's prompting loop (docs/loop.md). The loop prompt goes on every `every`
    (`turn_end`: re-sent by the worker when a turn completes; a duration: the controller
    prompts idle agents that long after their last loop prompt) and on each `on` wake-up,
    while `while` holds. Then `final`, once per live agent, then idle."""

    model_config = {"extra": "forbid", "populate_by_name": True}

    prompt: str | None = None
    prompt_file: str | None = None  # inlined into `prompt` by `flock launch`
    every: str = "turn_end"
    on: list[str] = Field(default_factory=list)  # message_board.mention, world.event:*
    while_: LoopWhile = Field(default_factory=LoopWhile, alias="while")
    final: LoopFinal | None = None
    wake: LoopWake = Field(default_factory=LoopWake)
    # Controller only: one line after the rendered loop prompt naming the board notifications
    # (mentions, replies in the agent's threads) since its last round. Never sent to workers.
    notices: bool = False

    @model_serializer(mode="wrap")
    def _dump(self, handler):  # `while` on the wire, as written in specs
        out = handler(self)
        if isinstance(out, dict) and "while_" in out:
            out["while"] = out.pop("while_")
        if isinstance(out, dict) and not self.notices:
            out.pop("notices", None)  # off: not dumped, so workers never see the key
        return out

    @field_validator("every")
    @classmethod
    def _every(cls, v: str) -> str:
        if v != "turn_end" and parse_duration(v).total_seconds() <= 0:
            raise ValueError("every must be turn_end or a duration > 0")
        return v

    @field_validator("on")
    @classmethod
    def _sources(cls, v: list[str]) -> list[str]:
        for x in v:
            if not _LOOP_SOURCE_RE.match(x):
                raise ValueError(f"{x!r}: expected message_board.<name>, world.<name> "
                                 "or artifacts.<name>")
        return v

    @model_validator(mode="after")
    def _has_prompt(self) -> LoopSpec:
        if self.prompt is None and self.prompt_file is None:
            raise ValueError("loop needs prompt or prompt_file")
        return self

    @property
    def every_s(self) -> float | None:
        """Seconds between loop prompts for a duration `every`, else None."""
        if self.every == "turn_end":
            return None
        return parse_duration(self.every).total_seconds()


def hit_max_turns(turn_ended: dict) -> bool:
    """A `turn_ended` that stopped at the type's `max_turns` (its `reason`, or `subtype: error_max_turns`).
    Not a failed turn for a loop."""
    return turn_ended.get("reason") == "max_turns" or turn_ended.get("subtype") == "error_max_turns"


def worker_type_spec(ts: TypeSpec) -> TypeSpec:
    """The type spec as workers get it: without the controller-only `loop.notices` (workers
    validate it with extra=forbid; one older than the key would refuse the whole spec)."""
    if ts.loop is None or not ts.loop.notices:
        return ts
    return ts.model_copy(update={"loop": ts.loop.model_copy(update={"notices": False})})


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
    """Loop settings this version accepts in the model but cannot run: world-backed wakes and
    prompt files not inlined. `flock launch` and the controller refuse a spec with any of them."""
    out: list[str] = []
    for name, ts in spec.types.items():
        lp = ts.loop
        if lp is None:
            continue
        where = f"types.{name}.loop"
        out += [f"{where}.on: {x} {NEEDS_WORLD}" for x in lp.on if x.startswith("world.")]
        if lp.prompt is None:
            out.append(f"{where}.prompt_file: not inlined (use `flock launch`, which reads it)")
        if lp.final is not None and lp.final.prompt is None:
            out.append(f"{where}.final.prompt_file: not inlined (use `flock launch`, which reads it)")
    return out


CommsSpec.model_rebuild()
WorldSpec.model_rebuild()
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


def type_spec_rev(ts: TypeSpec) -> str:
    """A short content hash of a type spec: the heartbeat sends a type spec only when the
    worker's revision differs (docs/live.md)."""
    return hashlib.sha256(ts.model_dump_json().encode()).hexdigest()[:16]


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
    turns: int = 0
    last_sync: datetime | None = None
    msgboard_principal: str | None = None
    rev: str | None = None  # `type_spec_rev` the agent applied at its last turn boundary
    start_rev: str | None = None  # the one its running harness process started with
    endpoint: str | None = None  # `host:port` of the model server it is pinned to (base_urls)
    trial: dict[str, Any] | None = None  # harness harbor: {id, attempt, alive} of its Trial


class WorkerMem(BaseModel):
    """The worker's cgroup memory (MB) at its last sample (flock.worker.cgroup)."""

    current_mb: float | None = None
    limit_mb: float | None = None  # None: no limit found, nothing is armed
    nonreclaim_mb: float | None = None  # anon + shmem + slab_unreclaimable + stacks + pagetables
    slope_mb_h: float | None = None  # of nonreclaim_mb over the last hour
    eta_limit_s: float | None = None  # nonreclaim_mb reaches limit_mb at that slope
    psi_some_avg10: float | None = None
    oom_kills: int | None = None  # memory.events oom_kill, cumulative for the cgroup


class RestoreProgress(BaseModel):
    """Restores in flight on the worker: files downloaded of the files listed."""

    agents: int = 0
    files_done: int = 0
    files_total: int = 0
    started: datetime | None = None  # the oldest restore in flight


class ToolCounters(BaseModel):
    """One world tool's calls since this worker process started."""

    n: int = 0
    err: int = 0
    refused: int = 0  # the world said no for now: HTTP 429 or a budget/busy error code
    timeouts: int = 0
    sum_s: float = 0.0
    max_s: float = 0.0


class WorkerHeartbeat(BaseModel):
    worker: str
    deadline: datetime
    draining: bool = False
    agents: list[AgentHeartbeat]
    # `type_spec_rev` of the type spec the worker has (docs/live.md): the controller sends the
    # current one in the heartbeat response only when it differs
    spec_rev: str | None = None
    # the worker's last heartbeat after a preemption notice (SIGUSR1): it synced and exits;
    # the controller relaunches the slot at once, as a planned restart (docs/slurm.md)
    preempted: bool = False
    # the worker's last heartbeat after a finished drain (`unsynced` names any agent whose final
    # sync failed): it exits now, and the controller ends the slot without waiting for the job
    drained: bool = False
    # agents whose turn a preemption or a bounded drain cut; they get the type's
    # resume_prompt after the restore
    interrupted: list[str] = Field(default_factory=list)
    # telemetry (docs/design.md "Heartbeat"); None from workers that do not send it
    mem: WorkerMem | None = None
    restore: RestoreProgress | None = None  # None when no restore is in flight
    tools: dict[str, ToolCounters] | None = None  # world tools by name
    knobs_rev: str | None = None  # hash of the worker's effective settings
    # nonreclaimable memory passed FLOCK_MEM_ROTATE of the limit: the controller drains the
    # worker when its rotation budget allows (docs/slurm.md "Memory")
    wants_rotation: bool = False
    # agents whose last push to the bucket failed (retried once), with the error; the drain's
    # final heartbeat carries them so `flock stop --release` can name them
    unsynced: dict[str, str] = Field(default_factory=dict)


class Event(BaseModel):
    """One line of an agent's event stream.

    The on-disk / over-the-wire shape is pi session format v3 (flock.worker.events
    documents the encoding). `type`, `ts`, `agent` are required by us;
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

    kind: Literal["loop", "final"]
    type: str
    round: int | None = None  # None: the worker numbers it (its own next round)
    started_at: datetime | None = None  # when the loop started, for {elapsed}
    wall_s: float | None = None  # `while.until` in seconds, for {wall}
    cap_usd: float | None = None  # `while.agent_usd` for this agent, for {cap}
    why: str = ""
    # False: a wake prompt outside `max_rounds` (`wake.counts_as_round: false`); it carries the
    # current round and does not advance the worker's round counter
    counted: bool = True
    wake: bool = False  # a wake prompt (`on:`), counted or not: a deferral gives back its wake


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
    # One run of the loop (POST /swarms/{swarm}/loop start begins a new one): a new run resets the
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
    tools_version: str | None = None  # the swarm's tools bundle; None: no tools
    comms: CommsSpec | None = None  # the swarm's comms block (docs/comms.md), channel resolved
    world: WorldSpec | None = None  # the swarm's world (docs/world.md), manifest fetched


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
    provider: str | None = None  # its inference block's name (ResolvedModel.provider_name)
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
    turns: int = 0
    created_at: datetime
    last_heartbeat: datetime | None = None
    last_sync: datetime | None = None
    msgboard_principal: str | None = None
    parent: str | None = None  # spawning agent; None for a replica
    children: list[str] = Field(default_factory=list)
    depth: int = 0  # generations this agent may still spawn below itself
    rev: str | None = None  # AgentHeartbeat.rev / start_rev
    start_rev: str | None = None
    endpoint: str | None = None  # AgentHeartbeat.endpoint


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
    pending_reason: str | None = None  # while queued: the scheduler's word, e.g. "PENDING (Resources)"
    # from the last heartbeat (WorkerHeartbeat)
    mem: WorkerMem | None = None
    restore: RestoreProgress | None = None
    tools: dict[str, ToolCounters] | None = None
    knobs_rev: str | None = None
    wants_rotation: bool = False


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
    agents_done: int = 0  # out of the loop (max_rounds, agent_usd, errors)
    rounds_min: int = 0
    rounds_max: int = 0
    rounds_total: int = 0
    wakes_total: int = 0  # wake prompts sent (`on:`); in rounds_total too only with counts_as_round
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
    """An inference block the types run on (`SwarmSpec.inference_blocks`)."""

    name: str  # `inference`, or `types.<t>.inference` for a type's own
    endpoint: str  # Inference.describe(): `hf_router deepinfra` or the URLs
    types: list[str]  # agent types on it
    # each of their models -> the context window pinned at apply (None: not pinned yet)
    models: dict[str, int | None] = Field(default_factory=dict)
    working: int  # agents of those types in a turn now (from heartbeats)
    max_working: int | None = None  # enforced at prompt admission unless FLOCK_MAX_WORKING_ENFORCE=0
    # Agents with turns but zero reported tokens: the server sends no usage, cost is unknown.
    zero_usage_agents: int = 0


class ServerStatus(BaseModel):
    """One server flock started for an inference block (`hf_jobs`, controller/servers.py): a
    replica of one model's group."""

    model: str
    kind: str  # the block kind that started it: hf_jobs
    group: str  # the group's key (its jobs' `inference` label)
    types: list[str]  # the agent types it serves
    flavor: str
    namespace: str
    replica: int
    job_id: str | None = None
    url: str | None = None  # the OpenAI base URL (`.../v1`)
    stage: str | None = None  # the job's: SCHEDULING, RUNNING, COMPLETED, CANCELED, ERROR, ...
    message: str | None = None  # the job's status message (e.g. waiting for hardware)
    # starting (the job is scheduled or the server does not answer yet), ready (it answers
    # GET /v1/models), failed (the job ended or never answered: `error`), stopped (cancelled)
    state: Literal["starting", "ready", "failed", "stopped"]
    started_at: datetime | None = None  # the job's creation
    ready_at: datetime | None = None
    error: str | None = None


class SwarmStatus(BaseModel):
    swarm: str
    namespace: str
    bucket: str
    types: dict[str, TypeStatus]
    providers: dict[str, ProviderStatus] = Field(default_factory=dict)
    servers: list[ServerStatus] = Field(default_factory=list)  # hf_jobs servers flock started
    agents_total: int
    workers_total: int
    pending_prompts: int
    cost_usd: float = 0.0
    # limits.budget_usd as a stop (docs/loop.md): None (under the threshold or no budget),
    # "final_reserve" (loops stopped, final rounds) or "hard" (every agent stopped).
    budget_stop: Literal["final_reserve", "hard"] | None = None
    # pause holds (docs/loop.md "Pause and resume"): {types: {T: {since, reason, by, auto}},
    # agents: {A: ...}}; None when nothing is held
    held: dict[str, dict[str, dict[str, Any]]] | None = None
    # `flock stop --release`: {since, reason, unsynced: {agent: error}}; None when not released
    released: dict[str, Any] | None = None
    # bumped by every apply that changes the spec (docs/live.md)
    success: dict[str, Any] | None = None
    generation: int | None = None


class LoopRequest(BaseModel):
    """`POST /swarms/{swarm}/loop`: stop or (re)start one type's loop."""

    type: str
    action: Literal["start", "stop"]
    final: bool = True  # stop: run the final round (if the type has one)


class PreflightResult(BaseModel):
    """One provider-backed model probed with a tool call and a follow-up turn."""

    model: str  # "<inference block>:<model id>" (ResolvedModel.qualified)
    provider: str  # the inference block's name
    types: list[str]
    # "responses" (openai_compat) | "messages" (anthropic) |
    # "chat_completions" (openai_compat; Harbor's Pi)
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
LOCAL_HEARTBEAT_INTERVAL = timedelta(seconds=5)  # compute local: agent states within seconds
# Worker liveness defaults; a spec's `compute:` overrides them (ComputeSpec.liveness).
UNKNOWN_AFTER = timedelta(seconds=90)
DEAD_AFTER = timedelta(seconds=180)
CRASH_BACKOFF = timedelta(seconds=30)  # first relaunch wait, doubling per crash in a row
# compute local: 3 and 6 missed local heartbeats, the jobs ratio; an exited worker
# process is caught on the next reconcile pass anyway, so these only time out a hung one
LOCAL_LIVENESS = (timedelta(seconds=15), timedelta(seconds=30), timedelta(seconds=2))


class Liveness(NamedTuple):
    """A swarm's worker liveness timings, in seconds (`ComputeSpec.liveness`)."""

    unknown_after: float
    dead_after: float
    crash_backoff: float

CRASHLOOP_THRESHOLD = 5
LAUNCHES_PER_WINDOW = 400  # per 5-minute Hub rate-limit window, PRO assumption
RATE_WINDOW = timedelta(minutes=5)
JOB_LABELS = ("swarm", "type", "worker")  # every worker job carries these labels
