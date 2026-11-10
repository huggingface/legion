"""flock's run spec: shepherd's `Config` plus flock's blocks (loop, ramps, workers, compute, spawn).

One `spec.yaml` (plus its prompt files) per run. A shepherd config loads unchanged.
`Spec.load` fills `${VAR}` / `${VAR:-default}` in string values and inlines prompt files.
`compile_spec` turns a Spec into the `SwarmSpec` flock's controller and workers run: one type
per role, `harness: harbor`, each agent one Harbor Trial on shepherd's task path.

Each role is a flock type, so a loop's `while.budget_usd` is per role; `limits.budget_usd` is
the run's.
"""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from harbor.utils.env import is_sensitive_env_key
from pydantic import BeforeValidator, Field, field_validator, model_validator

from flock.models import (
    ComputeSpec,
    DashboardSpec,
    LoopSpec,
    SpawnRule,
    SwarmSpec,
    TypeSpec,
    WorkerKnobs,
    load_spec_yaml,
    max_workers,
    parse_duration,
)
from flock.world.protocol import ADMIN_TOKEN_ENV

from .config import (
    AgentSettings,
    Config,
    EnvironmentSettings,
    InferenceSettings,
    Limits,
    RoleName,
    RoleSettings,
    Settings,
)

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
# the tools a role without `spawn:` rules is not shown (it may still prompt its parent)
SPAWNING = ("spawn.spawn", "spawn.remove")
PROVIDER = "inference"  # the compiled spec's provider: prices and admits the run's models


def seconds(value):
    """'30s', '10m', '48h', '2d' or a number of seconds -> seconds."""
    if isinstance(value, int | float):
        return value
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", str(value))
    if not m:
        raise ValueError(f"bad duration {value!r}; use e.g. 30s, 10m, 48h")
    return float(m[1]) * _UNITS[m[2]]


Duration = Annotated[float, BeforeValidator(seconds), Field(gt=0)]
# [input, output, cached] USD per 1M tokens; Harbor reports no cost for router models.
Pricing = Annotated[list[Annotated[float, Field(ge=0)]], Field(min_length=3, max_length=3)]


def flock_duration(value: float) -> str:
    """Seconds as flock writes durations: a whole number of a unit."""
    return f"{value:.0f}s" if value == int(value) else f"{round(value * 1000)}ms"


class Ramp(Settings):
    """Start `start` agents of the role, then `add` more every `every`, up to the role's count."""

    start: int = Field(ge=0, strict=True)
    add: int = Field(ge=1, strict=True)
    every: Duration


class Loop(LoopSpec):
    """flock's loop (docs/loop.md) plus `resume_prompt`, the type's prompt after a restart.
    Board wakes are `message_board.<kind>`; `comms.<kind>` is read as `message_board.<kind>`."""

    resume_prompt: str = (
        "Your session was restarted. Check your notes and state, and continue your work."
    )

    @field_validator("on", mode="before")
    @classmethod
    def _board(cls, value):
        if not isinstance(value, list):
            return value
        return [
            "message_board." + x.removeprefix("comms.")
            if isinstance(x, str) and x.startswith("comms.")
            else x
            for x in value
        ]


class Workers(Settings):
    """How flock packs agents on workers: main's per-type keys (unset: main's defaults). A
    role's `workers:` overrides keys. `initial` + `max_extra`: worker slots for spawned agents."""

    agents_per_worker: int | None = Field(default=None, ge=1, strict=True)
    max_working_per_worker: int | None = Field(default=None, ge=1, strict=True)
    flavor: str | None = None
    worker_lifetime: str | None = None
    lifetime_jitter: str | None = None
    drain_window: str | None = None
    idle_grace: str | None = None
    turn_timeout: str | None = None
    worker: WorkerKnobs | None = None
    # the workers' own env: what a task's `[environment.env]` `${VAR}` reads, as Harbor fills
    # it from the worker's environment. Stored in the spec, so compile_spec refuses
    # secret-looking names (a task secret goes in the worker host's env, e.g. Slurm's run.env)
    env: dict[str, str] | None = None
    initial: int | None = Field(default=None, ge=0, strict=True)
    max_extra: int | None = Field(default=None, ge=0, strict=True)

    @field_validator(
        "worker_lifetime", "lifetime_jitter", "drain_window", "idle_grace", "turn_timeout"
    )
    @classmethod
    def _duration(cls, value):
        if value is not None:
            parse_duration(value)
        return value


class Role(RoleSettings):
    count: int = Field(default=1, ge=0, strict=True)  # 0: spawned agents only (`spawn:`)
    model: str | None = Field(default=None, min_length=1)  # the run's model otherwise
    ramp: Ramp | None = None
    loop: Loop | None = None  # replaces the run's loop for this role
    workers: Workers | None = None  # overrides keys of the run's workers
    pricing: Pricing | None = None  # its own `model`'s; budgets need it (else the run's)

    @model_validator(mode="after")
    def ramp_within_count(self):
        if self.ramp is not None and self.ramp.start > self.count:
            raise ValueError("ramp.start must be <= count")
        return self


class Agents(AgentSettings):
    roles: dict[RoleName, Role] | None = Field(default=None, min_length=1)


class Inference(InferenceSettings):
    max_working: int | None = Field(default=None, ge=1, strict=True)  # agents in a turn at once


class SpecLimits(Limits):
    trial_hours: float = Field(default=23, gt=0)  # a new Trial after this; hf-sandbox caps at 24 h
    budget_usd: float | None = Field(default=None, ge=0)  # the whole run
    alive: int | None = Field(default=None, ge=0, strict=True)  # agents not removed
    spawned_total: int | None = Field(default=None, ge=0, strict=True)


class Spec(Config):
    agents: Agents = Field(default_factory=Agents)
    inference: Inference = Field(default_factory=Inference)
    environment: EnvironmentSettings = Field(default_factory=EnvironmentSettings)
    limits: SpecLimits = Field(default_factory=SpecLimits)
    loop: Loop | None = None
    pricing: Pricing | None = None
    workers: Workers = Field(default_factory=Workers)
    # Where flock's workers run (docs/local.md); a shepherd config runs here, on its engine.
    compute: ComputeSpec = Field(default_factory=lambda: ComputeSpec(kind="local"))
    namespace: str = ""  # bills the HF Jobs workers and the hf-sandbox sandboxes (unset: the token's)
    bucket: str = ""
    dashboard: DashboardSpec | None = None
    spawn: dict[str, SpawnRule] = Field(default_factory=dict)
    # agent names: flock's random `<adjective>-<noun>` (words), or shepherd's roster() names
    # (`agent-001`, ... numbered across the roles in order) for a task that relies on them
    naming: Literal["words", "shepherd"] = "words"

    @field_validator("compute", mode="before")
    @classmethod
    def _compute_short(cls, value):
        return {"kind": value} if isinstance(value, str) else value

    def roles(self):
        """Role name -> Role; `agents.count` is one role, `agent`, whose agents have no role."""
        return self.agents.roles or {"agent": Role(count=self.agents.count)}

    def type_loops(self):
        """Each role's loop: its own, else the run's. Roles sharing the run's loop split its
        `final.reserve_usd`, which is held back once (flock sums its types' reserves)."""
        roles = self.roles()
        shared = sum(r.loop is None for r in roles.values())
        out = {}
        for name, role in roles.items():
            loop = role.loop or self.loop
            if loop is not None and role.loop is None and loop.final is not None:
                final = loop.final.model_copy(
                    update={"reserve_usd": loop.final.reserve_usd / shared}
                )
                loop = loop.model_copy(update={"final": final})
            if loop is not None:
                out[name] = loop
        return out

    @model_validator(mode="after")
    def checks(self):
        limits, backend = self.limits, self.environment.backend
        loops = self.type_loops().values()
        for loop in loops:
            try:
                until = parse_duration(loop.while_.until).total_seconds()
            except (TypeError, ValueError):  # unset, or a time of day
                until = None
            if until is not None and limits.run_timeout_seconds <= until:
                raise ValueError("limits.run_timeout_seconds must be longer than loop.while.until")
            if loop.while_.until is None and "run_timeout_seconds" not in limits.model_fields_set:
                raise ValueError(
                    "a loop without while.until runs for limits.run_timeout_seconds: set it "
                    "(the default is 900 s, 15 minutes) or while.until"
                )
        shared = sum(role.loop is None for role in self.roles().values())
        if self.loop is not None and self.loop.while_.budget_usd is not None and shared > 1:
            raise ValueError(
                f"loop.while.budget_usd would apply to each of the {shared} roles sharing the "
                "run's loop: set limits.budget_usd for the run, or a budget in each role's loop"
            )
        reserve = sum(loop.final.reserve_usd for loop in loops if loop.final)
        if limits.budget_usd is not None and reserve > limits.budget_usd:
            raise ValueError(
                f"loop final.reserve_usd ({reserve:g}) is more than limits.budget_usd"
            )
        for name, role in self.roles().items() if backend == "hf-sandbox" else ():
            turn = ((role.workers and role.workers.turn_timeout) or self.workers.turn_timeout
                    or TypeSpec.model_fields["turn_timeout"].default)
            if limits.trial_hours * 3600 + parse_duration(turn).total_seconds() >= 86400:
                raise ValueError(
                    f"hf-sandbox caps a sandbox at 24 h: limits.trial_hours plus {name}'s "
                    f"turn_timeout ({turn}, a Trial's last turn) must stay below it"
                )
        if backend == "hf-sandbox" and "spawn" in self.resources:
            raise ValueError("resources.spawn needs a mount: use backend docker or podman")
        if backend == "hf-sandbox" and (self.resources.get("message_board") or {}).get("rules"):
            # no mounts, no rule proxy: the board runs its rules unenforced and warns
            self.resources["message_board"].setdefault("transport", "http")
        budgets = limits.budget_usd is not None or any(
            loop.while_.agent_usd is not None or loop.while_.budget_usd is not None
            for loop in loops
        )
        if budgets and self.pricing is None:
            raise ValueError("budgets need pricing: [input, output, cached] USD per 1M tokens")
        for name, role in self.roles().items():
            if budgets and role.model not in (None, self.model) and role.pricing is None:
                raise ValueError(f"agents.roles.{name} has its own model: budgets need its pricing")
        if self.compute.kind == "hf-jobs" and backend != "hf-sandbox":
            raise ValueError("compute hf-jobs cannot run containers: use backend hf-sandbox")
        if self.compute.kind == "slurm" and backend == "docker":
            raise ValueError("compute slurm runs rootless: use backend podman or hf-sandbox")
        return self

    @classmethod
    def load(cls, path: Path, env=None):
        """`env` fills `${VAR}` (default: the process environment)."""
        path = Path(path).resolve()
        here = path.parent
        # YAML 1.2 booleans, as flock's specs: `on:` stays a key
        data = load_spec_yaml(path.read_text())
        spec = cls.model_validate(substitute(data, os.environ if env is None else env))
        # as shepherd's Config.load: a path set in the YAML is relative to it
        for field in ("output_dir", "tools_dir", "resources_dir"):
            value = getattr(spec, field)
            if field in spec.model_fields_set:
                value = here / value.expanduser()
            setattr(spec, field, value.resolve())
        if spec.team_file is not None:
            spec.team_file = (here / spec.team_file.expanduser()).resolve()
        roles = (spec.agents.roles or {}).values()
        for role in roles:
            if role.role_file is not None:
                role.role_file = (here / role.role_file.expanduser()).resolve()
        for loop in (spec.loop, *(role.loop for role in roles)):
            for prompted in (loop, loop and loop.final):
                if prompted and prompted.prompt_file is not None:
                    prompted.prompt = (here / Path(prompted.prompt_file).expanduser()).read_text(
                        encoding="utf-8"
                    )
                    prompted.prompt_file = None  # run.json carries the text
        return spec


def compile_spec(spec: Spec, task: dict, bundle=None, folders: str | None = None) -> SwarmSpec:
    """The SwarmSpec flock runs `spec` as: one type per role, `harness: harbor`, whose `harbor:`
    block holds what a worker needs to start each agent's Trial. `task` is the pinned task
    (`ResolvedTask.manifest()`); `bundle` the built tool bundle (None: no tools); `folders` the
    version of `pack_folders` (None: workers read tools_dir and resources_dir where they are)."""
    team = spec.team_file.read_text(encoding="utf-8") if spec.team_file is not None else ""
    roles, loops = spec.roles(), spec.type_loops()
    label = {name: name if spec.agents.roles else None for name in roles}
    # the folders: a worker builds the tool bundle and loads the resource modules from them
    config = spec.model_dump(mode="json", include={
        "inference", "task", "environment", "tools_dir", "tools", "resources_dir", "namespace"})
    config["limits"] = spec.limits.model_dump(
        include={"agent_max_turns", "agent_timeout_seconds", "trial_hours"}
    )
    # the inference key goes into every agent's container: never the token a remote worker
    # syncs the bucket and starts sandboxes with
    remote = spec.environment.backend == "hf-sandbox" or spec.compute.kind in ("slurm", "hf-jobs")
    if remote and spec.inference.api_key_env == "HF_TOKEN":
        raise ValueError("inference.api_key_env: HF_TOKEN is the workers' Hub token (bucket, "
                         "sandboxes) and would reach every agent's container: name an "
                         "inference-only token, e.g. HF_INFERENCE_TOKEN")
    board = spec.resources.get("message_board") or {}
    if remote and board.get("space_id") and board.get("private") is not False:
        raise ValueError("resources.message_board.private: set it to false: a private Space's "
                         "gate needs the workers' Hub token in every agent's container")
    for where, w in [("workers", spec.workers)] + [
            (f"agents.roles.{n}.workers", r.workers) for n, r in roles.items() if r.workers]:
        if bad := sorted(k for k in w.env or {} if is_sensitive_env_key(k)):
            # Harbor's check is a loose substring match: KEY, TOKEN, SECRET, PASSWORD,
            # CREDENTIAL, AUTH anywhere in a name refuse it, harmless or not
            raise ValueError(f"{where}.env: {', '.join(bad)} cannot be set here: the block is "
                             "stored in the spec and lands in the worker job's plain env, so "
                             "names containing KEY/TOKEN/SECRET/PASSWORD/CREDENTIAL/AUTH are "
                             "refused (even if harmless). Put a task secret in the worker host's "
                             "environment instead: on Slurm add NAME=value to run.env, which "
                             "every worker job sources, and the task's ${NAME} reads it there. "
                             "On hf-jobs only the inference key and resource tokens are passed "
                             "as secrets")
    for name, options in spec.resources.items():
        if options.get("token_env") == ADMIN_TOKEN_ENV:
            raise ValueError(f"resources.{name}.token_env: {ADMIN_TOKEN_ENV} is the world's "
                             "admin secret, never a worker's: name the world's agent secret "
                             "(its FLOCK_WORLD_TOKEN)")
    # besides the inference key (the provider's): what the workers read on their own host
    secrets = resource_secrets(spec) + (["HF_TOKEN"] if spec.environment.backend == "hf-sandbox"
                                        else [])  # sandboxes start with the worker's HF token
    secrets = list(dict.fromkeys(secrets))  # a Space board's HF_TOKEN may be the sandboxes' too
    types = {}
    for name, role in roles.items():
        model = role.model or spec.model
        packing = spec.workers.model_dump(exclude_none=True)
        packing |= role.workers.model_dump(exclude_none=True) if role.workers else {}
        slots = {k: packing.pop(k) for k in ("initial", "max_extra") if k in packing}
        ramp, loop = role.ramp, loops.get(name)
        if loop is not None and loop.while_.until is None:  # the run's timeout ends the loop
            until = flock_duration(spec.limits.run_timeout_seconds)
            loop = loop.model_copy(update={"while_": loop.while_.model_copy(
                update={"until": until})})
        tools = bundle.manifest if bundle is not None else {}
        if tools and name not in spec.spawn:
            tools = {**tools, "tools": {k: v for k, v in tools["tools"].items()
                                        if k not in SPAWNING}}
        types[name] = {
            **packing,
            "harness": "harbor",
            "model": f"{PROVIDER}:{model}",
            "replicas": 0 if ramp else role.count,  # the ramp raises them
            "replicas_ramp": ramp and {
                "start": ramp.start, "add": ramp.add, "every": flock_duration(ramp.every),
                "max": role.count,
            },
            "workers": slots or None,
            "max_turns": spec.limits.agent_max_turns,
            "secrets": secrets,
            "loop": loop and loop.model_dump(exclude={"resume_prompt"}),
            "resume_prompt": loop and loop.resume_prompt,
            "harbor": {
                "task": task,
                "config": {**config, "model": model},
                "team": team,
                "role": label[name],
                "role_text": role.role_file.read_text(encoding="utf-8") if role.role_file else "",
                "tools": tools,
                "tools_sha256": bundle.sha256 if bundle is not None else None,
                "resources": spec.resources,
                # shepherd's names and roles; words: none here, each assign carries the run's
                "roster": spec.agents.roster() if spec.naming == "shepherd" else {},
                "folders": folders,
            },
        }
    swarm = SwarmSpec.model_validate({
        "swarm": spec.name,
        "namespace": spec.namespace,
        "bucket": spec.bucket,
        "providers": {PROVIDER: {
            "type": "openai_compat",
            "base_url": spec.inference.base_url,
            "api_key_env": spec.inference.api_key_env,
            "max_working": spec.inference.max_working,
            # no pricing: the run's cost shows as 0 (Harbor reports none for router models)
            "pricing": {"*": spec.pricing or [0.0, 0.0, 0.0]} | {
                role.model: role.pricing for role in roles.values() if role.model and role.pricing
            },
        }},
        "types": types,
        "dashboard": spec.dashboard,
        "spawn": spec.spawn,
        "limits": spec.limits.model_dump(include={"alive", "spawned_total", "budget_usd"}),
        "compute": spec.compute,
        "naming": spec.naming,
    })
    most = [max_workers(ts.model_copy(update={"replicas": ts.replicas_ramp.max}) if ts.replicas_ramp
                        else ts) for ts in swarm.types.values()]  # a ramp's max, not its start
    if board.get("space_id") and sum(most) > 1:
        # each worker runs its own resources, and each deploy() rotates the board's operator token
        raise ValueError("resources.message_board.space_id: the worker that starts the board "
                         "deploys it, so a run of more than one worker (a role each, or "
                         "agents_per_worker below the count) cannot: deploy the board once and "
                         "set message_board.url and token_env, or fit the run on one worker")
    return swarm


def pack_folders(tools_dir, resources_dir):
    """The run's tools and resources folders (`tools/`, `resources/`; a missing one left out,
    a link inside either refused) as one flock tools bundle (flock.tools.bundle): `flock launch`
    uploads it to the controller as the swarm's tools version, and every worker fetches it by
    that content hash and unpacks it in its own run dir (flock.worker.harbor)."""
    from flock.tools.bundle import pack
    from flock.tools.loader import Manifest

    with tempfile.TemporaryDirectory() as d:
        for name, src in (("tools", Path(tools_dir)), ("resources", Path(resources_dir))):
            if src.is_dir():  # a link inside is kept, so `pack` refuses it (the folder may be one)
                shutil.copytree(src, Path(d) / name, symlinks=True,
                                ignore=shutil.ignore_patterns("__pycache__"))
        return pack(d, Manifest())  # no flock folder tools: Harbor's bundle carries the tools


def resource_secrets(spec: Spec) -> list[str]:
    """The env names resources read a token from on the worker: a plugin's `token_env` (its
    default too) when it connects to a `url`, HF_TOKEN when it deploys a Space (`space_id`). A
    missing module is refused here, before any worker or sandbox starts."""
    from .resources import load_module

    names = []
    for name, options in spec.resources.items():
        path = spec.resources_dir / f"{name}.py"
        if not path.is_file():
            renamed = " (the board resource is now message_board)" if name == "board" else ""
            raise ValueError(f"resources.{name}: no resource module {path}{renamed}")
        schema = getattr(load_module(path), "Options", None)
        settings = schema.model_validate(options) if schema is not None else None
        if getattr(settings, "url", None) and getattr(settings, "token_env", None):
            names.append(settings.token_env)
        if getattr(settings, "space_id", None):
            names.append("HF_TOKEN")
    return names


# From flock's specrender: only UPPERCASE names are variables, so `{spend}` and `${spend}` in
# prompts pass through; `$${` is a literal `${`.
_VAR = re.compile(r"\$\$(?=\{)|\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def substitute(data, env):
    """Fill placeholders in every string value. A string that is one placeholder takes the
    value's JSON type (`"${N:-20}"` -> 20, `"0123"` stays a string). Raises naming every unset
    variable without a default."""
    missing = set()

    def rep(m):
        name, default = m.groups()
        if name is None:
            return "$"
        value = env.get(name)
        if value or (value is not None and default is None):  # `:-` as in sh: empty = unset
            return value
        if default is not None:
            return default
        missing.add(name)
        return m.group(0)

    def walk(v):
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if not isinstance(v, str):
            return v
        out = _VAR.sub(rep, v)
        whole = _VAR.fullmatch(v)
        if whole and whole.group(1) and out:
            try:
                typed = json.loads(out)
            except ValueError:
                return out
            if isinstance(typed, bool | int | float):
                return typed
        return out

    result = walk(data)
    if missing:
        raise ValueError("unset variables without a default: " + ", ".join(sorted(missing)))
    return result
