"""flock's run spec: shepherd's `Config` plus a prompting loop, ramps, pricing and hf-sandbox.

One `spec.yaml` (plus its prompt files) per run. A shepherd config loads unchanged.
`Spec.load` fills `${VAR}` / `${VAR:-default}` in string values and inlines prompt files.
"""

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BeforeValidator, ConfigDict, Field, model_validator

from .config import (
    AgentSettings,
    Config,
    EnvironmentSettings,
    Limits,
    RoleName,
    RoleSettings,
    Settings,
)

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def seconds(value):
    """'30s', '10m', '48h', '2d' or a number of seconds -> seconds."""
    if isinstance(value, int | float):
        return value
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", str(value))
    if not m:
        raise ValueError(f"bad duration {value!r}; use e.g. 30s, 10m, 48h")
    return float(m[1]) * _UNITS[m[2]]


Duration = Annotated[float, BeforeValidator(seconds), Field(gt=0)]


class Ramp(Settings):
    """Start `start` agents of the role, then `add` more every `every`, up to the role's count."""

    start: int = Field(ge=0, strict=True)
    add: int = Field(ge=1, strict=True)
    every: Duration


class Role(RoleSettings):
    ramp: Ramp | None = None

    @model_validator(mode="after")
    def ramp_within_count(self):
        if self.ramp is not None and self.ramp.start > self.count:
            raise ValueError("ramp.start must be <= count")
        return self


class Agents(AgentSettings):
    roles: dict[RoleName, Role] | None = Field(default=None, min_length=1)


class Environment(EnvironmentSettings):
    backend: Literal["docker", "podman", "hf-sandbox"] = "docker"
    kwargs: dict[str, Any] = Field(default_factory=dict)  # Harbor's EnvironmentConfig.kwargs


class SpecLimits(Limits):
    trial_hours: float = Field(default=23, gt=0)  # a new Trial after this; hf-sandbox caps at 24 h
    budget_usd: float | None = Field(default=None, ge=0)  # the whole run


class _Prompted(Settings):
    prompt: str | None = None
    prompt_file: Path | None = None  # inlined into `prompt` by Spec.load

    @model_validator(mode="after")
    def one_prompt(self):
        if (self.prompt is None) == (self.prompt_file is None):
            raise ValueError("set prompt or prompt_file, not both")
        return self


class While(Settings):
    """The loop goes on while every set condition holds."""

    until: Duration | None = None  # since the run started
    max_rounds: int | None = Field(default=None, ge=1, strict=True)  # loop prompts per agent
    agent_usd: float | None = Field(default=None, ge=0)


class Final(_Prompted):
    reserve_usd: float = Field(default=0, ge=0)  # held back from limits.budget_usd


class Loop(_Prompted):
    model_config = ConfigDict(validate_by_name=True, serialize_by_alias=True)

    # Seconds between loop prompts; turn_end (0) sends the next one as soon as a turn ends.
    every: Annotated[
        float, BeforeValidator(lambda v: 0 if v == "turn_end" else seconds(v)), Field(ge=0)
    ] = 0
    while_: While = Field(default_factory=While, alias="while")
    final: Final | None = None
    resume_prompt: str = (
        "Your session was restarted. Check your notes and state, and continue your work."
    )


class Spec(Config):
    agents: Agents = Field(default_factory=Agents)
    environment: Environment = Field(default_factory=Environment)
    limits: SpecLimits = Field(default_factory=SpecLimits)
    loop: Loop | None = None
    # [input, output, cached] USD per 1M tokens; Harbor reports no cost for router models.
    pricing: (
        Annotated[list[Annotated[float, Field(ge=0)]], Field(min_length=3, max_length=3)] | None
    ) = None

    @model_validator(mode="after")
    def checks(self):
        loop = self.loop
        if loop and loop.while_.until and self.limits.run_timeout_seconds <= loop.while_.until:
            raise ValueError("limits.run_timeout_seconds must be longer than loop.while.until")
        if loop is None and any(r.ramp for r in (self.agents.roles or {}).values()):
            raise ValueError("ramp needs loop")
        if (
            self.environment.backend == "hf-sandbox"
            and "board" in self.resources
            and not self.resources["board"].get("url")
        ):
            raise ValueError("hf-sandbox has no mounts: set resources.board.url")
        budgets = self.limits.budget_usd is not None or (loop and loop.while_.agent_usd is not None)
        if budgets and self.pricing is None:
            raise ValueError("budgets need pricing: [input, output, cached] USD per 1M tokens")
        return self

    @classmethod
    def load(cls, path: Path, env=None):
        """`env` fills `${VAR}` (default: the process environment)."""
        path = Path(path).resolve()
        here = path.parent
        spec = cls.model_validate(
            substitute(yaml.safe_load(path.read_text()), os.environ if env is None else env)
        )
        # as shepherd's Config.load
        spec.output_dir = (here / spec.output_dir.expanduser()).resolve()
        spec.tools_dir = (here / spec.tools_dir.expanduser()).resolve()
        spec.resources_dir = (here / spec.resources_dir.expanduser()).resolve()
        if spec.team_file is not None:
            spec.team_file = (here / spec.team_file.expanduser()).resolve()
        for role in (spec.agents.roles or {}).values():
            if role.role_file is not None:
                role.role_file = (here / role.role_file.expanduser()).resolve()
        for prompted in (spec.loop, spec.loop and spec.loop.final):
            if prompted and prompted.prompt_file is not None:
                prompted.prompt = (here / prompted.prompt_file.expanduser()).read_text(
                    encoding="utf-8"
                )
                prompted.prompt_file = None  # run.json carries the text
        return spec


# From flock's specrender: only UPPERCASE names are variables, so `{spend}` and `${spend}` in
# prompts pass through; `$${` is a literal `${`.
_VAR = re.compile(r"\$\$(?=\{)|\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def substitute(data, env):
    """Fill placeholders in every string value. A string that is one placeholder takes the
    value's YAML type (`"${N:-20}"` -> 20). Raises naming every unset variable without a default."""
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
                typed = yaml.safe_load(out)
            except yaml.YAMLError:
                return out
            if isinstance(typed, bool | int | float):
                return typed
        return out

    result = walk(data)
    if missing:
        raise ValueError("unset variables without a default: " + ", ".join(sorted(missing)))
    return result
