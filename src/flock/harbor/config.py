# from shepherd f3970bc:src/flock/config.py
"""The small, deliberately strict experiment interface."""

import os
import re
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PI_VERSION = "1.5.0"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TaskSettings(Settings):
    repo: str
    dataset: str | None = None
    selector: str = Field(min_length=1)
    revision: str = Field(default="main", min_length=1)

    @field_validator("repo")
    @classmethod
    def hub_repository(cls, value):
        url = urlsplit(value)
        if (
            url.scheme != "https"
            or url.netloc != "huggingface.co"
            or not re.fullmatch(r"/datasets/[\w.-]+/[\w.-]+", url.path)
            or url.query
            or url.fragment
        ):
            raise ValueError("task.repo must be an HF dataset repository URL")
        return value

    @field_validator("dataset")
    @classmethod
    def versioned_dataset(cls, value):
        if value is not None and not re.fullmatch(r"[^@\s]+@[^@\s]+", value):
            raise ValueError(
                "task.dataset must contain a name and version: name@version"
            )
        return value


class InferenceSettings(Settings):
    base_url: str = "https://router.huggingface.co/v1"
    api_key_env: str = Field(default="HF_TOKEN", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    api_format: Literal["openai_chat_completions"] = "openai_chat_completions"
    reasoning_effort: (
        Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None

    @field_validator("base_url")
    @classmethod
    def endpoint(cls, value):
        url = urlsplit(value)
        if (
            url.scheme not in ("https", "http")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "inference.base_url must be an HTTP(S) URL without credentials"
            )
        return value.rstrip("/")


class RoleSettings(Settings):
    count: int = Field(default=1, ge=1, strict=True)
    role_file: Path | None = None


RoleName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")]


class AgentSettings(Settings):
    count: int | None = Field(default=None, ge=1, strict=True)
    roles: dict[RoleName, RoleSettings] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def count_or_roles(self):
        if self.roles is not None and self.count is not None:
            raise ValueError("Set agents.count or agents.roles, not both")
        if self.roles is None and self.count is None:
            self.count = 2
        return self

    def roster(self):
        roles = (
            [name for name, role in self.roles.items() for _ in range(role.count)]
            if self.roles is not None
            else [None] * self.count
        )
        return {f"agent-{i + 1:03d}": role for i, role in enumerate(roles)}


class EnvironmentSettings(Settings):
    backend: Literal["docker", "podman"] = "docker"


class Limits(Settings):
    run_timeout_seconds: float = Field(default=900, gt=0)
    agent_max_turns: int = Field(default=30, ge=1, strict=True)
    agent_timeout_seconds: float | None = Field(default=None, gt=0)


class Config(Settings):
    name: str = Field(default="pi-team", pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    harness: Literal["pi"] = "pi"
    model: str = Field(min_length=1)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    task: TaskSettings
    team_file: Path | None = None
    resources_dir: Path = Path("resources")
    resources: dict[str, dict] = Field(default_factory=dict)
    tools_dir: Path = Path("tools")
    tools: list[str] = Field(default_factory=list)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    environment: EnvironmentSettings = Field(default_factory=EnvironmentSettings)
    limits: Limits = Field(default_factory=Limits)
    output_dir: Path = Path("results")

    @classmethod
    def load(cls, path: Path):
        path = path.resolve()
        spec = cls.model_validate(yaml.safe_load(path.read_text()))
        spec.output_dir = (path.parent / spec.output_dir.expanduser()).resolve()
        spec.tools_dir = (path.parent / spec.tools_dir.expanduser()).resolve()
        spec.resources_dir = (path.parent / spec.resources_dir.expanduser()).resolve()
        if spec.team_file is not None:
            spec.team_file = (path.parent / spec.team_file.expanduser()).resolve()
        for role in (spec.agents.roles or {}).values():
            if role.role_file is not None:
                role.role_file = (path.parent / role.role_file.expanduser()).resolve()
        return spec

    def check_credentials(self):
        if not os.environ.get(self.inference.api_key_env):
            raise ValueError(
                f"Set {self.inference.api_key_env} before running an experiment"
            )

    def check_model_connection(self):
        from harbor.agents.installed.pi import Pi

        Pi.resolve_model_connection_config(
            self.model,
            lambda key, *others: next(
                (
                    (name, os.environ[name])
                    for name in (key, *others)
                    if os.environ.get(name)
                ),
                None,
            ),
            base_url=self.inference.base_url,
            model_api_key_name=self.inference.api_key_env,
            model_api_format=self.inference.api_format,
        )
