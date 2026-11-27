"""An env world (docs/world.md, "Env worlds"): World API v1 tools that the gateway
(resources.gateway) serves on the world route by forwarding each call to an upstream HTTP API,
with the calling agent's own upstream credential added there.

The run's spec.yaml configures it under `resources.world.env`. Each agent's credential is
derived from the run's secret (`secret_env`) and its name, so a gateway needs no list of them.
Agents never hold one: a call acts only as its caller, on the routes the config names, and every
credential is redacted from results and errors.

The env process (the upstream) gets the credentials from this module's launcher, which runs a
command with seat i's credential in `credential_env` (agent i+1 of the roster, `naming:
consecutive-id`) and without the secret:

    python -m legion.harbor.env spec.yaml -- <command> [args...]
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from legion.harbor.spec import Agents, substitute
from legion.models import load_spec_yaml
from legion.world import ToolError
from legion.world.protocol import WORLD_API, WorldManifest, base_url
from legion.world.server import redact

ENV_NAME = r"^[A-Za-z_][A-Za-z0-9_]*$"
CREDENTIAL = "{credential}"  # in a header value: the calling agent's credential


class Route(BaseModel):
    """One tool: a fixed upstream route. Arguments named in `query` go in the query string, the
    rest in a JSON body (a GET has none)."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str = Field(pattern=r"^/[A-Za-z0-9._~/-]*$")
    query: list[str] = Field(default_factory=list)
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}})
    read_only: bool = False

    @field_validator("path")
    @classmethod
    def _path(cls, v: str) -> str:
        if any(seg in (".", "..") for seg in v.split("/")):  # httpx would resolve them
            raise ValueError("path must not have . or .. segments")
        return v

    @model_validator(mode="after")
    def _args(self):
        props = set(self.parameters.get("properties") or {})
        if unknown := set(self.query) - props:
            raise ValueError(f"query names arguments not in parameters: {sorted(unknown)}")
        if self.method == "GET" and (body := props - set(self.query)):
            raise ValueError(f"a GET has no body: list {sorted(body)} in query")
        return self


class Config(BaseModel):
    """`resources.world.env` in a run's spec.yaml."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    name: str
    description: str = ""
    upstream: str  # may hold ${HF_NETWORK_GROUP_PREFIX} (`$${` in the spec): the worker expands it
    secret_env: str = Field(pattern=ENV_NAME)  # the run's secret; never in the spec
    credential_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*\{seat\}[A-Za-z0-9_]*$")
    headers: dict[str, str]
    tools: dict[str, Route] = Field(min_length=1)

    @field_validator("upstream")
    @classmethod
    def _upstream(cls, v: str) -> str:
        return base_url(v, "upstream")

    @field_validator("headers")
    @classmethod
    def _headers(cls, v: dict[str, str]) -> dict[str, str]:
        if not any(CREDENTIAL in value for value in v.values()):
            raise ValueError(f"no header carries {CREDENTIAL}")
        return v

    @model_validator(mode="after")
    def _manifest(self):
        WorldManifest.model_validate(self.manifest())  # name, tool names, schemas
        return self

    def manifest(self) -> dict[str, Any]:
        tools = [{"name": name, "description": r.description, "parameters": r.parameters,
                  **({"annotations": {"readOnlyHint": True}} if r.read_only else {})}
                 for name, r in self.tools.items()]
        # the version changes whenever the tools do
        version = hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()[:12]
        return {"world_api": WORLD_API, "name": self.name, "version": version,
                "description": self.description, "tools": tools, "capabilities": [], "hooks": []}


def credential(secret: str, agent: str) -> str:
    """`agent`'s upstream credential: what a gateway adds to its calls and the launcher hands
    the env process, so neither needs the other's list."""
    return hmac.new(secret.encode(), f"legion-env-credential\0{agent}".encode(),
                    hashlib.sha256).hexdigest()


def load(path: Path, env=None) -> tuple[Config, list[str]]:
    """The config and the roster (seat order) from the run's spec.yaml at `path`. Only the env
    block and `agents` are read (their ${VAR}s filled from `env`)."""
    env = os.environ if env is None else env
    data = load_spec_yaml(path.read_text()) or {}
    block = ((data.get("resources") or {}).get("world") or {}).get("env")
    if block is None:
        raise ValueError(f"{path} has no resources.world.env")
    if data.get("naming") != "consecutive-id":
        raise ValueError("an env world needs naming: consecutive-id (agent i of the roster "
                         "has seat i-1)")
    config = Config.model_validate(substitute(block, env))
    return config, list(Agents.model_validate(substitute(data.get("agents") or {}, env)).roster())


async def call(config: Config, client: httpx.AsyncClient, tool: str, args: dict[str, Any],
               credential: str, secrets: list[str]) -> Any:
    """`tool` with checked `args`, sent upstream with `credential`; the result, or a ToolError.
    Both are redacted with `secrets` (longest first)."""
    route = config.tools[tool]
    headers = {k: v.replace(CREDENTIAL, credential) for k, v in config.headers.items()}
    query = {k: v for k, v in args.items() if k in route.query}
    body = {k: v for k, v in args.items() if k not in route.query}
    try:
        r = await client.request(route.method, route.path, params=query, headers=headers,
                                 json=None if route.method == "GET" else body)
    except httpx.HTTPError as e:  # its text may quote a header: the type only
        raise ToolError(f"upstream unreachable ({type(e).__name__})", "upstream_unreachable",
                        retry_after_s=5) from None
    try:
        result = r.json()
    except ValueError:
        result = r.text
    result = redact(result, secrets)
    if r.is_success:
        return result
    retry = r.headers.get("retry-after", "")
    raise ToolError(f"upstream answered {r.status_code}: "
                    f"{result if isinstance(result, str) else json.dumps(result)}"[:1000],
                    f"upstream_{r.status_code}",
                    retry_after_s=float(retry) if retry.isdigit() else None)


def main(argv: list[str] | None = None) -> None:
    """`<spec.yaml> -- <command> [args...]`: run the command with each seat's credential."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3 or argv[1] != "--":
        raise SystemExit("usage: python -m legion.harbor.env <spec.yaml> -- <command> [args...]")
    config, names = load(Path(argv[0]))
    secret = os.environ.get(config.secret_env, "")
    if len(secret) < 16:
        raise SystemExit(f"Set {config.secret_env} to the run's secret (16+ characters)")
    env = {k: v for k, v in os.environ.items() if k != config.secret_env}
    env |= {config.credential_env.format(seat=i): credential(secret, name)
            for i, name in enumerate(names)}
    os.execvpe(argv[2], argv[2:], env)


if __name__ == "__main__":
    main()
