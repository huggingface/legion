"""The harness and compute-backend registries: adding either is one module plus one entry here
(docs/design.md, "Adding a harness or a compute backend"). The spec loader, controller, worker
and CLI all read these entries. Code is named by import path ("module:function") and imported
only when used, so the controller never imports worker-only dependencies.

Still by hand for a new harness CLI: its install line in Dockerfile.worker and in the bootstrap
script (`_BOOTSTRAP`, flock/controller/jobs.py)."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class HarnessEntry:
    adapter: str  # "module:function"; function(workdir, agent name, TypeSpec) -> HarnessAdapter
    provider_type: str  # the provider `type` a provider-backed model must have
    api: str  # the wire API it speaks to that provider (preflight probes this one)
    instructions_file: str  # the instructions file written into each agent's workdir
    version: str | None = None  # pinned CLI version (also in Dockerfile.worker); None: in-process
    sends_extra: bool = False  # it can add a model's `extra` request-body fields


HARNESSES: dict[str, HarnessEntry] = {
    "claude": HarnessEntry("flock.worker.adapters.claude:make_adapter", "anthropic", "messages",
                           "CLAUDE.md", version="2.1.296"),
    "codex": HarnessEntry("flock.worker.adapters.codex:make_adapter", "openai_compat", "responses",
                          "AGENTS.md", version="0.162.1"),
    # @earendil-works/pi-coding-agent; installed only on pi workers (~20 s)
    "pi": HarnessEntry("flock.worker.adapters.pi:make_adapter", "openai_compat", "chat_completions",
                       "AGENTS.md", version="1.2.0"),
    # in-process chat-completions loop (docs/llm-harness.md)
    "llm": HarnessEntry("flock.worker.adapters.llm:make_adapter", "openai_compat",
                        "chat_completions", "AGENTS.md", sends_extra=True),
}

# `compute.kind` / FLOCK_COMPUTE -> "module:function"; function(env, deployment, data_dir) ->
# JobsClient. The function also rejects a controller setup the backend cannot run in.
COMPUTE: dict[str, str] = {
    "hf-jobs": "flock.controller.jobs:hub_jobs_client",
    "local": "flock.controller.local:local_jobs_client",  # docs/local.md
    "slurm": "flock.controller.slurm:slurm_jobs_client",  # docs/slurm.md
}


def load(path: str) -> Any:
    module, _, name = path.partition(":")
    return getattr(import_module(module), name)
