"""The harness and compute-backend registries: adding either is one module plus one entry here
(docs/design.md, "Adding a harness or a compute backend"). The spec loader, controller, worker
and CLI all read these entries. Code is named by import path ("module:function") and imported
only when used, so the controller never imports worker-only dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class HarnessEntry:
    adapter: str  # "module:function"; function(workdir, agent name, TypeSpec) -> HarnessAdapter
    # (harbor's also takes `host=`, the worker's HarborHost)
    api: str  # the wire API it speaks to its inference endpoints (preflight probes this one)


HARNESSES: dict[str, HarnessEntry] = {
    # one Harbor Trial per agent on the upstream Harbor task path (legion.harbor). Harbor installs Pi
    # (legion.harbor.config.PI_VERSION) in the task's container and the Trial delivers the
    # instructions
    "harbor": HarnessEntry("legion.worker.adapters.harbor:make_adapter", "chat_completions"),
}

# `compute.kind` / LEGION_COMPUTE -> "module:function"; function(env, deployment, data_dir) ->
# JobsClient. The function also rejects a controller setup the backend cannot run in.
COMPUTE: dict[str, str] = {
    "hf-jobs": "legion.controller.jobs:hub_jobs_client",
    "local": "legion.controller.local:local_jobs_client",  # docs/local.md
    "slurm": "legion.controller.slurm:slurm_jobs_client",  # docs/slurm.md
}

# Inference servers legion starts itself: the `inference` block's key -> "module:function";
# function(env) -> a Starter (legion.controller.servers), which starts a model's servers (or
# reattaches to the running ones), checks them until they answer, and stops them
SERVERS: dict[str, str] = {
    "hf_jobs": "legion.controller.hf_jobs_servers:hf_jobs_servers",  # docs/open-models.md
}


def load(path: str) -> Any:
    module, _, name = path.partition(":")
    return getattr(import_module(module), name)
