"""HF Jobs seam: a small protocol, the real `huggingface_hub` client, and a fake for tests."""

from __future__ import annotations

import os
import shlex
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from swarm_harness.models import TypeSpec

TERMINAL_STAGES = frozenset({"COMPLETED", "CANCELED", "ERROR", "DELETED"})
WORKER_COMMAND = ["swarm-harness-worker"]
# Default worker image when a type does not override `image`. Set by the deployment.
# Unset (or SWARM_HARNESS_WORKER_BOOTSTRAP=1): bootstrap mode, see `bootstrap_command`.
DEFAULT_WORKER_IMAGE = os.environ.get("SWARM_HARNESS_WORKER_IMAGE") or None

# Bootstrap mode: no prebuilt image. A stock Node image installs the pinned harness CLIs, uv and
# the swarm-harness wheel at job start (about 10 s, experiments/common/job_bootstrap.sh), then
# runs the worker. Versions match Dockerfile.worker, which stays the eventual path.
BOOTSTRAP_IMAGE = "node:22-bookworm"
CLAUDE_CODE_VERSION = "2.1.296"
CODEX_VERSION = "0.162.1"
HF_HUB_VERSION = "2.2.0"
CLAUDE_AGENT_SDK_VERSION = "0.2.166"

_BOOTSTRAP = r"""set -euo pipefail
export PATH="$HOME/.local/bin:$PATH" DEBIAN_FRONTEND=noninteractive
t0=$(date +%s)
command -v python3 >/dev/null || { apt-get update -qq && apt-get install -y -qq python3 >/dev/null; }
npm i -g --silent @anthropic-ai/claude-code@{claude} @openai/codex@{codex} >/dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv tool install -q huggingface_hub=={hub} >/dev/null 2>&1
mkdir -p /opt/wheels
WHEEL=/opt/wheels/$(basename "$SWARM_HARNESS_WHEEL_URL")
hf buckets cp "$SWARM_HARNESS_WHEEL_URL" "$WHEEL" >/dev/null 2>&1 || uv run -q --no-project \
  --with huggingface_hub=={hub} python3 -c 'import sys; from huggingface_hub import HfApi
ns, name, path = sys.argv[1].removeprefix("hf://buckets/").split("/", 2)
HfApi().download_bucket_files(f"{ns}/{name}", [(path, sys.argv[2])], raise_on_missing_files=True)
' "$SWARM_HARNESS_WHEEL_URL" "$WHEEL"
uv tool install -q "$WHEEL[worker]" --with claude-agent-sdk=={sdk} --with huggingface_hub=={hub} \
  || python3 -m pip install -q --break-system-packages "$WHEEL[worker]" \
       claude-agent-sdk=={sdk} huggingface_hub=={hub}
export IS_SANDBOX=1 CLAUDE_CONFIG_DIR=/state/claude CODEX_HOME=/state/codex WORK_ROOT=/work/agents
export SWARM_HARNESS_CLAUDE_CLI="$(command -v claude)" SWARM_HARNESS_CODEX_CLI="$(command -v codex)"
export PYTHONUNBUFFERED=1
mkdir -p /work/agents /state/claude /state/codex
if [ -n "${OPENAI_API_KEY:-}" ]; then printenv OPENAI_API_KEY | codex login --with-api-key >/dev/null; fi
echo "bootstrap: $(($(date +%s) - t0)) s; claude $(claude --version); codex $(codex --version); $(python3 --version)"
exec swarm-harness-worker
"""


def bootstrap_mode(image: str | None = DEFAULT_WORKER_IMAGE) -> bool:
    return os.environ.get("SWARM_HARNESS_WORKER_BOOTSTRAP") == "1" or not image


def bootstrap_command() -> list[str]:
    """`bash -c <script>` for BOOTSTRAP_IMAGE; the job env must carry SWARM_HARNESS_WHEEL_URL."""
    script = _BOOTSTRAP
    for key, value in {
        "claude": CLAUDE_CODE_VERSION,
        "codex": CODEX_VERSION,
        "hub": HF_HUB_VERSION,
        "sdk": CLAUDE_AGENT_SDK_VERSION,
    }.items():
        script = script.replace("{" + key + "}", shlex.quote(value))
    return ["bash", "-c", script]


@dataclass
class JobState:
    stage: str  # SCHEDULING, RUNNING, COMPLETED, CANCELED, ERROR, DELETED
    message: str | None = None

    @property
    def terminal(self) -> bool:
        return self.stage in TERMINAL_STAGES


@dataclass
class JobListing:
    job_id: str
    labels: dict[str, str]
    stage: str
    env: dict[str, str]
    message: str | None = None


class JobsClient(Protocol):
    def launch(
        self,
        worker_id: str,
        spec: TypeSpec,
        env: dict[str, str],
        secrets: dict[str, str],
        labels: dict[str, str],
        timeout: float,
        *,
        namespace: str,
    ) -> str: ...

    def inspect(self, job_id: str, *, namespace: str) -> JobState: ...

    def cancel(self, job_id: str, *, namespace: str) -> None: ...

    def list(self, labels: dict[str, str], *, namespace: str) -> list[JobListing]: ...


class HubJobsClient:
    """HF Jobs via huggingface_hub. Always passes `namespace=` (saves a whoami per call)."""

    def __init__(
        self,
        token: str | None = None,
        image: str | None = DEFAULT_WORKER_IMAGE,
        wheel_url: str | None = None,
    ):
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        self.image = image
        self.bootstrap = bootstrap_mode(image)
        self.wheel_url = wheel_url or os.environ.get("SWARM_HARNESS_WHEEL_URL")
        if self.bootstrap and not self.wheel_url:
            raise ValueError("bootstrap mode needs SWARM_HARNESS_WHEEL_URL (an hf://buckets/ path)")
        self.calls: Counter[str] = Counter()  # Hub API calls by method, for the logs

    def launch_args(self, spec: TypeSpec, env: dict[str, str]) -> dict[str, Any]:
        """Image, command and env for one worker job (a type's own `image` always wins)."""
        if spec.image or not self.bootstrap:
            return {"image": spec.image or self.image, "command": WORKER_COMMAND, "env": env}
        env = {**env, "SWARM_HARNESS_WHEEL_URL": self.wheel_url}
        return {"image": BOOTSTRAP_IMAGE, "command": bootstrap_command(), "env": env}

    def launch(self, worker_id, spec, env, secrets, labels, timeout, *, namespace) -> str:
        self.calls["run_job"] += 1
        job = self.api.run_job(
            **self.launch_args(spec, env),
            secrets=secrets,
            flavor=spec.flavor,
            timeout=int(timeout),
            labels=labels,
            namespace=namespace,
        )
        return job.id

    def inspect(self, job_id, *, namespace) -> JobState:
        self.calls["inspect_job"] += 1
        job = self.api.inspect_job(job_id=job_id, namespace=namespace)
        status = job.status
        return JobState(
            stage=str(getattr(status.stage, "value", status.stage)), message=status.message
        )

    def cancel(self, job_id, *, namespace) -> None:
        self.calls["cancel_job"] += 1
        self.api.cancel_job(job_id=job_id, namespace=namespace)

    def list(self, labels, *, namespace) -> list[JobListing]:
        self.calls["list_jobs"] += 1
        out = []
        for job in self.api.list_jobs(labels=labels, namespace=namespace):
            stage = job.status.stage if job.status else "UNKNOWN"
            out.append(
                JobListing(
                    job_id=job.id,
                    labels=dict(job.labels or {}),
                    stage=str(getattr(stage, "value", stage)),
                    env=dict(job.environment or {}),
                    message=job.status.message if job.status else None,
                )
            )
        return out


@dataclass
class FakeJobsClient:
    """Records launches; tests flip stages with `set_stage`."""

    launches: list[dict[str, Any]] = field(default_factory=list)
    cancels: list[str] = field(default_factory=list)
    states: dict[str, JobState] = field(default_factory=dict)
    fail_launches: bool = False
    inspects: int = 0
    lists: int = 0

    def launch(self, worker_id, spec, env, secrets, labels, timeout, *, namespace) -> str:
        if self.fail_launches:
            raise RuntimeError("launch failed (fake)")
        job_id = f"job-{len(self.launches) + 1:04d}"
        self.launches.append(
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "spec": spec,
                "env": dict(env),
                "secrets": dict(secrets),
                "labels": dict(labels),
                "timeout": timeout,
                "namespace": namespace,
            }
        )
        self.states[job_id] = JobState("RUNNING")
        return job_id

    def set_stage(self, job_id: str, stage: str, message: str | None = None) -> None:
        self.states[job_id] = JobState(stage, message)

    def inspect(self, job_id, *, namespace) -> JobState:
        self.inspects += 1
        return self.states.get(job_id, JobState("DELETED"))

    def cancel(self, job_id, *, namespace) -> None:
        self.cancels.append(job_id)
        if job_id in self.states and not self.states[job_id].terminal:
            self.states[job_id] = JobState("CANCELED")

    def list(self, labels, *, namespace) -> list[JobListing]:
        self.lists += 1
        out = []
        for launch in self.launches:
            if launch["namespace"] != namespace:
                continue
            if all(launch["labels"].get(k) == v for k, v in labels.items()):
                state = self.states[launch["job_id"]]
                out.append(
                    JobListing(launch["job_id"], launch["labels"], state.stage, launch["env"])
                )
        return out

    def job_for(self, worker_id: str) -> str:
        """Latest job id launched for a worker (test helper)."""
        return [x["job_id"] for x in self.launches if x["worker_id"] == worker_id][-1]
