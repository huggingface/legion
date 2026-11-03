"""HF Jobs seam: a small protocol, the real `huggingface_hub` client, and a fake for tests."""

from __future__ import annotations

import os
import shlex
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from flock.models import TypeSpec
from flock.registry import HARNESSES

TERMINAL_STAGES = frozenset({"COMPLETED", "CANCELED", "ERROR", "DELETED"})
WORKER_COMMAND = ["flock-worker"]
# Default worker image when a type does not override `image`. Set by the deployment.
# Unset (or FLOCK_WORKER_BOOTSTRAP=1): bootstrap mode, see `bootstrap_command`.
DEFAULT_WORKER_IMAGE = os.environ.get("FLOCK_WORKER_IMAGE") or None

# Bootstrap mode: no prebuilt image. A stock Node image installs the pinned harness CLIs, uv and
# the flock wheel at job start (about 10 s, experiments/common/job_bootstrap.sh), then
# runs the worker. Versions match Dockerfile.worker, which stays the eventual path.
# Pi needs Node >= 22.19; pinned to the Node the Pi adapter was tested on.
BOOTSTRAP_IMAGE = "node:22.23.3-bookworm"
CLAUDE_CODE_VERSION = HARNESSES["claude"].version
CODEX_VERSION = HARNESSES["codex"].version
PI_VERSION = HARNESSES["pi"].version
HF_HUB_VERSION = "2.2.0"
CLAUDE_AGENT_SDK_VERSION = "0.2.166"

_BOOTSTRAP = r"""set -euo pipefail
export PATH="$HOME/.local/bin:$PATH" DEBIAN_FRONTEND=noninteractive
t0=$(date +%s)
command -v python3 >/dev/null || { apt-get update -qq && apt-get install -y -qq python3 >/dev/null; }
npm i -g --silent @anthropic-ai/claude-code@{claude} @openai/codex@{codex} >/dev/null 2>&1
if [ "${FLOCK_HARNESS:-}" = pi ]; then
  npm i -g --silent @earendil-works/pi-coding-agent@{pi} >/dev/null 2>&1
  export FLOCK_PI_CLI="$(command -v pi)"
fi
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv tool install -q huggingface_hub=={hub} >/dev/null 2>&1
mkdir -p /opt/wheels
WHEEL=/opt/wheels/$(basename "$FLOCK_WHEEL_URL")
hf buckets cp "$FLOCK_WHEEL_URL" "$WHEEL" >/dev/null 2>&1 || uv run -q --no-project \
  --with huggingface_hub=={hub} python3 -c 'import sys; from huggingface_hub import HfApi
ns, name, path = sys.argv[1].removeprefix("hf://buckets/").split("/", 2)
HfApi().download_bucket_files(f"{ns}/{name}", [(path, sys.argv[2])], raise_on_missing_files=True)
' "$FLOCK_WHEEL_URL" "$WHEEL"
uv tool install -q "$WHEEL[worker]" --with claude-agent-sdk=={sdk} --with huggingface_hub=={hub} \
  || python3 -m pip install -q --break-system-packages "$WHEEL[worker]" \
       claude-agent-sdk=={sdk} huggingface_hub=={hub}
# Agents call the controller with the same package's `flock` CLI (docs/spawn.md).
command -v flock >/dev/null || { echo "bootstrap: flock CLI is not on PATH" >&2; exit 1; }
export IS_SANDBOX=1 CLAUDE_CONFIG_DIR=/state/claude CODEX_HOME=/state/codex WORK_ROOT=/work/agents
export FLOCK_LLM_HOME=/state/llm PI_CODING_AGENT_DIR=/state/pi PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0
export FLOCK_CLAUDE_CLI="$(command -v claude)" FLOCK_CODEX_CLI="$(command -v codex)"
export PYTHONUNBUFFERED=1
mkdir -p /work/agents /state/claude /state/codex /state/pi /state/llm
if [ -n "${OPENAI_API_KEY:-}" ]; then printenv OPENAI_API_KEY | codex login --with-api-key >/dev/null; fi
echo "bootstrap: $(($(date +%s) - t0)) s; claude $(claude --version); codex $(codex --version); $(python3 --version)"
exec flock-worker
"""


def bootstrap_mode(image: str | None = DEFAULT_WORKER_IMAGE) -> bool:
    return os.environ.get("FLOCK_WORKER_BOOTSTRAP") == "1" or not image


def bootstrap_command() -> list[str]:
    """`bash -c <script>` for BOOTSTRAP_IMAGE; the job env must carry FLOCK_WHEEL_URL."""
    script = _BOOTSTRAP
    for key, value in {
        "claude": CLAUDE_CODE_VERSION,
        "codex": CODEX_VERSION,
        "pi": PI_VERSION,
        "hub": HF_HUB_VERSION,
        "sdk": CLAUDE_AGENT_SDK_VERSION,
    }.items():
        script = script.replace("{" + key + "}", shlex.quote(value))
    return ["bash", "-c", script]


@dataclass
class JobState:
    stage: str  # SCHEDULING, RUNNING, COMPLETED, CANCELED, ERROR, DELETED
    message: str | None = None
    # the job ended because its machine was taken away (a Spot reclaim, a Slurm preemption):
    # a planned restart, not a crash (reconcile._job_ended)
    preempted: bool = False

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
        network_group: str | None = None,
    ):
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        self.image = image
        self.bootstrap = bootstrap_mode(image)
        self.wheel_url = wheel_url or os.environ.get("FLOCK_WHEEL_URL")
        if self.bootstrap and not self.wheel_url:
            raise ValueError("bootstrap mode needs FLOCK_WHEEL_URL (an hf://buckets/ path)")
        # Job mode (docs/deploy-job.md): every worker joins the controller's network group and
        # reaches it directly, never through the Spaces edge or the jobs proxy.
        self.network_group = network_group
        self.calls: Counter[str] = Counter()  # Hub API calls by method, for the logs

    def launch_args(self, spec: TypeSpec, env: dict[str, str]) -> dict[str, Any]:
        """Image, command and env for one worker job (a type's own `image` always wins)."""
        if spec.image or not self.bootstrap:
            return {"image": spec.image or self.image, "command": WORKER_COMMAND, "env": env}
        env = {**env, "FLOCK_WHEEL_URL": self.wheel_url, "FLOCK_HARNESS": spec.harness.value}
        return {"image": BOOTSTRAP_IMAGE, "command": bootstrap_command(), "env": env}

    def launch(self, worker_id, spec, env, secrets, labels, timeout, *, namespace) -> str:
        self.calls["run_job"] += 1
        group = {"network_group": self.network_group} if self.network_group else {}
        job = self.api.run_job(
            **self.launch_args(spec, env),
            **group,
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

    def logs_tail(self, job_id, *, namespace, n: int = 5) -> list[str]:
        """Last `n` log lines of a finished job (one Hub call), for a crash's last_error."""
        self.calls["fetch_job_logs"] += 1
        return list(self.api.fetch_job_logs(job_id=job_id, namespace=namespace, tail=n))[-n:]

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


def hub_jobs_client(env: Mapping[str, str], dep: Any, data_dir: str) -> HubJobsClient:
    """The controller's client for compute `hf-jobs` (flock.registry)."""
    return HubJobsClient(token=env.get("HF_TOKEN"), network_group=dep.network_group)


@dataclass
class FakeJobsClient:
    """Records launches; tests flip stages with `set_stage`."""

    launches: list[dict[str, Any]] = field(default_factory=list)
    cancels: list[str] = field(default_factory=list)
    states: dict[str, JobState] = field(default_factory=dict)
    fail_launches: bool = False
    launch_error: Exception | None = None  # raised by launch() when set
    inspects: int = 0
    lists: int = 0
    logs: dict[str, list[str]] = field(default_factory=dict)  # job id -> log lines (tests)
    network_group: str | None = None  # recorded on every launch, as HubJobsClient sends it

    def launch(self, worker_id, spec, env, secrets, labels, timeout, *, namespace) -> str:
        if self.fail_launches:
            raise RuntimeError("launch failed (fake)")
        if self.launch_error is not None:
            raise self.launch_error
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
                "network_group": self.network_group,
            }
        )
        self.states[job_id] = JobState("RUNNING")
        return job_id

    def set_stage(self, job_id: str, stage: str, message: str | None = None) -> None:
        self.states[job_id] = JobState(stage, message)

    def inspect(self, job_id, *, namespace) -> JobState:
        self.inspects += 1
        return self.states.get(job_id, JobState("DELETED"))

    def logs_tail(self, job_id, *, namespace, n: int = 5) -> list[str]:
        return self.logs.get(job_id, [])[-n:]

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
