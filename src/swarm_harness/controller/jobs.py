"""HF Jobs seam: a small protocol, the real `huggingface_hub` client, and a fake for tests."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from swarm_harness.models import TypeSpec

TERMINAL_STAGES = frozenset({"COMPLETED", "CANCELED", "ERROR", "DELETED"})
WORKER_COMMAND = ["swarm-harness-worker"]
# Default worker image when a type does not override `image`. Set by the deployment.
DEFAULT_WORKER_IMAGE = os.environ.get(
    "SWARM_HARNESS_WORKER_IMAGE", "ghcr.io/huggingface/swarm-harness-worker:latest"
)


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

    def __init__(self, token: str | None = None, image: str = DEFAULT_WORKER_IMAGE):
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        self.image = image

    def launch(self, worker_id, spec, env, secrets, labels, timeout, *, namespace) -> str:
        job = self.api.run_job(
            image=spec.image or self.image,
            command=WORKER_COMMAND,
            env=env,
            secrets=secrets,
            flavor=spec.flavor,
            timeout=int(timeout),
            labels=labels,
            namespace=namespace,
        )
        return job.id

    def inspect(self, job_id, *, namespace) -> JobState:
        job = self.api.inspect_job(job_id=job_id, namespace=namespace)
        status = job.status
        return JobState(
            stage=str(getattr(status.stage, "value", status.stage)), message=status.message
        )

    def cancel(self, job_id, *, namespace) -> None:
        self.api.cancel_job(job_id=job_id, namespace=namespace)

    def list(self, labels, *, namespace) -> list[JobListing]:
        out = []
        for job in self.api.list_jobs(labels=labels, namespace=namespace):
            stage = job.status.stage if job.status else "UNKNOWN"
            out.append(
                JobListing(
                    job_id=job.id,
                    labels=dict(job.labels or {}),
                    stage=str(getattr(stage, "value", stage)),
                    env=dict(job.environment or {}),
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
