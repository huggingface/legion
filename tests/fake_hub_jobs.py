"""A fake of the HF Jobs part of `huggingface_hub.HfApi`, for the `hf_jobs` servers
(legion.controller.hf_jobs_servers): it keeps jobs in memory and answers with the real
`JobInfo` / `JobHardwareInfo`, built from the JSON the Jobs API returns. Tests move a job's
stage with `set_stage`, and make the Hub fail with `runs_left` (run_job calls that succeed,
then a quota error), `list_errors`, `inspect_errors` and `cancel_errors` (the next N such calls
raise). No job ever runs."""

from __future__ import annotations

from datetime import UTC, datetime

from huggingface_hub import JobHardwareInfo, JobInfo

HARDWARE = {"cpu-basic": None, "l4x1": ("L4", "1"), "a100-large": ("A100", "1"),
            "a100x4": ("A100", "4")}


class FakeHubJobs:
    def __init__(self, account: str = "me"):
        self.account = account
        self.jobs: dict[str, dict] = {}  # id -> the Jobs API's JSON
        self.runs: list[dict] = []  # each run_job call's arguments
        self.cancelled: list[str] = []
        self.logs: dict[str, list[str]] = {}
        self.runs_left: int | None = None
        self.list_errors = self.inspect_errors = self.cancel_errors = 0

    def whoami(self) -> dict:
        return {"name": self.account, "type": "user"}

    def list_jobs_hardware(self) -> list[JobHardwareInfo]:
        return [JobHardwareInfo(
            name=name, prettyName=name, cpu="8 vCPU", ram="30 GB", unitCostMicroUSD=1,
            unitCostUSD=0.000001, unitLabel="minute",
            accelerator=gpu and {"type": "gpu", "model": gpu[0], "quantity": gpu[1],
                                 "vram": "80 GB", "manufacturer": "Nvidia"})
            for name, gpu in HARDWARE.items()]

    def run_job(self, *, image, command, env=None, secrets=None, flavor=None, timeout=None,
                labels=None, expose=None, namespace=None, network_group=None) -> JobInfo:
        if self.runs_left is not None:
            if self.runs_left <= 0:
                raise RuntimeError("402: jobs quota exceeded (fake)")
            self.runs_left -= 1
        job_id = f"{len(self.jobs) + 1:024x}"
        self.runs.append({"id": job_id, "image": image, "command": command, "env": env,
                          "secrets": secrets, "flavor": flavor, "timeout": timeout,
                          "labels": labels, "expose": expose, "namespace": namespace,
                          "network_group": network_group})
        self.jobs[job_id] = {
            "id": job_id, "createdAt": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "dockerImage": image,
            "command": command, "flavor": flavor, "labels": labels,
            "owner": {"id": "u1", "name": namespace, "type": "user"},
            "status": {"stage": "SCHEDULING", "message": None,
                       "exposeUrls": [f"https://{job_id}--{p}.hf.jobs" for p in expose or []]}}
        return JobInfo(**self.jobs[job_id])

    def set_stage(self, job_id: str, stage: str, message: str | None = None) -> None:
        self.jobs[job_id]["status"].update(stage=stage, message=message)

    def inspect_job(self, *, job_id, namespace=None) -> JobInfo:
        if self.inspect_errors:
            self.inspect_errors -= 1
            raise ConnectionError("the Hub did not answer (fake)")
        return JobInfo(**self.jobs[job_id])

    def list_jobs(self, *, labels=None, namespace=None) -> list[JobInfo]:
        if self.list_errors:
            self.list_errors -= 1
            raise ConnectionError("the Hub did not answer (fake)")
        return [JobInfo(**j) for j in self.jobs.values() if j["owner"]["name"] == namespace
                and all((j["labels"] or {}).get(k) == v for k, v in (labels or {}).items())]

    def cancel_job(self, *, job_id, namespace=None) -> None:
        if self.cancel_errors:
            self.cancel_errors -= 1
            raise ConnectionError("the Hub did not answer (fake)")
        self.cancelled.append(job_id)
        if self.jobs[job_id]["status"]["stage"] not in ("COMPLETED", "ERROR", "CANCELED"):
            self.set_stage(job_id, "CANCELED")

    def update_job_labels(self, *, job_id, labels, namespace=None) -> JobInfo:
        self.jobs[job_id]["labels"] = dict(labels)  # replaces them all, as the Hub does
        return JobInfo(**self.jobs[job_id])

    def fetch_job_logs(self, *, job_id, namespace=None, tail=None) -> list[str]:
        return self.logs.get(job_id, [])[-(tail or 0):]
