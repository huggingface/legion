"""`inference.hf_jobs` (flock.registry SERVERS): vLLM servers on HF Jobs, one job per replica,
behind the jobs proxy (docs/open-models.md "Servers flock starts").

Each job runs `vllm serve <model>` on `hf_jobs.image` with the port exposed token-protected
(`expose=[8000]`): the proxy serves it at `https://<job id>--8000.hf.jobs` to an HF token with
read access to the job's namespace, and drops the `Authorization` header before the server, so
vLLM runs without `--api-key`. Jobs carry the group's labels ({swarm: <swarm>, inference:
<group key>, replica: <n>}), which is how a restarted controller finds them. The controller's
HF token starts, lists, cancels and probes them; agents call them with their own key
(`api_key_env`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import httpx

from flock.controller.servers import Group, Server
from flock.models import DEFAULT_SERVER_TIMEOUT, VLLM_PORT, HFJobs, parse_duration, vllm_arg_key

log = logging.getLogger("flock.servers")

PROBE_TIMEOUT_S = 10.0
# Job env: plain logs (the progress bars' carriage returns make the log tail unreadable)
JOB_ENV = {"HF_HUB_DISABLE_PROGRESS_BARS": "1"}


def vllm_command(model: str, block: HFJobs, max_working: int | None, gpus: int) -> list[str]:
    """`vllm serve` for one replica. flock owns the address, the served name and the tool
    flags; `max_num_seqs` (the block's max_working) and `tensor_parallel_size` (the flavor's
    GPUs, divided by any data or pipeline parallel size set) are filled in unless `vllm_args`
    sets them; every `vllm_args` entry is a flag (`true`/`false`: --flag/--no-flag)."""
    given = {vllm_arg_key(e): e.split("=", 1)[1].strip() for e in block.vllm_args}
    split = int(given.get("data_parallel_size", 1)) * int(given.get("pipeline_parallel_size", 1))
    args = {"tensor_parallel_size": str(max(1, gpus // max(1, split)))}
    if max_working is not None:
        args["max_num_seqs"] = str(max_working)
    args |= given  # an entry overrides a filled-in key
    cmd = ["vllm", "serve", model, "--host", "0.0.0.0", "--port", str(VLLM_PORT),
           "--served-model-name", model,
           "--enable-auto-tool-choice", "--tool-call-parser", block.tool_call_parser]
    if block.reasoning_parser:
        cmd += ["--reasoning-parser", block.reasoning_parser]
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        cmd += [flag] if value == "true" else [f"--no-{flag[2:]}"] if value == "false" else [
            flag, value]
    return cmd


def _stage(job: Any) -> tuple[str | None, str | None]:
    status = getattr(job, "status", None)
    if status is None:
        return None, None
    stage = getattr(status.stage, "value", status.stage)
    return (str(stage) if stage is not None else None), getattr(status, "message", None)


def _epoch(t: Any) -> float | None:
    return t.timestamp() if hasattr(t, "timestamp") else None


class HFJobsServers:
    """The `hf_jobs` starter. `api` is an `HfApi` (tests pass a fake); `token` authenticates
    the readiness probes at the jobs proxy; `transport` lets tests fake the servers."""

    def __init__(self, api: Any, token: str | None,
                 transport: httpx.BaseTransport | None = None, clock=time.time):
        self.api, self.token, self.transport, self.clock = api, token, transport, clock
        self._account: str | None = None
        self._gpus: dict[str, int] = {}

    def namespace(self, group: Group) -> str:
        """Who runs and pays: the block's or spec's namespace, else the token's account
        (resolved once). Written into the group, so status shows it."""
        if not group.namespace:
            if self._account is None:
                self._account = self.api.whoami()["name"]
            group.namespace = self._account
        return group.namespace

    def gpus(self, flavor: str) -> int:
        """The flavor's GPU count (`hf jobs hardware`), cached."""
        if flavor not in self._gpus:
            hw = next((h for h in self.api.list_jobs_hardware() if h.name == flavor), None)
            if hw is None:
                raise ValueError(f"hf_jobs.flavor {flavor!r} is not an HF Jobs flavor "
                                 "(`hf jobs hardware` lists them)")
            if hw.accelerator is None:
                raise ValueError(f"hf_jobs.flavor {flavor!r} has no GPU: vLLM needs one")
            self._gpus[flavor] = int(hw.accelerator.quantity)
        return self._gpus[flavor]

    def _server(self, job: Any, replica: int) -> Server:
        urls = list(getattr(getattr(job, "status", None), "expose_urls", None) or [])
        base = urls[0] if urls else f"https://{job.id}--{VLLM_PORT}.hf.jobs"
        stage, message = _stage(job)
        return Server(replica=replica, job_id=job.id, url=base.rstrip("/") + "/v1", stage=stage,
                      message=message, started_at=_epoch(getattr(job, "created_at", None)))

    def reattach(self, group: Group) -> list[Server]:
        """The group's running jobs (by its labels), one per replica: a second job for the same
        replica (two controllers raced) is cancelled."""
        ns = self.namespace(group)
        found: dict[int, Server] = {}
        for job in self.api.list_jobs(labels=group.labels, namespace=ns):
            if (_stage(job)[0] or "") in ("COMPLETED", "CANCELED", "ERROR", "DELETED"):
                continue
            replica = int((getattr(job, "labels", None) or {}).get("replica", 0))
            if replica in found:
                self.api.cancel_job(job_id=job.id, namespace=ns)
                continue
            found[replica] = self._server(job, replica)
        return [found[r] for r in sorted(found)]

    def start(self, group: Group) -> list[Server]:
        block: HFJobs = group.block
        ns = self.namespace(group)
        servers = self.reattach(group)
        for s in [s for s in servers if s.replica >= block.replicas]:  # replicas went down
            self.api.cancel_job(job_id=s.job_id, namespace=ns)
        servers = [s for s in servers if s.replica < block.replicas]
        have = {s.replica for s in servers}
        missing = [r for r in range(block.replicas) if r not in have]
        if missing:
            command = vllm_command(group.model, block, group.max_working, self.gpus(block.flavor))
            timeout = int(parse_duration(block.timeout or DEFAULT_SERVER_TIMEOUT).total_seconds())
            for replica in missing:
                job = self.api.run_job(image=block.image, command=command, env=dict(JOB_ENV),
                                       flavor=block.flavor, timeout=timeout,
                                       labels={**group.labels, "replica": str(replica)},
                                       expose=[VLLM_PORT], namespace=ns)
                servers.append(self._server(job, replica))
        return sorted(servers, key=lambda s: s.replica)

    def check(self, group: Group, servers: list[Server]) -> None:
        ns = self.namespace(group)
        for s in servers:
            if s.ended or s.job_id is None:
                continue
            s.stage, s.message = _stage(self.api.inspect_job(job_id=s.job_id, namespace=ns))
            if s.stage == "RUNNING" and s.ready_at is None and self._answers(s.url, group.model):
                s.ready_at = self.clock()

    def _answers(self, url: str | None, model: str) -> bool:
        """GET <url>/models through the proxy lists the model (vLLM answers once loaded)."""
        if not url:
            return False
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            with httpx.Client(transport=self.transport, timeout=PROBE_TIMEOUT_S) as c:
                r = c.get(f"{url}/models", headers=headers)
            served = [m.get("id") for m in r.json().get("data") or [] if isinstance(m, dict)]
        except (httpx.HTTPError, ValueError, AttributeError):
            return False
        return r.status_code == 200 and model in served

    def logs(self, group: Group, server: Server, n: int) -> list[str]:
        if server.job_id is None:
            return []
        lines = self.api.fetch_job_logs(job_id=server.job_id, namespace=self.namespace(group),
                                        tail=n)
        return list(lines)[-n:]

    def stop(self, group: Group, servers: list[Server]) -> None:
        ns = self.namespace(group)
        for s in servers:
            if s.job_id is not None and not s.ended:
                self.api.cancel_job(job_id=s.job_id, namespace=ns)
                s.stage = "CANCELED"


def hf_jobs_servers(env: Mapping[str, str]) -> HFJobsServers:
    """The starter (flock.registry SERVERS): HF_TOKEN from `env`, else the saved login."""
    from huggingface_hub import HfApi, get_token

    token = env.get("HF_TOKEN") or get_token()
    return HFJobsServers(HfApi(token=token), token)
