"""Servers flock starts for `inference.hf_jobs` (controller/servers.py, hf_jobs_servers.py),
through the controller's API, with the Hub's jobs faked (tests/fake_hub_jobs.py) and each job's
vLLM behind a fake jobs proxy: the launch and its readiness wait, the pinned urls and context
window, a restarted controller reattaching, cancel, failures; roles with their own model or
settings; the vllm_args rules."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from flock import ops
from flock.client import Client
from flock.controller.api import create_app
from flock.controller.hf_jobs_servers import HFJobsServers
from flock.controller.servers import Servers
from flock.models import DEFAULT_VLLM_IMAGE, SwarmSpec

from .conftest import HB, TOKEN, wire
from .fake_hub_jobs import FakeHubJobs

MODEL = "Qwen/Qwen3.8-27B"
HF_JOBS = {"flavor": "a100-large", "tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3",
           "vllm_args": ["max_model_len=131072"], "timeout": "2h"}
KEY = "hf-controller"  # HF_TOKEN on the controller: starts the jobs, passes the proxy


class Proxy:
    """The jobs proxy and each job's vLLM: `GET /v1/models` needs the HF token, answers 503
    until the test puts the job in `up`, then lists the model the job serves."""

    def __init__(self, hub: FakeHubJobs):
        self.hub, self.up = hub, set()

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            job = request.url.host.split("--")[0]
            if request.headers.get("authorization") != f"Bearer {KEY}":
                return httpx.Response(401)
            if job not in self.up:
                return httpx.Response(503)
            served = self.hub.jobs[job]["command"][2]
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": served, "object": "model", "max_model_len": 131072}]})
        return httpx.MockTransport(handle)


@pytest.fixture
def hub() -> FakeHubJobs:
    return FakeHubJobs()


@pytest.fixture
def proxy(hub) -> Proxy:
    return Proxy(hub)


@pytest.fixture
def controller(store, hub, proxy, clock):
    """A controller on `store` (call it again: a restarted one), and a client for it."""
    def make():
        epoch = lambda: clock().timestamp()
        servers = Servers({}, {"hf_jobs": HFJobsServers(hub, KEY, proxy.transport(), epoch)},
                          clock=epoch)
        app = create_app(store, TOKEN, clock=clock, secret_source={"HF_TOKEN": KEY},
                         inference_transport=proxy.transport(), servers=servers,
                         servers_interval=0, loop_interval=0)
        return app, TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
    return make


def spec(inference=None, **types) -> dict:
    types = types or {"agent": {**HB, "model": MODEL, "replicas": 3}}
    return SwarmSpec.model_validate({
        "swarm": "s", "compute": "local", "types": types,
        "inference": inference or {"hf_jobs": HF_JOBS, "max_working": 3}}).model_dump(mode="json")


def test_launch_wait_pin_reattach_and_cancel(controller, hub, proxy, store, clock):
    app, c = controller()
    body = spec()
    r = c.put("/swarms/s", json=body)
    assert r.status_code == 400 and "hf_jobs servers are not started" in r.text
    [row] = c.post("/swarms/s/servers", json=body).json()
    [run] = hub.runs
    job = run["id"]
    assert run["command"] == [
        "vllm", "serve", MODEL, "--host", "0.0.0.0", "--port", "8000", "--served-model-name",
        MODEL, "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3", "--tensor-parallel-size", "1", "--max-num-seqs", "3",
        "--max-model-len", "131072"]
    assert (run["image"], run["flavor"], run["expose"], run["timeout"], run["namespace"]) == (
        DEFAULT_VLLM_IMAGE, "a100-large", [8000], 7200, "me")  # namespace: the token's account
    assert run["labels"] == {"swarm": "s", "inference": row["group"], "replica": "0"}
    assert (row["state"], row["stage"], row["namespace"]) == ("starting", "SCHEDULING", "me")

    servers = app.state.servers
    hub.set_stage(job, "RUNNING")
    servers.tick()  # running, vLLM still loading (503)
    assert c.get("/swarms/s/servers").json()[0]["state"] == "starting"
    assert c.put("/swarms/s", json=body).status_code == 400
    proxy.up.add(job)
    servers.tick()
    [row] = c.get("/swarms/s/servers").json()
    assert (row["state"], row["url"]) == ("ready", f"https://{job}--8000.hf.jobs/v1")

    assert c.put("/swarms/s", json=body).status_code == 200  # the rest is the `urls` path
    pinned = store.get_swarm("s")["spec"].types["agent"].resolved_model
    assert (pinned.inference.urls, pinned.context_window) == ([row["url"]], 131072)
    status = c.get("/swarms/s").json()["status"]
    assert status["providers"]["inference"]["models"] == {MODEL: 131072}
    assert [(s["job_id"], s["state"], s["flavor"]) for s in status["servers"]] == [
        (job, "ready", "a100-large")]
    assert c.put("/swarms/s", json=body).status_code == 200  # a re-apply keeps them

    app, c = controller()  # a restarted controller: its first tick finds the job by its labels
    app.state.servers.tick()
    assert [(s["job_id"], s["state"]) for s in c.get("/swarms/s/servers").json()] == [
        (job, "ready")]
    c.post("/swarms/s/servers", json=body)  # `flock launch` again: nothing new starts
    assert len(hub.runs) == 1 and hub.cancelled == []
    assert c.delete("/swarms/s").json()["cancelled"] == [job]  # the run ends: cancelled
    assert hub.jobs[job]["status"]["stage"] == "CANCELED"


def test_each_model_gets_its_own_servers(controller, hub, proxy, store):
    """A role with its own model gets jobs with the run's settings; one with its own hf_jobs
    block, its own. Each type is pinned to its group's servers."""
    app, c = controller()
    body = spec(None, agent={**HB, "model": MODEL, "replicas": 2},
                helper={**HB, "model": "Qwen/Small", "replicas": 1},
                big={**HB, "model": MODEL, "replicas": 1,
                     "inference": {"hf_jobs": {**HF_JOBS, "flavor": "a100x4", "replicas": 2}}})
    rows = c.post("/swarms/s/servers", json=body).json()
    assert sorted((r["model"], r["flavor"], r["replica"], tuple(r["types"])) for r in rows) == [
        (MODEL, "a100-large", 0, ("agent",)), (MODEL, "a100x4", 0, ("big",)),
        (MODEL, "a100x4", 1, ("big",)), ("Qwen/Small", "a100-large", 0, ("helper",))]
    by_flavor = {(r["flavor"], r["command"][2]): r["command"] for r in hub.runs}
    assert "--max-num-seqs" not in by_flavor["a100x4", MODEL]  # its block has no max_working
    assert by_flavor["a100x4", MODEL][by_flavor["a100x4", MODEL].index(
        "--tensor-parallel-size") + 1] == "4"  # the flavor's GPUs
    for run in hub.runs:
        hub.set_stage(run["id"], "RUNNING")
        proxy.up.add(run["id"])
    app.state.servers.tick()
    assert c.put("/swarms/s", json=body).status_code == 200
    urls = {n: ts.resolved_model.inference.urls
            for n, ts in store.get_swarm("s")["spec"].types.items()}
    job = {(r["flavor"], r["command"][2], r["labels"]["replica"]): r["id"] for r in hub.runs}
    url = lambda *k: f"https://{job[k]}--8000.hf.jobs/v1"
    assert urls == {"agent": [url("a100-large", MODEL, "0")],
                    "helper": [url("a100-large", "Qwen/Small", "0")],
                    "big": [url("a100x4", MODEL, "0"), url("a100x4", MODEL, "1")]}


def test_a_changed_block_moves_the_type_to_new_servers(controller, hub, proxy, store, clock,
                                                        monkeypatch):
    """`flock update` of the hf_jobs block starts new servers and the apply pins them (a block
    change: the type's workers restart on them); the old ones serve the draining workers' turns
    until turn_timeout, then are cancelled."""
    app, c = controller()

    def up(body) -> str:
        c.post("/swarms/s/servers", json=body)
        job = hub.runs[-1]["id"]
        hub.set_stage(job, "RUNNING")
        proxy.up.add(job)
        app.state.servers.tick()
        r = c.put("/swarms/s", json=body)
        assert r.status_code == 200, r.text
        return job

    old = up(spec())
    body = spec({"hf_jobs": {**HF_JOBS, "flavor": "a100x4"}, "max_working": 3})
    new = up(body)
    pinned = store.get_swarm("s")["spec"].types["agent"].resolved_model.inference
    assert (pinned.hf_jobs.flavor, pinned.urls) == ("a100x4", [f"https://{new}--8000.hf.jobs/v1"])
    assert hub.cancelled == []
    clock.advance(30 * 60)  # turn_timeout
    app.state.servers.tick()
    assert hub.cancelled == [old]
    assert [(r["job_id"], r["state"]) for r in c.get("/swarms/s/servers").json()] == [
        (old, "stopped"), (new, "ready")]
    wire(monkeypatch, app)  # `flock update` again: the stopped group is no longer the spec's
    assert [r["job_id"] for r in ops.start_servers(Client(), SwarmSpec.model_validate(body))] == [
        new]


def test_vllm_args_rules(controller, hub):
    """Keys flock owns are refused; max_num_seqs and tensor_parallel_size are filled in unless
    vllm_args sets them; an entry overrides by key, `true`/`false` are flags."""
    for entry, needle in [("port=9000", "flock sets it"), ("served-model-name=x", "flock sets it"),
                          ("tool_call_parser=hermes", "set hf_jobs.tool_call_parser"),
                          ("api_key=k", "drops Authorization"), ("enforce_eager", "KEY=VALUE")]:
        with pytest.raises(ValidationError, match=needle):
            spec({"hf_jobs": {**HF_JOBS, "vllm_args": [entry]}})
    with pytest.raises(ValidationError, match="max_model_len is set twice"):
        spec({"hf_jobs": {**HF_JOBS, "vllm_args": ["max_model_len=1", "max-model-len=2"]}})
    _, c = controller()
    args = ["max-num-seqs=8", "data_parallel_size=2", "enforce_eager=true",
            "enable_prefix_caching=false"]
    c.post("/swarms/s/servers", json=spec(
        {"hf_jobs": {**HF_JOBS, "flavor": "a100x4", "vllm_args": args}, "max_working": 3}))
    command = " ".join(hub.runs[0]["command"])
    assert command.endswith("--tensor-parallel-size 2 --max-num-seqs 8 --data-parallel-size 2 "
                            "--enforce-eager --no-enable-prefix-caching")


def test_failures_stop_the_apply_and_cancel(controller, hub, proxy, clock):
    """A job that ends while starting (an OOM, an image pull) and one that never answers within
    startup_timeout fail their group with the reason and the job's last log lines, cancel what
    runs, and the apply says why."""
    app, c = controller()
    body = spec()
    job = c.post("/swarms/s/servers", json=body).json()[0]["job_id"]
    hub.set_stage(job, "ERROR", "exit code 1")
    hub.logs[job] = ["Loading weights", "torch.OutOfMemoryError: CUDA out of memory."]
    app.state.servers.tick()
    [row] = c.get("/swarms/s/servers").json()
    assert row["state"] == "failed" and row["error"] == (
        f"job {job} ERROR: exit code 1; last log lines: Loading weights | "
        "torch.OutOfMemoryError: CUDA out of memory.")
    r = c.put("/swarms/s", json=body)
    assert r.status_code == 400 and "CUDA out of memory" in r.text
    assert c.get("/swarms/s/servers").json()[0]["state"] == "failed"

    job = c.post("/swarms/s/servers", json=body).json()[0]["job_id"]  # a new launch: a new job
    hub.set_stage(job, "RUNNING")
    clock.advance(20 * 60 + 1)  # startup_timeout 20m: never answered
    app.state.servers.tick()
    [row] = c.get("/swarms/s/servers").json()
    assert row["state"] == "failed" and row["error"].startswith(
        f"not ready after startup_timeout 20m: job {job} RUNNING")
    assert hub.cancelled == [job]
