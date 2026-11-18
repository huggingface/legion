"""Launch-time bucket resolution and billing through real specs, CLI and controller."""

import json

import httpx
import pytest
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from pydantic import ValidationError
from typer.testing import CliRunner

from flock import cli, rundir
from flock.client import Client
from flock.controller.jobs import HubJobsClient
from flock.controller.localrun import state_env
from flock.controller.reconcile import reconcile
from flock.models import SwarmSpec
from flock.ops import launch, read_spec, up, update
from flock.worker.sync import HubBucketStore

from .conftest import TOKEN, make_spec, wire, write_spec
from .fake_hub_jobs import FakeHubJobs


class BillingHub(FakeHubJobs):
    def __init__(self):
        super().__init__(account="launcher")
        self.whoamis = 0
        self.buckets = []
        self.uploads = []
        self.tokens = []
        self.fail = False

    def whoami(self):
        self.whoamis += 1
        return super().whoami()

    def create_bucket(self, bucket_id, *, private, exist_ok):
        if self.fail:
            raise PermissionError("403: cannot create bucket")
        self.buckets.append((bucket_id, private, exist_ok))

    def batch_bucket_files(self, bucket_id, *, add):
        self.uploads.append((bucket_id, [r for _, r in add]))


@pytest.mark.parametrize(("namespace", "bucket_id", "qualified", "lookups"), [
    ("", "runs", "launcher/runs", 1),
    ("org", "runs", "org/runs", 0),
    ("org", "storage/runs", "storage/runs", 0),
    ("", "storage/runs", "storage/runs", 0),
])
def test_launch_resolves_and_creates_bucket_before_workers(
        api, app, store, settings, clock, monkeypatch, tmp_path,
        namespace, bucket_id, qualified, lookups):
    hub = BillingHub()

    def hf_api(token=None):
        hub.tokens.append(token)
        return hub

    monkeypatch.setattr("huggingface_hub.HfApi", hf_api)
    wire(monkeypatch, app)
    monkeypatch.setenv("HF_TOKEN", "infrastructure-key")
    spec = SwarmSpec.model_validate({**make_spec().model_dump(mode="json"),
                                     "namespace": namespace, "bucket_id": bucket_id})
    path = write_spec(tmp_path / "spec.yaml", spec)
    result = CliRunner().invoke(cli.app, ["launch", path, "--skip-preflight"])
    assert result.exit_code == 0, result.output
    stored = store.get_swarm("demo")["spec"]
    assert stored.bucket_id == qualified and stored.namespace == namespace
    assert hub.buckets == [(qualified, True, True)] and hub.whoamis == lookups
    assert hub.tokens == ["infrastructure-key"]
    assert read_spec(path).bucket_id == bucket_id  # source YAML stays as written

    # An unchanged source still compares equal to the stored, qualified spec on update.
    assert not update(api, read_spec(path), path, preflight=False)["diff"]
    assert hub.whoamis == lookups * 2
    jobs = HubJobsClient(token="infrastructure-key", image="worker")
    reconcile(store, jobs, clock(), settings)
    assert len(hub.runs) == 2
    assert {r["namespace"] for r in hub.runs} == {namespace or "launcher"}
    assert {r["env"]["BUCKET_ID"] for r in hub.runs} == {qualified}
    assert store.get_swarm("demo")["spec"].namespace == namespace

    # Checkpoint and agent prefixes work on the resolved ID without a further whoami.
    env, _ = state_env({"HF_TOKEN": "infrastructure-key"}, stored)
    before = hub.whoamis
    checkpoint = HubBucketStore(env["FLOCK_STATE_URL"], token="infrastructure-key")
    checkpoint.upload([(tmp_path / "state.db", "state.db")])
    agents = HubBucketStore(stored.bucket_id, token="infrastructure-key")
    agents.upload([(tmp_path / "session", "demo/researcher/a/session")])
    assert hub.uploads == [(qualified, ["controller/demo/state.db"]),
                           (qualified, ["demo/researcher/a/session"])]
    assert hub.whoamis == before


def test_bucket_failure_refuses_launch_before_jobs(api, app, store, monkeypatch, tmp_path):
    hub = BillingHub()
    hub.fail = True
    monkeypatch.setattr("huggingface_hub.HfApi", lambda **_: hub)
    wire(monkeypatch, app)
    path = write_spec(tmp_path / "spec.yaml", make_spec())
    result = CliRunner().invoke(cli.app, ["launch", path, "--skip-preflight"])
    assert result.exit_code == 1 and "cannot create bucket" in result.output
    assert not store.get_swarm("demo") and not hub.runs


def test_update_resolves_a_bare_name_after_explicit_storage_namespace(api, tmp_path):
    data = {**make_spec().model_dump(mode="json"), "namespace": "org", "bucket_id": "storage/runs"}
    api.create(SwarmSpec.model_validate(data))
    path = write_spec(tmp_path / "spec.yaml", SwarmSpec.model_validate({**data, "bucket_id": "runs"}))
    assert update(api, read_spec(path), path, preflight=False)["applied"]
    assert api.swarm("demo")["spec"]["bucket_id"] == "org/runs"


@pytest.mark.parametrize("token_file", ["cluster.env", "run.env", "explicit.env"])
@pytest.mark.parametrize("saved_login", ["saved-login-user", None])
def test_cli_update_uses_launch_env_files_without_retargeting_bucket(
        app, store, tmp_path, monkeypatch, token_file, saved_login):
    wire(monkeypatch, app)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("FLOCK_RUN_DIR", raising=False)
    monkeypatch.delenv("FLOCK_URL", raising=False)
    tokens = []

    def hf_api(token=None):
        tokens.append(token)
        if token != "file-only-token" and saved_login is None:
            raise RuntimeError("no saved login")
        hub = BillingHub()
        hub.account = "launcher" if token == "file-only-token" else saved_login
        return hub

    monkeypatch.setattr("huggingface_hub.HfApi", hf_api)
    run = tmp_path / "run"
    (run / "hosts").mkdir(parents=True)
    (run / "hosts" / "controller.host").write_text("http://controller.invalid\n")
    (run / "run.env").write_text(f"FLOCK_TOKEN={TOKEN}\n")
    with (run / token_file).open("a") as f:
        f.write("HF_TOKEN=file-only-token\n")
    args = ["--env", str(run / token_file)] if token_file == "explicit.env" else []
    spec = SwarmSpec.model_validate({**make_spec().model_dump(),
                                     "namespace": "", "bucket_id": "runs"})
    path = write_spec(tmp_path / "spec.yaml", spec)
    runner = CliRunner()
    launched = runner.invoke(cli.app, ["--run", str(run), "launch", path,
                                      "--skip-preflight", *args])
    assert launched.exit_code == 0, launched.output
    assert store.get_swarm("demo")["spec"].bucket_id == "launcher/runs"
    for check in (["--check"], []):
        updated = runner.invoke(cli.app, ["--run", str(run), "update", path,
                                         "--skip-preflight", *check, *args])
        assert updated.exit_code == 0, updated.output
        assert "bucket_id:" not in updated.output
        assert store.get_swarm("demo")["spec"].bucket_id == "launcher/runs"
    assert tokens == ["file-only-token"] * 3


def test_local_start_uses_the_resolved_bucket_and_token_from_env_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "run.env"
    source.write_text("HF_TOKEN=infrastructure-key\n")
    spec = SwarmSpec.model_validate({**make_spec().model_dump(), "compute": "local"})

    def start(_cmd, **kwargs):
        assert kwargs["env"]["HF_TOKEN"] == "infrastructure-key"
        assert kwargs["env"]["FLOCK_STATE_URL"] == "hf://buckets/ns/sh/controller/demo"
        raise RuntimeError("checked subprocess environment")

    monkeypatch.setattr("subprocess.Popen", start)
    with pytest.raises(RuntimeError, match="checked subprocess environment"):
        up(Client(env={}), spec, "spec.yaml", env_files=[source])


@pytest.mark.parametrize("bucket_id", ["hf://buckets/ns/runs", "ns/runs/prefix", "/runs",
                                       "ns/", "ns//runs", "..", "ns/..", "runs name"])
def test_spec_refuses_bucket_urls_and_prefixes(bucket_id):
    with pytest.raises(ValidationError, match="bucket_id must be"):
        SwarmSpec.model_validate({**make_spec().model_dump(), "bucket_id": bucket_id})


@pytest.mark.parametrize("block", [{"hf_router": {"bill_to": "org"}},
                                   {"hf_jobs": {"flavor": "a100-large",
                                                "tool_call_parser": "qwen3_coder",
                                                "namespace": "org"}}])
def test_spec_refuses_old_inference_billing_overrides(block):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SwarmSpec.model_validate({**make_spec().model_dump(), "inference": block})


def test_init_reports_a_missing_bucket_without_creating_it(monkeypatch):
    hub = BillingHub()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda **_: hub)

    def missing(*_a, **_k):
        raise HfHubHTTPError("missing", response=httpx.Response(
            404, request=httpx.Request("POST", "https://huggingface.co/api/buckets/launcher/runs")))

    monkeypatch.setattr(hub, "batch_bucket_files", missing)
    spec = SwarmSpec.model_validate({**make_spec().model_dump(), "namespace": "",
                                     "bucket_id": "runs"})
    check = rundir.check_bucket(spec, {"HF_TOKEN": "infrastructure-key"}, rundir.InitReport())
    assert check.ok and "launcher/runs: missing; will be created private" in check.detail
    assert spec.bucket_id == "runs" and spec.namespace == "" and not hub.buckets


# Keep the real Hub implementation available despite the suite's default create-bucket fake.
CREATE_BUCKET = HfApi.create_bucket


def test_existing_bucket_creation_uses_hub_exist_ok_without_visibility_update(
        api, monkeypatch, respx_mock, tmp_path):
    monkeypatch.setattr(HfApi, "create_bucket", CREATE_BUCKET)
    create = respx_mock.post("https://huggingface.co/api/buckets/ns/sh").mock(
        return_value=httpx.Response(409, json={"url": "https://huggingface.co/buckets/ns/sh"}))
    path = write_spec(tmp_path / "spec.yaml", make_spec())
    launch(api, read_spec(path), path, preflight=False)
    assert create.call_count == 1
    request = create.calls[0].request
    assert json.loads(request.content) == {"private": True}
    assert request.headers["authorization"] == "Bearer fake-hub-token"
    assert len(respx_mock.calls) == 1  # no PATCH to change an existing bucket's visibility
