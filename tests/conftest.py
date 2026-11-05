from __future__ import annotations

import os
import random
import stat
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from flock import ops
from flock.controller.api import create_app
from flock.controller.jobs import FakeJobsClient
from flock.controller.reconcile import Settings, reconcile
from flock.controller.store import Store
from flock.models import SwarmSpec

from . import fake_openai
from .tinyrun import TinyRun

TOKEN = "test-token"
T0 = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)


class Clock:
    def __init__(self, t: datetime = T0):
        self.t = t

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> datetime:
        self.t += timedelta(seconds=seconds)
        return self.t


def make_spec(swarm: str = "demo", **types: dict) -> SwarmSpec:
    if not types:
        types = {"researcher": {"replicas": 3, "agents_per_worker": 2}}
    full = {}
    for name, overrides in types.items():
        base = {
            "harness": "pi",
            "model": "claude-opus-5-5",
            "secrets": ["ANTHROPIC_API_KEY"],
            "env": {"MSGBOARD_URL": "https://example.invalid"},
        }
        full[name] = {**base, **overrides}
    return SwarmSpec.model_validate(
        {"swarm": swarm, "namespace": "ns", "bucket": "hf://buckets/ns/sh", "types": full}
    )


def wire(monkeypatch, app) -> None:
    """Every `flock.client` call goes to the in-process controller `app`."""
    monkeypatch.setattr("flock.client.connect",
                        lambda *_: TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}))


@pytest.fixture
def api(monkeypatch, app):
    """A `flock.client.Client` wired to the in-process controller."""
    from flock.client import Client

    wire(monkeypatch, app)
    return Client()


def write_spec(path, spec: SwarmSpec) -> str:
    path.write_text(yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True)))
    return str(path)


STUBS = Path(__file__).parent / "fixtures" / "slurm_stubs"


class Slurm:
    """The stub cluster: job files under `dir/jobs/<id>.*`."""

    def __init__(self, root: Path) -> None:
        self.dir = root

    def ids(self) -> list[str]:
        return sorted(p.stem for p in (self.dir / "jobs").glob("*.state"))

    def argv(self, job_id: str) -> list[str]:
        return (self.dir / "jobs" / f"{job_id}.argv").read_text().splitlines()

    def set(self, job_id: str, state: str, exit_code: str = "0:0") -> None:
        (self.dir / "jobs" / f"{job_id}.state").write_text(state + "\n")
        (self.dir / "jobs" / f"{job_id}.exit").write_text(exit_code + "\n")

    def flag(self, job_id: str, name: str) -> None:
        (self.dir / "jobs" / f"{job_id}.{name}").touch()

    def calls(self) -> list[str]:
        p = self.dir / "calls"
        return p.read_text().splitlines() if p.exists() else []


@pytest.fixture
def slurm(tmp_path, monkeypatch) -> Slurm:
    """The stub cluster (tests/fixtures/slurm_stubs) first on PATH, its jobs in tmp_path/slurm."""
    d = tmp_path / "slurm"
    d.mkdir()
    for f in STUBS.iterdir():  # executable even if the checkout lost the mode bits
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{STUBS}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SLURM_STUB_DIR", str(d))
    return Slurm(d)


@pytest.fixture(autouse=True)
def no_local_checkpoint(monkeypatch):
    """A local run checkpoints to <bucket>/controller/<swarm> by default: never from tests."""
    monkeypatch.setenv("FLOCK_STATE_URL", "")


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """A tiny real run (tests/tinyrun.py) working in tmp_path, its model server started."""
    srv = fake_openai.serve(0, str(tmp_path / "requests.jsonl"))
    monkeypatch.chdir(tmp_path)
    for k in [k for k in os.environ if k.startswith("FLOCK_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("FLOCK_STATE_URL", "")  # see no_local_checkpoint
    monkeypatch.setenv("FLOCK_HEALTH_INTERVAL", "1")  # a verdict within a second
    monkeypatch.setenv("FLUSH_INTERVAL", "0.2")  # workers send their events within 0.2 s
    monkeypatch.setenv("FLOCK_RECONCILE_INTERVAL", "0.2")  # the local run's cadence
    monkeypatch.setenv("FLOCK_LOCAL_REFRESH", "0.2")
    monkeypatch.setenv("FLOCK_LOOP_INTERVAL", "0.2")
    monkeypatch.setenv("FAKE_KEY", "fake-key")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    waited = []

    def sleep(_s: float) -> None:  # ops' polls (`stop --release` has no limit): 0.2 s, 60 s at most
        waited.append(0.2)
        assert sum(waited) <= 60, "flock waited over 60 s"
        time.sleep(0.2)

    monkeypatch.setattr(ops, "_sleep", sleep)
    run = TinyRun(tmp_path, f"http://127.0.0.1:{srv.server_port}/v1", monkeypatch)
    yield run
    run.close()
    srv.shutdown()


@pytest.fixture
def store(tmp_path) -> Store:
    s = Store(tmp_path / "data")
    yield s
    s.close()


@pytest.fixture
def jobs() -> FakeJobsClient:
    return FakeJobsClient()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        public_url="https://controller.invalid",
        token=TOKEN,
        secret_source={"ANTHROPIC_API_KEY": "sk-test"},
        rng=random.Random(0),
    )


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def tick(store, jobs, settings, clock):
    """Run one reconcile pass at the clock's time; returns the actions."""
    return lambda: reconcile(store, jobs, clock(), settings)


@pytest.fixture
def app(store, clock):
    return create_app(store, TOKEN, clock=clock)


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        yield c
