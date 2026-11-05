from __future__ import annotations

import os
import random
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from flock.controller.api import create_app
from flock.controller.jobs import FakeJobsClient
from flock.controller.reconcile import Settings, reconcile
from flock.controller.store import Store
from flock.models import SwarmSpec

from . import fake_openai
from .tinyrun import TinyRun

# Tests of the experiments under experiments/: those target the pre-cleanup layout (run them from
# commit 8949085) and are not collected here; they go with the experiments in a later cleanup.
collect_ignore = [f"test_{name}.py" for name in (
    "chatbot2_experiment", "chatbot2_selfhosted", "chatbot_experiment", "eliza_experiment",
    "hierarchy_experiment", "interviewer_experiment", "levels_experiment", "run3_experiment",
    "run4_cluster_review", "run4_experiment", "run4_spawner", "teams_experiment",
    "voting_experiment")]

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
            "harness": "claude",
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
    monkeypatch.setenv("FAKE_KEY", "fake-key")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
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
