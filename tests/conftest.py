from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from flock.controller.api import create_app
from flock.controller.jobs import FakeJobsClient
from flock.controller.reconcile import Settings, reconcile
from flock.controller.store import Store
from flock.models import SwarmSpec

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


@pytest.fixture(autouse=True)
def no_local_checkpoint(monkeypatch):
    """A local run checkpoints to <bucket>/controller/<swarm> by default: never from tests."""
    monkeypatch.setenv("FLOCK_STATE_URL", "")


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
