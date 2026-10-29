"""Spawn end to end in one process: the real controller app and reconcile pass, FakeJobsClient,
real Worker loops talking to the app over ASGI, and the scripted harness. Every agent-side call
uses the FLOCK_AGENT_TOKEN the worker put into that agent's harness env, as `flock` would.

1 leader spawns 5 analysts (3 per worker: 2 workers); an analyst spawns a helper whose depth is
0, so the helper's own spawn is refused with `depth`; a repeated key returns 200; a child
prompts its parent; removing the analyst takes the helper with it on the workers too; scaling
the leader to 0 takes the remaining analysts."""

import asyncio
import random
from datetime import UTC, datetime

import httpx

from flock.controller.api import create_app
from flock.controller.jobs import FakeJobsClient
from flock.controller.reconcile import Settings, reconcile
from flock.controller.store import Store
from flock.models import SwarmSpec
from flock.worker.events import parse_iso
from flock.worker.main import ControllerClient, Worker, WorkerConfig

from .worker_fakes import ScriptedAdapter, until

TOKEN = "e2e-token"
LEADER = "leader-0000"


def e2e_spec(leader_replicas: int = 1) -> SwarmSpec:
    base = {"harness": "claude", "model": "m-1", "idle_grace": "30s", "turn_timeout": "30s"}
    types = {
        "leader": {"replicas": leader_replicas, "workers": {"initial": 1}},
        "analyst": {"agents_per_worker": 3, "workers": {"initial": 1, "max_extra": 1}},
        "helper": {"agents_per_worker": 2, "workers": {"max_extra": 1}},
        "scout": {"agents_per_worker": 2, "workers": {"max_extra": 1}},
    }
    return SwarmSpec.model_validate({
        "swarm": "e2e", "namespace": "ns", "bucket": "hf://buckets/ns/b",
        "types": {k: {**base, **v} for k, v in types.items()},
        "spawn": {
            "leader": {"types": {"analyst": 5}, "depth": 2},
            "analyst": {"types": ["helper"], "max": 2, "depth": 2},
            "helper": {"types": ["scout"], "max": 1, "depth": 1},
        },
        "limits": {"alive": 50},
    })


class Harness:
    """Controller + one Worker task per launched job."""

    def __init__(self, tmp_path) -> None:
        self.tmp = tmp_path
        self.store = Store(tmp_path / "data")
        self.jobs = FakeJobsClient()
        self.settings = Settings(public_url="http://t", token=TOKEN, rng=random.Random(0),
                                 secret_source={})
        self.app = create_app(self.store, TOKEN)
        self.transport = httpx.ASGITransport(app=self.app)
        self.workers: dict[str, Worker] = {}
        self.tasks: list[asyncio.Task] = []
        self.adapters: dict[str, ScriptedAdapter] = {}

    def client(self, token: str = TOKEN) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport, base_url="http://t",
                                 headers={"Authorization": f"Bearer {token}"})

    def tick(self) -> None:
        reconcile(self.store, self.jobs, datetime.now(UTC), self.settings)
        for launch in self.jobs.launches:
            if launch["job_id"] not in self.workers:
                self._start(launch)

    def _start(self, launch: dict) -> None:
        env = launch["env"]
        cfg = WorkerConfig(url=env["FLOCK_URL"], token=launch["secrets"]["FLOCK_TOKEN"],
                           worker_id=env["WORKER_ID"], swarm=env["SWARM"], type=env["TYPE"],
                           deadline=parse_iso(env["DEADLINE"]), bucket=None,
                           work_root=self.tmp / "agents", poll_wait=1, heartbeat_interval=0.1,
                           flush_interval=0.05, poll_min_interval=0.02,
                           incarnation=env["WORKER_INCARNATION"])

        def factory(name, spec):
            self.adapters[name] = ScriptedAdapter(self.tmp / "home" / name, turn_time=0.02)
            return self.adapters[name]

        client = ControllerClient(cfg.url, cfg.token, transport=self.transport,
                                  incarnation=cfg.incarnation)
        w = Worker(cfg, client, adapter_factory=factory)
        self.workers[launch["job_id"]] = w
        self.tasks.append(asyncio.create_task(w.run()))

    def token_of(self, agent: str) -> str:
        """What the agent's harness saw as FLOCK_AGENT_TOKEN on its last start."""
        return self.adapters[agent].envs[-1]["FLOCK_AGENT_TOKEN"]

    def hosted(self, agent: str) -> str | None:
        return next((w.cfg.worker_id for w in self.workers.values() if agent in w.agents), None)

    async def started(self, agent: str) -> None:
        await until(lambda: agent in self.adapters and self.adapters[agent].envs, timeout=10)

    async def close(self) -> None:
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for w in self.workers.values():
            await w.client.aclose()
        self.store.close()


async def test_spawn_end_to_end(tmp_path):
    h = Harness(tmp_path)
    admin = h.client()
    try:
        r = await admin.put("/swarms/e2e", json=e2e_spec().model_dump(mode="json"))
        assert r.status_code == 200, r.text
        h.tick()
        await admin.post(f"/agents/{LEADER}/prompt", json={"text": "plan the work"})
        await h.started(LEADER)
        env = h.adapters[LEADER].envs[0]
        assert env["FLOCK_AGENT"] == LEADER and env["FLOCK_PARENT"] == ""
        assert env["FLOCK_TOKEN"] == "" and env["FLOCK_URL"] == "http://t"
        leader = h.client(h.token_of(LEADER))

        # 1. the leader spawns 5 analysts, each with a first prompt; the first one with a key
        names = []
        for i in range(5):
            body = {"type": "analyst", "prompt": f"part {i}", **({"key": "p0"} if i == 0 else {})}
            r = await leader.post(f"/agents/{LEADER}/spawn", json=body)
            assert r.status_code == 201, r.text
            names.append(r.json()["name"])
        assert names[0] == "leader-0000-analyst-0001" and names[4] == "leader-0000-analyst-0005"
        r = await leader.post(f"/agents/{LEADER}/spawn", json={"type": "analyst"})
        assert r.status_code == 409 and r.json()["reason"] == "max_children"
        r = await leader.post(f"/agents/{LEADER}/spawn", json={"type": "analyst", "key": "p0"})
        assert r.status_code == 200 and r.json() == {
            "name": names[0], "created": False, "parent": LEADER, "type": "analyst", "depth": 1}

        h.tick()
        for n in names:
            await h.started(n)
        await until(lambda: all(h.adapters[n].sent for n in names), timeout=10)
        placed = {n: h.hosted(n) for n in names}
        assert set(placed.values()) == {"e2e-analyst-w0000", "e2e-analyst-w0001"}
        assert sorted(placed.values()).count("e2e-analyst-w0000") == 3
        assert h.adapters[names[2]].sent == ["part 2"]
        assert h.adapters[names[2]].envs[0]["FLOCK_PARENT"] == LEADER
        st = (await leader.get(f"/agents/{LEADER}")).json()
        assert st["children"] == names and st["depth"] == 2

        # 2. an analyst spawns a helper (depth 1 - 1 = 0): the helper cannot spawn
        a0 = h.client(h.token_of(names[0]))
        r = await a0.post(f"/agents/{names[0]}/spawn", json={"type": "helper", "prompt": "dig"})
        assert r.status_code == 201 and r.json()["depth"] == 0
        helper = r.json()["name"]
        assert helper == "leader-0000-analyst-0001-helper-0006"
        h.tick()
        await h.started(helper)
        hc = h.client(h.token_of(helper))
        r = await hc.post(f"/agents/{helper}/spawn", json={"type": "scout"})
        assert r.status_code == 409 and r.json()["reason"] == "depth"
        assert (await hc.get("/whoami")).json()["parent"] == names[0]

        # 3. a child reports to its parent; its parent's stream shows the lineage
        r = await a0.post(f"/agents/{LEADER}/prompt", json={"text": "analyst 1: done"})
        assert r.status_code == 200
        await until(lambda: "analyst 1: done" in h.adapters[LEADER].sent, timeout=10)
        started: list = []
        for _ in range(200):
            evs = (await admin.get(f"/agents/{helper}/events")).json()["events"]
            started = [e for e in evs if e["type"] == "session_started"]
            if started:
                break
            await asyncio.sleep(0.02)
        assert started and started[0]["data"]["data"]["parent"] == names[0]

        # 4. removing the analyst cascades to the helper, on the controller and the workers
        r = await leader.delete(f"/agents/{names[0]}")
        assert r.status_code == 200 and r.json() == {"removed": [helper, names[0]]}
        await until(lambda: h.hosted(names[0]) is None and h.hosted(helper) is None, timeout=10)
        assert (await hc.get("/whoami")).status_code == 401  # tokens revoked
        evs = (await leader.get(f"/agents/{LEADER}/events")).json()["events"]
        removed = [e["data"] for e in evs if e["type"] == "child_removed"]
        assert removed == [{"child": names[0], "type": "analyst", "reason": "removed"}]

        # 5. the leader leaving (scale to 0) takes the remaining analysts
        r = await admin.put("/swarms/e2e", json=e2e_spec(0).model_dump(mode="json"))
        assert r.status_code == 200
        h.tick()
        assert h.store.list_agents("e2e") == []
        await until(lambda: all(h.hosted(n) is None for n in names), timeout=10)
    finally:
        await admin.aclose()
        await h.close()
