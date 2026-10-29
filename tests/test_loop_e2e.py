"""`loop:` end to end in one process: the real controller app, its loop engine ticking,
FakeJobsClient, a real Worker over ASGI and three `llm` agents against tests/fake_openai.py.

Spec: `first`, then `every: turn_end` with `while: {max_rounds: 3}`, then `final`. Each agent
gets the first prompt alone, exactly 3 loop prompts (rendered on the worker), then one final
prompt; one turn per prompt (nothing stacked); swarm events in order. Run twice: the worker
re-prompts (no `max_working`), and the controller re-prompts (`max_working: 3`)."""

import asyncio
import json
import random
from datetime import UTC, datetime

import httpx
import pytest

from flock.controller.api import create_app
from flock.controller.jobs import FakeJobsClient
from flock.controller.prices import type_price_table
from flock.controller.reconcile import Settings, reconcile
from flock.controller.store import Store
from flock.models import SwarmSpec
from flock.worker.adapters.llm import LLMAdapter
from flock.worker.events import parse_iso
from flock.worker.main import ControllerClient, Worker, WorkerConfig

from . import fake_openai
from .worker_fakes import until

TOKEN = "loop-e2e"
KEY_ENV = "FLOCK_TEST_LOOP_KEY"
AGENTS = [f"painter-000{i}" for i in range(3)]


def loop_spec(url: str, capped: bool) -> SwarmSpec:
    provider = {"type": "openai_compat", "base_url": url, "api_key_env": KEY_ENV, "self_hosted": True,
                **({"max_working": 3} if capped else {})}
    return SwarmSpec.model_validate({
        "swarm": "e2e", "namespace": "ns", "bucket": "hf://buckets/ns/b", "providers": {"gpu": provider},
        "types": {"painter": {
            "harness": "llm", "model": "gpu:fake-model", "replicas": 3, "agents_per_worker": 3,
            "idle_grace": "30s", "turn_timeout": "30s",
            "loop": {"first": "bash: echo first", "prompt": "bash: echo round {round}",
                     "every": "turn_end", "while": {"max_rounds": 3},
                     "final": {"prompt": "final after {round} rounds; last: {last_summary}"}}}}})


@pytest.fixture
def fake_server(tmp_path):
    srv = fake_openai.serve(0, str(tmp_path / "requests.jsonl"))
    srv.url = f"http://127.0.0.1:{srv.server_port}/v1"
    yield srv
    srv.shutdown()


@pytest.mark.parametrize("capped", [False, True], ids=["worker-side", "controller-side"])
async def test_loop_end_to_end(tmp_path, fake_server, monkeypatch, capped):
    monkeypatch.setenv(KEY_ENV, "sk-fake")
    store = Store(tmp_path / "data")
    jobs = FakeJobsClient()
    settings = Settings(public_url="http://t", token=TOKEN, rng=random.Random(0), secret_source={})
    app = create_app(store, TOKEN, loop_interval=0, enforce_max_working=True)
    transport = httpx.ASGITransport(app=app)
    admin = httpx.AsyncClient(transport=transport, base_url="http://t",
                              headers={"Authorization": f"Bearer {TOKEN}"})
    tasks: list[asyncio.Task] = []
    workers: list[Worker] = []

    async def ticker():
        while True:
            app.state.loops.tick()
            await asyncio.sleep(0.05)

    try:
        r = await admin.put("/swarms/e2e", json=loop_spec(fake_server.url, capped).model_dump(mode="json"))
        assert r.status_code == 200, r.text
        reconcile(store, jobs, datetime.now(UTC), settings)
        [launch] = jobs.launches
        env = launch["env"]
        cfg = WorkerConfig(url=env["FLOCK_URL"], token=launch["secrets"]["FLOCK_TOKEN"],
                           worker_id=env["WORKER_ID"], swarm="e2e", type="painter",
                           deadline=parse_iso(env["DEADLINE"]), bucket=None, work_root=tmp_path / "agents",
                           poll_wait=1, heartbeat_interval=0.1, flush_interval=0.05,
                           poll_min_interval=0.02, incarnation=env["WORKER_INCARNATION"])

        def factory(name, spec):
            return LLMAdapter(cfg.work_root / name, agent=name, home=tmp_path / "llm",
                              prices=type_price_table(spec), backoff_s=0.01)

        w = Worker(cfg, ControllerClient(cfg.url, cfg.token, transport=transport,
                                         incarnation=cfg.incarnation), adapter_factory=factory)
        workers.append(w)
        tasks += [asyncio.create_task(w.run()), asyncio.create_task(ticker())]

        async def loop_done() -> dict:
            return (await admin.get("/swarms/e2e/loop")).json()[0]

        state: dict = {}
        for _ in range(600):  # 30 s at most; about 2 s in practice
            state = await loop_done()
            if state["state"] == "done":
                break
            await asyncio.sleep(0.05)
        assert state["state"] == "done", state
        assert state["mode"] == ("controller" if capped else "worker")
        assert (state["reason"], state["rounds_min"], state["rounds_max"]) == ("max_rounds", 3, 3)
        assert (state["final_sent"], state["final_done"]) == (3, 3)
        await asyncio.sleep(0.5)  # then idle: nothing more is sent

        for name in AGENTS:
            await until(lambda n=name: sum(1 for e in store.read_events(n, 0)
                                           if e["type"] == "turn_ended") == 5, timeout=5)
            evs = (await admin.get(f"/agents/{name}/events")).json()["events"]
            seq = [e["type"] for e in evs if e["type"] in ("loop_prompt", "turn_started", "turn_ended")]
            assert seq == ["loop_prompt", "turn_started", "turn_ended"] * 5, (name, seq)  # never stacked
            prompts = [(e["data"]["data"]["kind"], e["data"]["data"]["round"])
                       for e in evs if e["type"] == "loop_prompt"]
            assert prompts == [("first", 0), ("loop", 1), ("loop", 2), ("loop", 3), ("final", 3)]
            users = [b["text"] for e in evs if e["type"] == "message"
                     and e["data"]["message"]["role"] == "user" for b in e["data"]["message"]["content"]]
            assert users == ["bash: echo first", "bash: echo round 1", "bash: echo round 2",
                             "bash: echo round 3",  # {last_summary}: the last reply (no `done` tool)
                             "final after 3 rounds; last: done: round 3\n[exit code 0]"]
            kinds = [e["type"] for e in evs]
            assert kinds.index("final_round") < max(i for i, k in enumerate(kinds) if k == "loop_prompt")
            assert all(e["data"]["data"]["status"] == "completed" for e in evs if e["type"] == "turn_ended")

        swarm = (await admin.get("/swarms/e2e/events")).json()["events"]
        kinds = [e["type"] for e in swarm if e["type"] != "loop_tick"]
        assert kinds == ["loop_started", "loop_stopped", "final_round", "final_round", "final_round",
                         "final_done"]
        assert next(e for e in swarm if e["type"] == "loop_stopped")["data"]["reason"] == "max_rounds"
        ticks = [e["data"] for e in swarm if e["type"] == "loop_tick"]
        assert sum(t["first"] for t in ticks) == 3
        assert sum(len(t["rounds"]) for t in ticks) == (9 if capped else 0)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for w in workers:
            for rt in w.agents.values():
                await rt.stop()
            await w.client.aclose()
        await admin.aclose()
        store.close()


@pytest.mark.parametrize("capped", [False, True], ids=["worker-side", "controller-side"])
async def test_all_agents_done_ends_the_loop(tmp_path, fake_server, monkeypatch, capped):
    """`stop_when: {all_agents_done: true}`: the `done` tool takes `finished`; each agent calls
    `done(finished: true)` in its first loop round, leaves the loop, and once all three have,
    the loop stops (`reason: done`) long before `max_rounds`, then the final round runs."""
    monkeypatch.setenv(KEY_ENV, "sk-fake")
    store = Store(tmp_path / "data")
    jobs = FakeJobsClient()
    settings = Settings(public_url="http://t", token=TOKEN, rng=random.Random(0), secret_source={})
    app = create_app(store, TOKEN, loop_interval=0, enforce_max_working=True)
    transport = httpx.ASGITransport(app=app)
    admin = httpx.AsyncClient(transport=transport, base_url="http://t",
                              headers={"Authorization": f"Bearer {TOKEN}"})
    tasks: list[asyncio.Task] = []
    workers: list[Worker] = []

    async def ticker():
        while True:
            app.state.loops.tick()
            await asyncio.sleep(0.05)

    raw = loop_spec(fake_server.url, capped).model_dump(mode="json", by_alias=True)
    lp = raw["types"]["painter"]["loop"]
    lp["prompt"] = 'call: done {"summary": "all built", "finished": true}'
    lp["while"]["max_rounds"] = 50
    lp["stop_when"] = {"all_agents_done": True}
    try:
        r = await admin.put("/swarms/e2e", json=raw)
        assert r.status_code == 200, r.text
        reconcile(store, jobs, datetime.now(UTC), settings)
        [launch] = jobs.launches
        env = launch["env"]
        cfg = WorkerConfig(url=env["FLOCK_URL"], token=launch["secrets"]["FLOCK_TOKEN"],
                           worker_id=env["WORKER_ID"], swarm="e2e", type="painter",
                           deadline=parse_iso(env["DEADLINE"]), bucket=None, work_root=tmp_path / "agents",
                           poll_wait=1, heartbeat_interval=0.1, flush_interval=0.05,
                           poll_min_interval=0.02, incarnation=env["WORKER_INCARNATION"])

        def factory(name, spec):
            return LLMAdapter(cfg.work_root / name, agent=name, home=tmp_path / "llm",
                              prices=type_price_table(spec), backoff_s=0.01)

        w = Worker(cfg, ControllerClient(cfg.url, cfg.token, transport=transport,
                                         incarnation=cfg.incarnation), adapter_factory=factory)
        workers.append(w)
        tasks += [asyncio.create_task(w.run()), asyncio.create_task(ticker())]
        state: dict = {}
        for _ in range(600):
            state = (await admin.get("/swarms/e2e/loop")).json()[0]
            if state["state"] == "done":
                break
            await asyncio.sleep(0.05)
        assert state["state"] == "done", state
        assert (state["reason"], state["rounds_max"]) == ("done", 1)
        assert {a["done_reason"] for a in state["per_agent"].values()} == {"done"}
        assert (state["final_sent"], state["final_done"]) == (3, 3)
        await asyncio.sleep(0.3)
        for name in AGENTS:
            evs = store.read_events(name, 0)
            ended = [e["data"]["data"] for e in evs if e["type"] == "turn_ended"]
            assert [d.get("finished") for d in ended] == [None, True, None]  # first, round 1, final
            prompts = [e["data"]["data"]["kind"] for e in evs if e["type"] == "loop_prompt"]
            assert prompts == ["first", "loop", "final"]
        reqs = [json.loads(x)["body"] for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
        done = next(t for t in reqs[0]["tools"] if t["function"]["name"] == "done")
        assert "finished" in done["function"]["parameters"]["properties"]
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for w in workers:
            for rt in w.agents.values():
                await rt.stop()
            await w.client.aclose()
        await admin.aclose()
        store.close()
