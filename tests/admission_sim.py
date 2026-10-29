"""A run-1k-shaped admission simulation against the real controller app (fake jobs, fake
workers, frozen clock). Not a test module: tests/test_admission.py drives it, and it also runs
standalone against any controller version (`python -m tests.admission_sim`) for before/after
numbers.

Shape (pixel run 1k, 2026-10-19): 3 self-hosted providers with `max_working`, one agent type
per provider (334 + 333 + 333 agents), 25 agents per worker, so 42 workers. One round is 3 s:
the driver prompts every agent the controller lists idle that has no pending prompt, every
worker polls its commands once (in worker-id order, as the run's early pollers did), turns
last 5-15 rounds (15-45 s), events are flushed every round and heartbeats go out every 10
rounds (30 s) with a per-worker phase.
"""

import asyncio
import random
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx

from flock.controller.api import create_app
from flock.controller.jobs import FakeJobsClient
from flock.controller.reconcile import AgentState, Settings, agent_state, reconcile
from flock.controller.store import Store
from flock.models import AgentHeartbeat, Event, PromptCommand, SwarmSpec, WorkerHeartbeat

TOKEN = "sim-token"
ROUND = 3.0  # s
HEARTBEAT_ROUNDS = 10  # 30 s
TYPES = {"painter-a": ("gpu1", 334), "painter-b": ("gpu2", 333), "painter-c": ("gpu3", 333)}


def sim_spec(max_working: int, per_worker: int = 25) -> SwarmSpec:
    providers = {
        prov: {"type": "openai_compat", "base_url": f"http://{prov}:8000/v1",
               "api_key_env": "VLLM_API_KEY", "self_hosted": True, "max_working": max_working}
        for prov, _ in TYPES.values()
    }
    types = {
        name: {"harness": "codex", "model": f"{prov}:LFM2.5-2.6B", "replicas": n,
               "agents_per_worker": per_worker}
        for name, (prov, n) in TYPES.items()
    }
    return SwarmSpec.model_validate({"swarm": "px", "namespace": "ns", "bucket": "hf://buckets/ns/b",
                                     "providers": providers, "types": types})


@dataclass
class SimResult:
    rounds: int
    agents: int
    ever: int  # agents that had at least one turn
    all_by: int | None  # first round at which every agent had had a turn
    turns: int
    peak_in_turn: dict[str, int]  # per provider, true agents in a turn (worker view)
    mean_in_turn: dict[str, float]
    coverage: list[int] = field(default_factory=list)  # agents with >= 1 turn, per round


class _Clock:
    def __init__(self) -> None:
        self.t = datetime(2026, 10, 19, 17, 42, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.t


async def simulate(rounds: int = 60, max_working: int = 128, seed: int = 0,
                   data_dir: str | None = None) -> SimResult:
    rng = random.Random(seed)
    tmp = tempfile.TemporaryDirectory() if data_dir is None else None
    store = Store(data_dir or tmp.name)
    clock = _Clock()
    jobs = FakeJobsClient()
    settings = Settings(public_url="http://sim", token=TOKEN, secret_source={"VLLM_API_KEY": "k"},
                        rng=random.Random(seed))
    app = create_app(store, TOKEN, clock=clock, enforce_max_working=True)
    spec = sim_spec(max_working)
    prov_of = {name: prov for name, (prov, _) in TYPES.items()}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://sim",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        r = await c.put("/swarms/px", json=spec.model_dump(mode="json"))
        r.raise_for_status()
        reconcile(store, jobs, clock(), settings)
        workers = sorted(w["id"] for w in store.list_workers("px"))
        hosted: dict[str, list[str]] = {w: [] for w in workers}
        for a in store.list_agents("px"):
            hosted[a["worker"]].append(a["name"])
        left: dict[str, int] = {}  # agent -> rounds left in its turn (0: idle)
        turn_no: dict[str, int] = {}
        outbox: dict[str, list[Event]] = {w: [] for w in workers}

        def heartbeat(wid: str) -> dict:
            agents = [AgentHeartbeat(name=n, state="working" if left.get(n) else "idle",
                                     turn_id=f"t-{turn_no.get(n, 0)}" if left.get(n) else None,
                                     turns=turn_no.get(n, 0)) for n in hosted[wid]]
            return WorkerHeartbeat(worker=wid, deadline=clock() + timedelta(hours=2),
                                   agents=agents).model_dump(mode="json")

        def event(kind: str, agent: str) -> Event:
            return Event(type=kind, ts=clock(), agent=agent, turn_id=f"t-{turn_no.get(agent, 0)}")

        for wid in workers:  # take the assigns, report every agent idle
            (await c.get(f"/workers/{wid}/commands", params={"wait": 0})).raise_for_status()
            (await c.post(f"/workers/{wid}/heartbeat", json=heartbeat(wid))).raise_for_status()

        in_turn_sum = dict.fromkeys(prov_of.values(), 0)
        peak = dict.fromkeys(prov_of.values(), 0)
        coverage: list[int] = []
        all_by = None
        for rnd in range(rounds):
            clock.t += timedelta(seconds=ROUND)
            now = clock().timestamp()
            # driver: prompt agents the controller lists idle and that have nothing pending
            wrows = {w["id"]: w for w in store.list_workers("px")}
            pending = store.pending_counts("px")
            for a in store.list_agents("px"):
                if pending.get(a["name"]) or agent_state(a, wrows.get(a["worker"]), now) \
                        != AgentState.idle:
                    continue
                cmd = PromptCommand(agent=a["name"], prompt_id=f"{a['name']}-{rnd}", text="paint")
                store.add_prompt(a["name"], cmd.model_dump_json(), now)
            # workers: poll once each; a prompt starts a turn unless the agent is in one
            for wid in workers:
                r = await c.get(f"/workers/{wid}/commands", params={"wait": 0})
                r.raise_for_status()
                for cmd in r.json()["commands"]:
                    n = cmd.get("agent")
                    if cmd["kind"] != "prompt" or left.get(n):
                        continue
                    turn_no[n] = turn_no.get(n, 0) + 1
                    left[n] = rng.randint(5, 15)
                    outbox[wid].append(event("turn_started", n))
            # turns run; the round's events and phased heartbeats go out
            for i, wid in enumerate(workers):
                for n in hosted[wid]:
                    if left.get(n):
                        left[n] -= 1
                        if not left[n]:
                            outbox[wid].append(event("turn_ended", n))
                if outbox[wid]:
                    body = {"worker": wid,
                            "events": [e.model_dump(mode="json") for e in outbox[wid]]}
                    (await c.post(f"/workers/{wid}/events", json=body)).raise_for_status()
                    outbox[wid] = []
                if (rnd + i) % HEARTBEAT_ROUNDS == 0:
                    (await c.post(f"/workers/{wid}/heartbeat", json=heartbeat(wid))).raise_for_status()
            in_turn = dict.fromkeys(prov_of.values(), 0)
            for n, k in left.items():
                if k:
                    in_turn[prov_of[n.rsplit("-", 1)[0]]] += 1
            for p, k in in_turn.items():
                in_turn_sum[p] += k
                peak[p] = max(peak[p], k)
            coverage.append(len(turn_no))
            if all_by is None and len(turn_no) == sum(len(v) for v in hosted.values()):
                all_by = rnd + 1
    store.close()
    if tmp is not None:
        tmp.cleanup()
    return SimResult(rounds=rounds, agents=sum(len(v) for v in hosted.values()),
                     ever=len(turn_no), all_by=all_by, turns=sum(turn_no.values()),
                     peak_in_turn=peak,
                     mean_in_turn={p: round(s / rounds, 1) for p, s in in_turn_sum.items()},
                     coverage=coverage)


if __name__ == "__main__":
    import sys

    res = asyncio.run(simulate(rounds=int(sys.argv[1]) if len(sys.argv) > 1 else 60))
    print(res.__dict__ | {"coverage": res.coverage[::5]})
