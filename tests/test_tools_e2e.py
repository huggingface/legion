"""Tools folder end to end in one process: `flock launch` (the real CLI) packs and uploads a
`tools/` folder next to the spec, the real controller app and reconcile pass place a 2-agent
`llm` swarm on two real Worker loops (FakeJobsClient, ASGI), each worker fetches the bundle by
hash, and the real LLMAdapter talks to tests/fake_openai.py, whose model calls `add(a, b)`.
The result comes back to the model and the agent's events carry the tools version. Then the
folder changes (a new `mul` tool), `flock update` runs on the running swarm, and the next
turns use the new version without any worker restart."""

from __future__ import annotations

import asyncio
import json
import random
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from flock import cli
from flock.controller.api import create_app
from flock.controller.jobs import FakeJobsClient
from flock.controller.reconcile import Settings, reconcile
from flock.controller.store import Store
from flock.tools.bundle import pack
from flock.worker.adapters.llm import LLMAdapter
from flock.worker.events import parse_iso
from flock.worker.main import ControllerClient, Worker, WorkerConfig

from . import fake_openai
from .worker_fakes import until

TOKEN = "tools-e2e-token"
KEY_ENV = "FAKE_LLM_KEY"
AGENTS = ["calc-0000", "calc-0001"]

TOOLS_V1 = '''
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b
'''
TOOLS_V2 = TOOLS_V1 + '''

def mul(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
'''


def spec_raw(url: str, tools) -> dict:
    return {
        "swarm": "tl", "namespace": "ns", "bucket": "hf://buckets/ns/b",
        "providers": {"gpu": {"type": "openai_compat", "base_url": url, "api_key_env": KEY_ENV,
                              "self_hosted": True}},
        "types": {"calc": {"harness": "llm", "model": "gpu:fake-model", "replicas": 2,
                           "agents_per_worker": 1, "idle_grace": "30s", "turn_timeout": "30s",
                           "tools": tools}},
    }


class Swarm:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.store = Store(tmp / "data")
        self.jobs = FakeJobsClient()
        self.settings = Settings(public_url="http://t", token=TOKEN, rng=random.Random(0),
                                 secret_source={})
        self.app = create_app(self.store, TOKEN)
        self.transport = httpx.ASGITransport(app=self.app)
        self.workers: dict[str, Worker] = {}
        self.tasks: list[asyncio.Task] = []

    def tick(self) -> None:
        reconcile(self.store, self.jobs, datetime.now(UTC), self.settings)
        for launch in self.jobs.launches:
            if launch["job_id"] not in self.workers:
                self._start(launch)

    def _start(self, launch: dict) -> None:
        env = launch["env"]
        work = self.tmp / env["WORKER_ID"] / "agents"  # one disk per worker job
        cfg = WorkerConfig(url=env["FLOCK_URL"], token=launch["secrets"]["FLOCK_TOKEN"],
                           worker_id=env["WORKER_ID"], swarm=env["SWARM"], type=env["TYPE"],
                           deadline=parse_iso(env["DEADLINE"]), bucket=None, work_root=work,
                           poll_wait=1, heartbeat_interval=0.1, flush_interval=0.05,
                           poll_min_interval=0.02, incarnation=env["WORKER_INCARNATION"])

        def factory(name, spec):
            return LLMAdapter(work / name, agent=name, home=work.parent / "llm", backoff_s=0.01)

        client = ControllerClient(cfg.url, cfg.token, transport=self.transport,
                                  incarnation=cfg.incarnation)
        w = Worker(cfg, client, adapter_factory=factory)
        self.workers[launch["job_id"]] = w
        self.tasks.append(asyncio.create_task(w.run()))

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport, base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"})

    async def close(self) -> None:
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for w in self.workers.values():
            await w.client.aclose()
        self.store.close()


async def apply(spec_file: Path, cmd: str = "launch") -> str:
    """`flock launch` (or `update`) in a thread (the CLI is synchronous; the workers keep running)."""
    r = await asyncio.to_thread(CliRunner().invoke, cli.app, [cmd, str(spec_file),
                                                              "--skip-preflight"])
    assert r.exit_code == 0, r.output
    return r.output


def flatten(e: dict) -> tuple[str, dict]:
    """(kind, payload) of a stored event: custom events by their customType."""
    d = e.get("data") or {}
    if e["type"] == "message":
        return "message", d.get("message") or {}
    return e["type"], d.get("data", d)


async def turns(admin: httpx.AsyncClient, agent: str, n: int) -> list[tuple[str, dict]]:
    """The agent's events once it has ended `n` turns."""
    evs: list[tuple[str, dict]] = []

    async def done() -> bool:
        nonlocal evs
        r = await admin.get(f"/agents/{agent}/events")
        evs = [flatten(e) for e in r.json()["events"]]
        return sum(k == "turn_ended" for k, _ in evs) >= n

    for _ in range(400):
        if await done():
            return evs
        await asyncio.sleep(0.05)
    raise AssertionError(f"{agent}: fewer than {n} turns ended: {evs}")


def turn_slices(evs: list[tuple[str, dict]]) -> list[list[tuple[str, dict]]]:
    out: list[list[tuple[str, dict]]] = []
    for k, d in evs:
        if k == "turn_started":
            out.append([])
        if out:
            out[-1].append((k, d))
    return out


async def test_tools_folder_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-e2e")
    log = tmp_path / "requests.jsonl"
    srv = fake_openai.serve(0, str(log))
    url = f"http://127.0.0.1:{srv.server_port}/v1"
    sw = Swarm(tmp_path)
    monkeypatch.setattr("flock.client.connect",
                        lambda *_: TestClient(sw.app, headers={"Authorization": f"Bearer {TOKEN}"}))
    exp = tmp_path / "experiment"
    (exp / "tools").mkdir(parents=True)
    (exp / "tools" / "calc.py").write_text(textwrap.dedent(TOOLS_V1))
    spec_file = exp / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(spec_raw(url, ["add"])))
    v1 = pack(exp / "tools").version
    admin = sw.client()
    try:
        out = await apply(spec_file)
        assert f"tools: version {v1}" in out
        sw.tick()
        assert len(sw.workers) == 2
        await until(lambda: all(w.tools.version == v1 for w in sw.workers.values()), timeout=10)

        for i, agent in enumerate(AGENTS):
            r = await admin.post(f"/agents/{agent}/prompt",
                                 json={"text": f'call: add {{"a": {i + 2}, "b": 40}}'})
            assert r.status_code == 200, r.text
        for i, agent in enumerate(AGENTS):
            [t] = turn_slices(await turns(admin, agent, 1))
            started, ended = t[0][1], next(d for k, d in t if k == "turn_ended")
            assert started["tools_version"] == ended["tools_version"] == v1
            assert ended["status"] == "completed"
            msgs = [d for k, d in t if k == "message"]
            call = next(m for m in msgs if m.get("stopReason") == "toolUse")
            assert call["content"][0] == {"type": "toolCall", "id": "call_0", "name": "add",
                                          "arguments": {"a": i + 2, "b": 40}}
            res = next(m for m in msgs if m["role"] == "toolResult")
            assert res["toolName"] == "add" and res["content"][0]["text"] == str(i + 42)
            assert res["toolsVersion"] == v1 and res["isError"] is False
            final = [m for m in msgs if m["role"] == "assistant"][-1]
            assert final["content"] == [{"type": "text", "text": f"done: {i + 42}"}]
        reqs = [json.loads(x)["body"] for x in log.read_text().splitlines()]
        offered = {t["function"]["name"]: t["function"] for t in reqs[0]["tools"]}
        assert set(offered) == {"bash", "add"}
        assert offered["add"]["description"] == "Add two integers and return the sum."
        assert offered["add"]["parameters"] == {
            "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"], "additionalProperties": False}
        # the model got the result as the tool message
        second = next(r for r in reqs if r["messages"][-1]["role"] == "tool")
        assert second["messages"][-1]["content"] in ("42", "43")
        # each worker fetched the bundle itself, outside the agents' workdirs
        for w in sw.workers.values():
            assert (w.cfg.work_root / "_tools" / v1 / "calc.py").is_file()

        # change the folder while the swarm runs; re-apply; the next turns use the new version
        (exp / "tools" / "calc.py").write_text(textwrap.dedent(TOOLS_V2))
        spec_file.write_text(yaml.safe_dump(spec_raw(url, "all")))
        v2 = pack(exp / "tools").version
        assert v2 != v1
        out = await apply(spec_file, "update")
        assert f"tools: {v1} -> {v2} (+mul)" in out, out
        await until(lambda: all(w.tools.version == v2 for w in sw.workers.values()), timeout=10)
        for agent in AGENTS:
            r = await admin.post(f"/agents/{agent}/prompt", json={"text": 'call: mul {"a": 6, "b": 7}'})
            assert r.status_code == 200
        for agent in AGENTS:
            _, t2 = turn_slices(await turns(admin, agent, 2))
            assert t2[0][1]["tools_version"] == v2
            assert next(d for k, d in t2 if k == "turn_ended")["tools_version"] == v2
            res = next(d for k, d in t2 if k == "message" and d["role"] == "toolResult")
            assert (res["toolName"], res["content"][0]["text"], res["toolsVersion"]) == ("mul", "42", v2)
        last = json.loads(log.read_text().splitlines()[-1])["body"]
        assert {t["function"]["name"] for t in last["tools"]} == {"bash", "add", "mul"}
        assert len(sw.workers) == 2  # no worker was replaced
    finally:
        await admin.aclose()
        await sw.close()
        srv.shutdown()
