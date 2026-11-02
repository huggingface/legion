"""Live changes end to end in one process (docs/live.md): the real CLI (`flock apply`,
`flock update`, `flock grant`, `flock revoke`), the real controller app and reconcile pass, two
real Worker loops (FakeJobsClient, ASGI) running the `llm` harness against
tests/fake_openai.py. The experiment folder changes while the swarm runs (system prompt, tools
folder, replicas); a grant takes `add` and `bash` away from one agent. Each change reaches
the agents at their next turn, with a one-line note in that prompt, without a worker
restart; grants survive `flock update`."""

from __future__ import annotations

import asyncio
import json
import textwrap

import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from flock import cli
from flock.models import type_spec_rev
from flock.tools.bundle import pack

from . import fake_openai
from .test_tools_e2e import KEY_ENV, TOKEN, TOOLS_V1, TOOLS_V2, Swarm, turn_slices, turns
from .worker_fakes import until

AGENTS = ["calc-0000", "calc-0001"]


def spec_raw(url: str, tools, replicas: int = 2) -> dict:
    return {
        "swarm": "tl", "namespace": "ns", "bucket": "hf://buckets/ns/b",
        "providers": {"gpu": {"type": "openai_compat", "base_url": url, "api_key_env": KEY_ENV,
                              "self_hosted": True}},
        "types": {"calc": {"harness": "llm", "model": "gpu:fake-model", "replicas": replicas,
                           "agents_per_worker": 1, "idle_grace": "30s", "turn_timeout": "30s",
                           "tools": tools, "system_prompt_file": "prompts/system.md"}},
    }


async def cli_run(*args: str) -> str:
    r = await asyncio.to_thread(CliRunner().invoke, cli.app, list(args))
    assert r.exit_code == 0, r.output
    return r.output


def requests_of(log, agent: str) -> list[dict]:
    """The model requests made for `agent` (its workdir is in the system message)."""
    out = []
    for line in log.read_text().splitlines():
        body = json.loads(line)["body"]
        if f"/{agent}" in body["messages"][0]["content"]:
            out.append(body)
    return out


def user_texts(body: dict) -> list[str]:
    return [m["content"] for m in body["messages"] if m["role"] == "user"]


async def test_update_and_grants_reach_running_agents(tmp_path, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-e2e")
    log = tmp_path / "requests.jsonl"
    srv = fake_openai.serve(0, str(log))
    url = f"http://127.0.0.1:{srv.server_port}/v1"
    sw = Swarm(tmp_path)
    monkeypatch.setattr(cli, "client",
                        lambda: TestClient(sw.app, headers={"Authorization": f"Bearer {TOKEN}"}))
    exp = tmp_path / "experiment"
    (exp / "tools").mkdir(parents=True)
    (exp / "prompts").mkdir()
    (exp / "tools" / "calc.py").write_text(textwrap.dedent(TOOLS_V1))
    (exp / "prompts" / "system.md").write_text("You are calc v1.")
    spec_file = exp / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(spec_raw(url, ["add"])))
    v1 = pack(exp / "tools").version
    admin = sw.client()

    async def prompt(agent: str, text: str, n: int) -> list:
        r = await admin.post(f"/agents/{agent}/prompt", json={"text": text})
        assert r.status_code == 200, r.text
        return turn_slices(await turns(admin, agent, n))[-1]

    try:
        await cli_run("apply", str(spec_file), "--skip-preflight")
        sw.tick()
        await until(lambda: len(sw.workers) == 2
                    and all(w.tools.version == v1 for w in sw.workers.values()), timeout=10)
        for agent in AGENTS:
            await prompt(agent, 'call: add {"a": 1, "b": 2}', 1)
        assert all("You are calc v1." in requests_of(log, a)[-1]["messages"][0]["content"]
                   for a in AGENTS)

        # 1. the folder changes: prompt, tools (+mul), selection; `flock update` applies it
        (exp / "prompts" / "system.md").write_text("You are calc v2.")
        (exp / "tools" / "calc.py").write_text(textwrap.dedent(TOOLS_V2))
        spec_file.write_text(yaml.safe_dump(spec_raw(url, "all")))
        v2 = pack(exp / "tools").version
        out = await cli_run("update", str(exp))
        assert out.splitlines() == [f"tools: {v1} -> {v2} (+mul)",
                                    "prompts changed: calc (system_prompt)",
                                    'types.calc.tools: ["add"] -> all'], out
        rev = type_spec_rev(sw.store.get_swarm("tl")["spec"].types["calc"])
        await until(lambda: all(w.tools.version == v2 and w.spec_rev == rev
                                for w in sw.workers.values()), timeout=10)
        t = await prompt("calc-0000", 'call: mul {"a": 6, "b": 7}', 2)
        assert t[0][1]["tools_version"] == v2
        res = next(d for k, d in t if k == "message" and d["role"] == "toolResult")
        assert (res["toolName"], res["content"][0]["text"]) == ("mul", "42")
        req = requests_of(log, "calc-0000")[-2]  # the call, before its tool result
        assert "You are calc v2." in req["messages"][0]["content"]
        assert "You are calc v1." not in req["messages"][0]["content"]
        assert user_texts(req)[-1] == ("[flock] Since your last turn: your system prompt "
                                       'changed.\n\ncall: mul {"a": 6, "b": 7}')
        assert {x["function"]["name"] for x in req["tools"]} == {"bash", "add", "mul"}

        # 2. a grant for one agent: no add, no bash, at most 2 model calls per prompt
        out = await cli_run("grant", "calc-0001", "tools=-add", "bash=off", "limits.max_turns=2")
        assert "agent calc-0001: tools -add, bash off, limits.max_turns 2" in out
        w1 = next(w for w in sw.workers.values() if "calc-0001" in w.agents)
        await until(lambda: w1.agents["calc-0001"]._next_grant is not None
                    and w1.agents["calc-0001"]._next_grant.bash is False, timeout=10)
        t = await prompt("calc-0001", 'call: mul {"a": 2, "b": 3}', 2)
        res = next(d for k, d in t if k == "message" and d["role"] == "toolResult")
        assert (res["toolName"], res["content"][0]["text"]) == ("mul", "6")
        req = requests_of(log, "calc-0001")[-2]
        assert {x["function"]["name"] for x in req["tools"]} == {"mul"}
        assert req["messages"][0]["content"].startswith("You are calc-0001")
        assert "You have no shell." in req["messages"][0]["content"]
        assert user_texts(req)[-1] == (
            "[flock] Since your last turn: your system prompt changed; your grants changed: "
            'tools -add, bash off, limits.max_turns 2.\n\ncall: mul {"a": 2, "b": 3}')
        t = await prompt("calc-0001", "bash: echo hi", 3)  # bash is gone for this agent
        res = next(d for k, d in t if k == "message" and d["role"] == "toolResult")
        assert res["isError"] and "unknown tool 'bash'; the tools are mul" in res["content"][0]["text"]
        st = (await admin.get("/agents/calc-0001")).json()
        assert st["grants"] == {"tools": {"add": False}, "limits": {"max_turns": 2}, "bash": False}

        # 3. `flock update` keeps grants: a replica change applies, the grant stays
        spec_file.write_text(yaml.safe_dump(spec_raw(url, "all", replicas=3)))
        out = await cli_run("update", str(spec_file), "--replicas", "spec")
        assert out.strip() == "types.calc.replicas: 2 -> 3"
        assert (await admin.get("/agents/calc-0001")).json()["grants"] == st["grants"]
        sw.tick()
        assert (await admin.get("/agents/calc-0002")).status_code == 200
        # the other agent never saw the grant
        await prompt("calc-0000", "bash: echo still-here", 3)
        req = requests_of(log, "calc-0000")[-1]
        assert {x["function"]["name"] for x in req["tools"]} == {"bash", "add", "mul"}

        # 4. revoke: bash and add are back at the next turn, with a note
        await cli_run("revoke", "calc-0001")
        await until(lambda: w1.agents["calc-0001"]._next_grant is None, timeout=10)
        await prompt("calc-0001", "bash: echo back", 4)
        req = requests_of(log, "calc-0001")[-2]
        assert {x["function"]["name"] for x in req["tools"]} == {"bash", "add", "mul"}
        assert user_texts(req)[-1].startswith("[flock] Since your last turn: your grants "
                                              "changed: all revoked.")
        assert len(sw.workers) == 3  # the new replica's worker; none of the first two restarted
    finally:
        await admin.aclose()
        await sw.close()
        srv.shutdown()
