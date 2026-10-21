"""Comms end to end in one process: `flock apply` (the real CLI) with a `comms:` block, the real
controller (ASGI) resolves the channel and posts the kickoff as `<swarm>-host` on the fake
msgboard, the reconcile pass places a 2-agent `llm` swarm on one real Worker, and the real
LLMAdapter talks to tests/fake_openai.py, whose model posts, is refused for a fast second post,
reads, gets mentioned (a controller comms event), and searches. Then the protocols and rules
change, `flock apply` runs again, and the running agents get both without a restart."""

from __future__ import annotations

import asyncio
import json

import yaml
from fastapi.testclient import TestClient

from flock import cli
from flock.controller.api import create_app

from . import fake_msgboard, fake_openai
from .test_tools_e2e import Swarm, apply, turn_slices, turns
from .worker_fakes import until

TOKEN = "tools-e2e-token"
KEY_ENV = "FAKE_LLM_KEY"
INVITE_ENV = "E2E_BOARD_INVITE"
A, B = "dev-0000", "dev-0001"


def spec_raw(llm_url: str, board_url: str, gap: str) -> dict:
    return {
        "swarm": "cm", "namespace": "ns", "bucket": "hf://buckets/ns/b",
        "providers": {"gpu": {"type": "openai_compat", "base_url": llm_url, "api_key_env": KEY_ENV,
                              "self_hosted": True}},
        "comms": {"kind": "msgboard", "url": board_url, "invite_env": INVITE_ENV,
                  "channel": "flock-test", "kickoff_file": "prompts/kickoff.md",
                  "protocols": ["protocols/lead-agree.md"],
                  "rules": {"min_post_gap": gap, "max_len": 500, "threads": "per_topic",
                            "no_group_mentions": True}},
        "types": {"dev": {"harness": "llm", "model": "gpu:fake-model", "replicas": 2,
                          "agents_per_worker": 2, "idle_grace": "30s", "turn_timeout": "30s",
                          "comms": True}},
    }


def tool_result(turn: list[tuple[str, dict]]) -> tuple[dict, bool]:
    res = next(d for k, d in turn if k == "message" and d["role"] == "toolResult")
    return json.loads(res["content"][0]["text"]), res["isError"]


async def test_comms_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-e2e")
    board, bsrv, board_url = fake_msgboard.serve()
    monkeypatch.setenv(INVITE_ENV, board.invite)  # the workers' job secret (in-process here)
    log = tmp_path / "requests.jsonl"
    srv = fake_openai.serve(0, str(log))
    llm_url = f"http://127.0.0.1:{srv.server_port}/v1"
    wakes: list[tuple[str, str, dict]] = []
    sw = Swarm(tmp_path)
    sw.settings.secret_source = {INVITE_ENV: board.invite}
    sw.app = create_app(sw.store, TOKEN, secret_source={INVITE_ENV: board.invite},
                        on_comms=lambda agent, reason, data: wakes.append((agent, reason, data)))
    import httpx
    sw.transport = httpx.ASGITransport(app=sw.app)
    monkeypatch.setattr(cli, "client",
                        lambda: TestClient(sw.app, headers={"Authorization": f"Bearer {TOKEN}"}))
    exp = tmp_path / "experiment"
    (exp / "prompts").mkdir(parents=True)
    (exp / "protocols").mkdir()
    (exp / "prompts" / "kickoff.md").write_text("Kickoff: build the bot. One thread per topic.")
    (exp / "protocols" / "lead-agree.md").write_text("PROTO-V1: one agent leads a topic, others reply agree.")
    spec_file = exp / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(spec_raw(llm_url, board_url, "30s")))
    admin = sw.client()

    async def prompt(agent: str, text: str, n: int) -> list[tuple[str, dict]]:
        r = await admin.post(f"/agents/{agent}/prompt", json={"text": text})
        assert r.status_code == 200, r.text
        return turn_slices(await turns(admin, agent, n))[-1]

    try:
        await apply(spec_file)
        stored = sw.store.get_swarm("cm")["spec"].comms
        assert stored.channel_id == 12 and stored.kickoff == "Kickoff: build the bot. One thread per topic."
        [kick] = board.msgs
        assert (kick["id"], kick["sender"]["external_id"]) == (stored.kickoff_id, "cm-host")
        sw.tick()
        [w] = sw.workers.values()
        assert sw.jobs.launches[0]["secrets"][INVITE_ENV] == board.invite
        await until(lambda: len(w.agents) == 2 and w.comms.active, timeout=10)
        assert board.joins == ["cm-host"]  # assigned is not joined

        # A posts a topic; joining happened at this first prompt
        t = await prompt(A, 'call: post {"text": "taking the fallback ladder", "thread": "new"}', 1)
        out, err = tool_result(t)
        assert not err and board.joins == ["cm-host", A]
        root = out["posted"]
        assert board.msgs[-1]["sender"]["external_id"] == A and out["thread"] == root
        # a fast second post is refused by the worker, never sent to the board
        t = await prompt(A, 'call: post {"text": "and another thing", "thread": "new"}', 2)
        out, err = tool_result(t)
        assert err and out["error"].startswith("refused: wait") and len(board.msgs) == 2
        # B reads the channel (kickoff and A's topic), then answers in A's thread with a mention
        t = await prompt(B, 'call: read {}', 1)
        out, err = tool_result(t)
        assert not err and [m["id"] for m in out["messages"]] == [kick["id"], root]
        assert out["messages"][1]["from"] == A
        t = await prompt(B, f'call: post {{"text": "@{A} I can help with the ladder", "thread": {root}}}', 2)
        out, err = tool_result(t)
        assert not err and out["thread"] == root
        await until(lambda: any(a == A and r == "mention" for a, r, _ in wakes), timeout=10)
        [(_, _, wake)] = [x for x in wakes if x[0] == A]
        assert wake == {"agent": A, "msg_id": out["posted"], "thread": root, "from": B}
        # A searches: the last word matches nothing, so it is dropped
        t = await prompt(A, 'call: search {"query": "help ladder zebra"}', 3)
        out, err = tool_result(t)
        assert not err and out["dropped"] == ["zebra"]
        assert [(h["author"], h["thread"]) for h in out["results"]] == [(B, root)]

        reqs = [json.loads(x)["body"] for x in log.read_text().splitlines()]
        system = reqs[0]["messages"][0]["content"]
        assert "PROTO-V1: one agent leads" in system and "#flock-test" in system
        assert {t["function"]["name"] for t in reqs[0]["tools"]} == {"bash", "post", "read", "search"}
        # one long-poll for the worker
        pollers = {who for _, p, q, who in board.requests if p == "/v1/activity" and "wait" in q}
        assert len(pollers) == 1
        # the tokens are in the worker's memory only
        tokens = [w.comms.members[n].ident.token for n in (A, B)]
        files = [p for p in w.cfg.work_root.rglob("*") if p.is_file()]
        assert not any(tok in p.read_text(errors="replace") for p in files for tok in tokens)
        assert not any(tok in json.dumps(rt.adapter.env) for rt in w.agents.values() for tok in tokens)

        # live settings: new protocol text and no gap; re-apply while the agents run
        (exp / "protocols" / "lead-agree.md").write_text("PROTO-V2: the lead posts a plan first.")
        spec_file.write_text(yaml.safe_dump(spec_raw(llm_url, board_url, "0s")))
        await apply(spec_file)
        await until(lambda: w.comms.spec.rules.min_post_gap == "0s", timeout=10)
        t = await prompt(A, f'call: post {{"text": "plan: probes first", "thread": {root}}}', 4)
        out, err = tool_result(t)
        assert not err, out
        last_sys = json.loads(log.read_text().splitlines()[-1])["body"]["messages"][0]["content"]
        assert "PROTO-V2" in last_sys and "PROTO-V1" not in last_sys
        assert [m["sender"]["external_id"] for m in board.msgs].count("cm-host") == 1  # kickoff once
        assert len(sw.workers) == 1
    finally:
        await admin.aclose()
        await sw.close()
        srv.shutdown()
        bsrv.shutdown()
        await asyncio.sleep(0)
