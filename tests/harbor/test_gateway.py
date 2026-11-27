"""resources.gateway on 127.0.0.1: per-Trial tokens, the world route (the socket proxy's checks,
against a real legion.world server, or an env world against a fake upstream API) and the
inference route against a fake model server."""

import asyncio
import http.server
import json
import logging
import os
import socket
import subprocess
import sys
import threading
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from harbor.models.trial.config import AgentConfig

from legion.harbor import env as env_world
from legion.harbor.tools.runner import load_module

from ..worker_fakes import until
from . import test_world
from .test_world import SECRET, TOOLS_DIR, WorldResource, world_tools

served = test_world.served  # the fixture: a legion.world server
Gateway = load_module(TOOLS_DIR.parent / "resources" / "gateway.py").Resource
KEY = "hf_real-inference-key"
MODEL = "moonshotai/Kimi-K2.5:deepinfra"
USAGE = {"prompt_tokens": 1500, "completion_tokens": 100,
         "prompt_tokens_details": {"cached_tokens": 500},
         "completion_tokens_details": {"reasoning_tokens": 40}}
CHUNKS = [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
          b'data: {"choices":[],"usage":' + json.dumps(USAGE).encode() + b"}\n\n",
          b"data: [DONE]\n\n"]


class Model(http.server.BaseHTTPRequestHandler):
    """A fake chat completions server. Streamed: the first chunk, then the rest once `go` is set
    (its usage line split in two writes); otherwise one JSON answer."""

    seen, go = [], threading.Event()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Model.seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
        self.send_response(200)
        if not body.get("stream"):
            data = json.dumps({"choices": [], "usage": {"prompt_tokens": 10,
                                                        "completion_tokens": 2}}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        first, usage, done = CHUNKS
        for i, part in enumerate([first, usage[:30], usage[30:], done]):
            if i == 1:
                Model.go.wait(5)
            try:
                self.wfile.write(part)
                self.wfile.flush()
            except BrokenPipeError:  # the gateway ended the call
                return

    def log_message(self, *args):
        pass


@pytest.fixture
def model():
    Model.seen.clear()
    Model.go.clear()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    Model.go.set()
    server.shutdown()


async def started(tmp_path, world=None):
    gw = Gateway({"port": 0}, state_dir=tmp_path, agents=[], secrets=[])
    gw.world, events = world, []
    await gw.start()
    watching = asyncio.create_task(gw.watch(lambda *e: events.append(e)))
    await asyncio.sleep(0)
    return gw, events, watching


def status(url, request):
    """The status of a raw request (sent as written: no client tidies its path)."""
    with socket.create_connection((urlsplit(url).hostname, urlsplit(url).port), 5) as sock:
        sock.sendall(request)
        return int(sock.recv(64).split()[1])


async def test_world_calls_act_only_as_the_tokens_agent(served, tmp_path, monkeypatch):
    monkeypatch.setenv("LEGION_WORLD_TOKEN", SECRET)
    world = WorldResource({"url": served.url, "transport": "gateway"}, state_dir=tmp_path,
                          agents=[], secrets=[])
    await world.start()
    gw, _, watching = await started(tmp_path, world)
    try:
        assert world.bind("agent-001") == {"metadata": {"agent": "agent-001"}}
        binding = gw.bind("agent-001")
        env, token = binding["env"], binding["env"]["WORLD_TOKEN"]
        assert env["WORLD_URL"] == f"{gw.url}/world" and binding["allowed_hosts"] == ["127.0.0.1"]
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("WORLD_AGENT", "agent-002")  # ignored: the gateway names the agent
        assert await asyncio.to_thread(world_tools.call, "submit", {"path": "main"}) == {
            "by": "agent-001", "path": "main"}

        def call(line):
            return status(gw.url, line + b"\r\nX-World-Token: " + token.encode() + b"\r\n"
                          + b"Content-Length: 2\r\n\r\n{}")

        assert await asyncio.to_thread(call, b"POST /world/tools/../hooks HTTP/1.1") == 403
        too_big = b"POST /world/tools/submit HTTP/1.1\r\nX-World-Token: " + token.encode() + (
            b"\r\nContent-Length: 999999999999\r\n\r\n")
        assert await asyncio.to_thread(status, gw.url, too_big) == 413
        assert await asyncio.to_thread(call, b"GET /board HTTP/1.1") == 404
        assert await asyncio.to_thread(call, b"GET /world/manifest HTTP/1.1") == 200
        assert await asyncio.to_thread(
            status, gw.url, b"GET /world/manifest HTTP/1.1\r\n\r\n") == 401

        new = gw.bind("agent-001")["env"]["WORLD_TOKEN"]  # a new Trial: the old token is dead
        assert await asyncio.to_thread(call, b"GET /world/manifest HTTP/1.1") == 401
        await gw.remove(["agent-001"])
        token = new
        assert await asyncio.to_thread(call, b"GET /world/manifest HTTP/1.1") == 401
    finally:
        watching.cancel()
        await gw.stop()
        await world.stop()


ENV_SECRET = "env-secret-0123456789"
CREDS = [env_world.credential(ENV_SECRET, f"agent-00{i}") for i in (1, 2)]  # seat order


class Upstream(http.server.BaseHTTPRequestHandler):
    """A fake upstream API, one bearer token per seat. It echoes each request with the token it
    got, also in its refusals (a careless API): the env world must redact them."""

    seen = []

    def answer(self):
        token = self.headers.get("Authorization", "").removeprefix("Bearer ")
        length = int(self.headers.get("Content-Length") or 0)
        url = urlsplit(self.path)
        seat = CREDS.index(token) if token in CREDS else None
        out = {"seat": seat, "path": url.path, "query": url.query, "token": token,
               "body": json.loads(self.rfile.read(length)) if length else None}
        Upstream.seen.append(out)
        data = json.dumps(out).encode()
        self.send_response(200 if seat is not None and url.path != "/v1/refuse" else 403)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = answer

    def log_message(self, *args):
        pass


def env_spec(tmp_path):
    """A run's spec.yaml with an env world (resources.world.env), and that block."""
    route = {"type": "object", "properties": {"limit": {"type": "integer"}}}
    block = {"name": "ants", "upstream": "http://${UPSTREAM_HOST}:${UPSTREAM_PORT}",
             "secret_env": "ANT_ENV_SECRET", "credential_env": "ANT_TOKEN_{seat}",
             "headers": {"Authorization": "Bearer {credential}"},
             "tools": {"observe": {"method": "GET", "path": "/v1/observe", "query": ["limit"],
                                   "read_only": True, "parameters": route},
                       "act": {"method": "POST", "path": "/v1/action", "parameters": route},
                       "refused": {"method": "POST", "path": "/v1/refuse"}}}
    spec = {"naming": "consecutive-id", "agents": {"count": 2},
            "resources": {"world": {"transport": "gateway", "env": block}}}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return tmp_path / "spec.yaml", block


async def test_an_env_world_acts_as_each_agent_with_its_own_credential(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG)
    Upstream.seen.clear()
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    _, block = env_spec(tmp_path)
    # as on a worker: launch keeps `$${...}` literal, the worker fills it from its environment
    monkeypatch.setenv("UPSTREAM_HOST", "127.0.0.1")
    monkeypatch.setenv("UPSTREAM_PORT", str(upstream.server_port))
    monkeypatch.setenv("ANT_ENV_SECRET", ENV_SECRET)
    world = WorldResource({"transport": "gateway", "env": block}, state_dir=tmp_path,
                          agents=["agent-001", "agent-002"], secrets=[])
    await world.start()
    gw, _, watching = await started(tmp_path, world)

    def call(agent, name, args):
        for key, value in gw.bind(agent)["env"].items():
            monkeypatch.setenv(key, value)
        return world_tools.call(name, args)

    try:
        a = await asyncio.to_thread(call, "agent-001", "observe", {"limit": 3})
        assert (a["seat"], a["path"], a["query"], a["body"]) == (0, "/v1/observe", "limit=3", None)
        b = await asyncio.to_thread(call, "agent-002", "act", {"limit": 1})
        assert (b["seat"], b["path"], b["body"]) == (1, "/v1/action", {"limit": 1})
        with pytest.raises(RuntimeError, match="unknown argument"):  # no identity override
            await asyncio.to_thread(call, "agent-001", "act", {"seat": 1})
        with pytest.raises(RuntimeError, match=r"upstream_403: .*\[redacted\]") as refused:
            await asyncio.to_thread(call, "agent-001", "refused", {})
        assert [s["seat"] for s in Upstream.seen] == [0, 1, 0]
        for secret in [ENV_SECRET, *CREDS]:
            assert secret not in str(refused.value) and secret not in caplog.text
            assert secret in world.secrets  # the worker redacts its logs with them
    finally:
        watching.cancel()
        await gw.stop()
        await world.stop()
        upstream.shutdown()


def test_the_env_launcher_hands_each_seat_the_gateways_credential(tmp_path):
    spec, _ = env_spec(tmp_path)
    show = [sys.executable, "-c", "import json, os; print(json.dumps(dict(os.environ)))"]
    env = {**os.environ, "ANT_ENV_SECRET": ENV_SECRET, "UPSTREAM_HOST": "u", "UPSTREAM_PORT": "1"}
    out = subprocess.run([sys.executable, "-m", "legion.harbor.env", str(spec), "--", *show],
                         env=env, capture_output=True, text=True, check=True).stdout
    child = json.loads(out)
    assert [child["ANT_TOKEN_0"], child["ANT_TOKEN_1"]] == CREDS  # agent-001, agent-002
    assert "ANT_ENV_SECRET" not in child and "ANT_TOKEN_2" not in child


async def test_inference_streams_through_with_the_key_and_counts_each_call_once(
    model, tmp_path, caplog
):
    caplog.set_level(logging.DEBUG)
    gw, events, watching = await started(tmp_path)
    agent = AgentConfig(name="pi", model_name=MODEL, base_url=f"{model}/v1/",
                        model_api_key_name="HF_INFERENCE_TOKEN", env={"HF_INFERENCE_TOKEN": KEY})
    try:
        gw.bind("agent-001")
        gw.route("agent-001", agent, KEY, {"X-HF-Bill-To": "ns"})
        token = agent.env["HF_INFERENCE_TOKEN"]
        assert token != KEY and agent.base_url == f"{gw.url}/inference"
        headers = {"Authorization": f"Bearer {token}", "X-Legion-Agent": "agent-002"}
        ask = {"model": MODEL, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        answers = []
        async with httpx.AsyncClient(base_url=agent.base_url, headers=headers) as client:
            async with client.stream("POST", "/chat/completions", json=ask) as r:
                assert await asyncio.wait_for(anext(r.aiter_raw()), 5) == CHUNKS[0]
            # the agent went away mid-answer: the upstream call ends too, and its slot is free
            await until(lambda: gw.inflight["agent-001"] == 0)
            assert not Model.go.is_set() and events == [] and "broken off" in caplog.text
            Model.seen.clear()
            async with client.stream("POST", "/chat/completions", json=ask) as r:
                chunks = r.aiter_raw()
                first = await asyncio.wait_for(anext(chunks), 5)  # the model holds the rest
                assert first == CHUNKS[0] and events == []
                Model.go.set()
                answers.append(first + b"".join([c async for c in chunks]))
            assert answers[0] == b"".join(CHUNKS)
            (seen,) = Model.seen
            assert seen["path"] == "/v1/chat/completions"
            assert seen["headers"]["Authorization"] == f"Bearer {KEY}"
            assert seen["headers"]["X-HF-Bill-To"] == "ns"
            assert "X-Legion-Agent" not in seen["headers"]
            assert seen["body"]["stream_options"] == {"include_usage": True}  # forced on
            assert events == [("agent-001", "usage", {"input": 1000, "output": 100,
                                                      "cache_read": 500, "cache_write": 0,
                                                      "reasoning": 40})]

            r = await client.post("/chat/completions", json={**ask, "stream": False})
            answers.append(r.content)
            assert r.json()["usage"]["prompt_tokens"] == 10
            assert events[1] == ("agent-001", "usage", {"input": 10, "output": 2, "cache_read": 0,
                                                        "cache_write": 0, "reasoning": 0})
            r = await client.post("/chat/completions", json={**ask, "model": "other"})
            assert r.status_code == 403
            gw.capped = lambda name: name == "agent-001"
            r = await client.post("/chat/completions", json=ask)
            assert r.status_code == 402
            answers.append(r.content)
        assert len(Model.seen) == 2 and len(events) == 2  # refusals never reach the model
        assert (await gw.export(tmp_path)) == {"usage": "usage.json"}
        assert json.loads((tmp_path / "usage.json").read_text())["agent-001"]["calls"] == 3
        assert not [a for a in answers if KEY.encode() in a] and KEY not in caplog.text
    finally:
        watching.cancel()
        await gw.stop()
