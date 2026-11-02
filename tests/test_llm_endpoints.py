"""Several servers per provider (`base_urls`): the spec, rendezvous pinning by agent name, and
the llm harness against two fake OpenAI servers: each agent talks to its own server, a live
add keeps old agents where they are and spreads new ones, a live remove moves only the agents
of the removed server (at their next turn start), and turn_ended names the server."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer

import pytest
from pydantic import ValidationError

from flock.models import ProviderSpec
from flock.worker.adapters.llm import LLMAdapter, endpoint_host, pick_endpoint

from . import fake_openai
from .test_worker_llm import KEY_ENV, llm_spec, turn


def test_spec_forms():
    p = ProviderSpec(type="openai_compat", api_key_env="K", base_urls=["http://a:1/v1/", "http://b:2/v1"])
    assert p.base_url == "http://a:1/v1" and p.endpoints == ["http://a:1/v1", "http://b:2/v1"]
    one = ProviderSpec(type="openai_compat", api_key_env="K", base_url="http://a:1/v1")
    assert one.endpoints == ["http://a:1/v1"] and one.base_urls is None
    for bad, why in (({"base_urls": []}, "at least one"),
                     ({"base_urls": ["http://a/v1", "http://a/v1/"]}, "repeat"),
                     ({"type": "anthropic", "base_urls": ["http://a"]}, "openai_compat")):
        with pytest.raises(ValidationError, match=why):
            ProviderSpec(**{"type": "openai_compat", "api_key_env": "K", **bad})


def test_rendezvous_pinning():
    a, b, c = "http://a:1/v1", "http://b:2/v1", "http://c:3/v1"
    names = [f"agent-{i:04d}" for i in range(1000)]
    two = {n: pick_endpoint(n, [a, b]) for n in names}
    assert two == {n: pick_endpoint(n, [b, a]) for n in names}  # order does not matter
    assert 400 < sum(u == a for u in two.values()) < 600  # roughly even
    # add c: everyone keeps the server they have; new agents spread over the three
    assert all(pick_endpoint(n, [a, b, c], two[n]) == two[n] for n in names)
    fresh = [pick_endpoint(f"new-{i}", [a, b, c]) for i in range(900)]
    assert all(200 < fresh.count(u) < 400 for u in (a, b, c))
    # remove b: a's agents stay, b's re-hash onto what is left
    after = {n: pick_endpoint(n, [a, c], two[n]) for n in names}
    assert all(after[n] == a for n in names if two[n] == a)
    assert {after[n] for n in names if two[n] == b} == {a, c}
    assert pick_endpoint("x", []) is None and endpoint_host(a) == "a:1"


@pytest.fixture
def servers(tmp_path):
    srvs = []
    for i in range(2):
        log = tmp_path / f"requests-{i}.jsonl"  # a Handler per server: its own request log
        handler = type(f"Handler{i}", (fake_openai.Handler,), {"log_path": str(log)})
        s = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        s.log, s.url = log, f"http://127.0.0.1:{s.server_port}/v1"
        srvs.append(s)
    yield srvs
    for s in srvs:
        s.shutdown()


def _hits(srv) -> int:
    return len(srv.log.read_text().splitlines()) if srv.log.exists() else 0


def _name_on(url: str, urls: list[str], skip: set[str] = frozenset()) -> str:
    return next(n for i in range(1000) if (n := f"agent-{i:04d}") not in skip
                and pick_endpoint(n, urls) == url)


def _spec(urls: list[str]):
    spec = llm_spec(urls[0])
    p = spec.resolved_model.provider
    r = spec.resolved_model.model_copy(update={"provider": p.model_copy(update={"base_urls": urls})})
    return spec.model_copy(update={"resolved_model": r})


async def _agent(tmp_path, name: str, spec, monkeypatch) -> LLMAdapter:
    wd = tmp_path / "work" / name
    wd.mkdir(parents=True, exist_ok=True)
    a = LLMAdapter(wd, agent=name, home=tmp_path / "llm", backoff_s=0.01)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    return a


async def _say(a: LLMAdapter, text: str = "hi") -> dict:
    await a.send(text)
    return (await turn(a))[-1].data


async def test_two_servers_pin_add_remove(tmp_path, servers, monkeypatch):
    s1, s2 = servers
    one, both = [s1.url], [s1.url, s2.url]
    host1, host2 = endpoint_host(s1.url), endpoint_host(s2.url)
    # pinning: each agent on the server its name hashes to, every request there
    n1 = _name_on(s1.url, both)
    n2 = _name_on(s2.url, both)
    a1, a2 = await _agent(tmp_path, n1, _spec(both), monkeypatch), await _agent(
        tmp_path, n2, _spec(both), monkeypatch)
    for _ in range(2):
        assert (await _say(a1))["endpoint"] == host1
        assert (await _say(a2))["endpoint"] == host2
    assert (_hits(s1), _hits(s2)) == (2, 2)
    # live remove (server 2 out): a2 moves at its next turn start, a1 stays
    a1.update_spec(_spec(one))
    a2.update_spec(_spec(one))
    d2 = await _say(a2)
    assert (d2["endpoint"], d2["endpoint_from"]) == (host1, host2)
    assert "endpoint_from" not in await _say(a1)
    assert _hits(s2) == 2
    # live add (server 2 back): old agents keep server 1; new agents spread over both
    a1.update_spec(_spec(both))
    a2.update_spec(_spec(both))
    assert (await _say(a1))["endpoint"] == host1 and (await _say(a2))["endpoint"] == host1
    n3 = _name_on(s2.url, both, {n1, n2})
    a3 = await _agent(tmp_path, n3, _spec(both), monkeypatch)
    assert (await _say(a3))["endpoint"] == host2
    sent = [json.loads(x) for x in s2.log.read_text().splitlines()]
    assert sent[-1]["body"]["messages"][-1]["content"] == "hi"
    for a in (a1, a2, a3):
        await a.stop()


async def test_one_url_is_unchanged(tmp_path, servers, monkeypatch):
    s1, _ = servers
    a = await _agent(tmp_path, "agent-0000", llm_spec(s1.url), monkeypatch)
    assert (await _say(a))["endpoint"] == endpoint_host(s1.url)
    await a.stop()
