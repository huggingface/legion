"""Preflight: one small tool-calling request per provider-backed model in a spec.

Run by the controller, which holds the provider keys (the same secrets it hands to jobs, read
by the names in `api_key_env`). Each probe is two streamed turns, the way a harness starts: turn
one asks the model to call a trivial tool; turn two sends the model's output back with the tool's
result and must complete. A server that accepts turn one but fails the follow-up (the HF router's
auto provider choice did, 2026-10-16) fails preflight. Zero reported usage is a warning. Every
harness speaks Chat Completions (`llm`, `pi`, `harbor`), so models are probed there, non-streamed
with the alias's `extra`, as the llm harness sends them; a tool call the model writes as text
(which the llm harness accepts) passes with a warning.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from typing import Any

import httpx

from flock.models import (
    Harness,
    PreflightReport,
    PreflightResult,
    ResolvedModel,
    SwarmSpec,
)
from flock.registry import HARNESSES

DEFAULT_TIMEOUT_S = 60.0
MAX_OUTPUT_TOKENS = 1024
PROMPT = (
    "This is a connectivity check. Call the get_time tool with timezone set to UTC. "
    "Do not answer in text."
)
TOOL_DESCRIPTION = "Return the current time in a timezone."
TOOL_RESULT = "2026-01-01T12:00:00Z"
TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"timezone": {"type": "string", "description": "IANA timezone, e.g. UTC"}},
    "required": ["timezone"],
}


def chat_request(r: ResolvedModel, tool_choice: str | None = None) -> dict[str, Any]:
    """Turn one as the llm harness sends it: non-streamed, temperature 0, `extra` on top, the
    type's `tool_choice` last (a named tool forces the probe's own tool)."""
    body: dict[str, Any] = {
        "model": r.model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": [{"type": "function", "function": {"name": "get_time", "description": TOOL_DESCRIPTION,
                                                    "parameters": TOOL_SCHEMA}}],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        **(r.extra or {}),
    }
    body.pop("stream", None)
    # an agent's response_format describes its own output, which the get_time probe cannot match
    body.pop("response_format", None)
    if tool_choice is not None:
        body["tool_choice"] = tool_choice if tool_choice in ("auto", "required", "none") else {
            "type": "function", "function": {"name": "get_time"}}
    return body


class ProbeError(Exception):
    pass


async def _chat_turn(client, url, body, headers, timeout) -> tuple[dict[str, Any], int]:
    """One non-streamed Chat Completions call -> (assistant message, tokens reported)."""
    resp = await client.post(url, json=body, headers=headers, timeout=timeout)
    if resp.status_code >= 400:
        raise ProbeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        data = resp.json()
    except ValueError as e:
        raise ProbeError(f"not JSON: {resp.text[:200]}") from e
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        raise ProbeError(f"no choices in the reply: {json.dumps(data)[:300]}")
    u = data.get("usage") or {}
    tokens = int(u.get("prompt_tokens") or 0) + int(u.get("completion_tokens") or 0)
    return choices[0].get("message") or {}, tokens


def _text_get_time_call(content: Any) -> bool:
    """The model wrote the tool call as text, e.g. {"tool": "get_time", ...} or get_time(...)."""
    from flock.worker.adapters.llm import split_thinking, text_tool_calls

    return bool(text_tool_calls(split_thinking(content)[1], "get_time"))


async def _probe_chat(client, r: ResolvedModel, key: str, result: PreflightResult,
                      timeout: float, tool_choice: str | None = None) -> int:
    """Two Chat Completions turns; returns tokens reported. Raises ProbeError. The follow-up
    goes out with `tool_choice: auto` (a forced choice would demand another call). With
    `base_urls`, every server is probed (an error names the one that failed)."""
    urls, total = r.provider.endpoints or [r.provider.base_url], 0
    for base in urls:
        try:
            total += await _probe_chat_at(client, base, r, key, result, timeout, tool_choice)
        except ProbeError as e:
            if len(urls) == 1:
                raise
            raise ProbeError(f"{base}: {e}") from e
    return total


async def _probe_chat_at(client, base: str, r: ResolvedModel, key: str, result: PreflightResult,
                         timeout: float, tool_choice: str | None) -> int:
    url, headers = f"{base}/chat/completions", {"Authorization": f"Bearer {key}"}
    body = chat_request(r, tool_choice)
    msg, tok1 = await _chat_turn(client, url, body, headers, timeout)
    calls = [c for c in msg.get("tool_calls") or [] if isinstance(c, dict)]
    if not calls and _text_get_time_call(msg.get("content")):
        result.warning = ("the model wrote its tool call as text, not tool_calls; the llm harness "
                          "accepts this (docs/llm-harness.md), other harnesses would not")
        calls = [{"id": "text_0", "type": "function",
                  "function": {"name": "get_time", "arguments": '{"timezone": "UTC"}'}}]
        msg = {"role": "assistant", "content": None, "tool_calls": calls}
    result.tool_call_seen = bool(calls)
    if not calls:
        return tok1
    from flock.worker.adapters.llm import split_thinking

    sent = {"role": "assistant", "content": split_thinking(msg.get("content"))[1], "tool_calls": calls}
    body2 = {**body, "tool_choice": "auto", "messages": [*body["messages"], sent, *(
        {"role": "tool", "tool_call_id": c.get("id"), "content": TOOL_RESULT} for c in calls)]}
    _, tok2 = await _chat_turn(client, url, body2, headers, timeout)
    return tok1 + tok2


async def probe(
    client: httpx.AsyncClient, r: ResolvedModel, key: str | None, types: list[str],
    api: str | None = None, tool_choice: str | None = None,
) -> PreflightResult:
    """Turn one asks for a tool call; turn two sends the call back with its result, the way a
    harness does on its first tool use (Chat Completions, not streamed, as the llm harness)."""
    p = r.provider
    api = api or "chat_completions"
    result = PreflightResult(model=r.qualified, provider=r.provider_name, types=types, api=api,
                             ok=False)
    if not key:
        result.unverified = True
        result.warning = (f"unverified: secret {p.api_key_env} is not set on the controller, "
                          "so the model was not probed")
        return result
    timeout = p.timeout_s or DEFAULT_TIMEOUT_S
    t0 = time.monotonic()
    tok2 = 0
    try:
        tok1 = await _probe_chat(client, r, key, result, timeout, tool_choice)
    except ProbeError as e:
        result.latency_ms = int((time.monotonic() - t0) * 1000)
        turn = "follow-up turn (tool result sent back)" if result.tool_call_seen else "first turn"
        result.error = f"{turn}: {e}"[:400]
        return result
    except (httpx.HTTPError, ValueError, KeyError) as e:
        result.latency_ms = int((time.monotonic() - t0) * 1000)
        result.error = f"{type(e).__name__}: {e}"[:300]
        return result
    result.latency_ms = int((time.monotonic() - t0) * 1000)
    if not result.tool_call_seen:
        result.error = "the reply has no tool call; agents on this model cannot use tools"
        return result
    result.follow_up_ok = result.ok = True
    result.usage_reported = (tok1 + tok2) > 0
    if not result.usage_reported:
        result.warning = "; ".join(filter(None, [result.warning, (
            "the server reported zero token usage on streamed requests; "
            "flock cannot count this model's cost")]))
    return result


def api_for(harness: Harness) -> str:
    """The wire API a type's harness speaks to its provider (`llm` and Pi: chat completions)."""
    return HARNESSES[harness].api


def models_to_probe(spec: SwarmSpec) -> list[tuple[ResolvedModel, list[str], str, str | None]]:
    """Distinct provider-backed (provider, model id, reasoning, API) in the spec, with their
    types, the API to probe and the llm tool_choice. `extra` and `tool_choice` are part of the
    key for chat completions (they are sent)."""
    seen: dict[tuple, tuple[ResolvedModel, list[str], str, str | None]] = {}
    for type_name, ts in spec.types.items():
        r = ts.resolved_model
        if r is None:
            continue
        api = api_for(ts.harness)
        if ts.harness == "harbor":  # Harbor's Pi sends the id without the provider prefix
            from harbor.agents.agent_auth_resolution import parse_model_name

            r = r.model_copy(update={"model_id": parse_model_name(r.model_id)[1]})
        extra = json.dumps(r.extra, sort_keys=True) if api == "chat_completions" else None
        choice = ts.tool_choice if api == "chat_completions" else None
        key = (r.provider_name, r.model_id, r.reasoning, api, extra, choice)
        seen.setdefault(key, (r, [], api, choice))[1].append(type_name)
    return list(seen.values())


async def run_preflight(
    spec: SwarmSpec, secrets: Mapping[str, str], client: httpx.AsyncClient
) -> PreflightReport:
    results = await asyncio.gather(*(
        probe(client, r, secrets.get(r.provider.api_key_env), types, api, choice)
        for r, types, api, choice in models_to_probe(spec)
    ))
    return PreflightReport(swarm=spec.swarm, ok=all(x.ok or x.unverified for x in results),
                           unverified=sum(x.unverified for x in results), results=list(results))
