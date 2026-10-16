"""Preflight: one small tool-calling request per provider-backed model in a spec.

Run by the controller, which holds the provider keys (the same secrets it hands to jobs, read
by the names in `api_key_env`). Each probe is two streamed turns, the way a harness starts: turn
one asks the model to call a trivial tool; turn two sends the model's output back with the tool's
result and must complete. A server that accepts turn one but fails the follow-up (the HF router's
auto provider choice did, 2026-10-16) fails preflight. Zero reported usage is a warning. `openai_compat` providers are probed on the Responses API,
because that is the only wire Codex 0.162.1 speaks (`wire_api = "chat"` is rejected);
`anthropic` providers on the Messages API, which is what Claude Code speaks.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from flock.models import (
    REASONING_EFFORT,
    PreflightReport,
    PreflightResult,
    ResolvedModel,
    SwarmSpec,
)

ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
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


def responses_request(r: ResolvedModel) -> dict[str, Any]:
    """Turn one, shaped like Codex's own requests: streamed, `store: false`, encrypted reasoning
    asked for, parallel tool calls on."""
    body: dict[str, Any] = {
        "model": r.model_id,
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": PROMPT}]}],
        "tools": [{"type": "function", "name": "get_time", "description": TOOL_DESCRIPTION,
                   "parameters": TOOL_SCHEMA}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
    }
    if r.reasoning is not None:
        body["reasoning"] = {"effort": REASONING_EFFORT[r.reasoning]}
    return body


def messages_request(r: ResolvedModel) -> dict[str, Any]:
    return {
        "model": r.model_id,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": True,
        "tools": [{"name": "get_time", "description": TOOL_DESCRIPTION, "input_schema": TOOL_SCHEMA}],
        "messages": [{"role": "user", "content": PROMPT}],
    }


class ProbeError(Exception):
    pass


async def _sse(client: httpx.AsyncClient, url: str, body: dict[str, Any], headers: dict[str, str],
               timeout: float) -> AsyncIterator[dict[str, Any]]:
    """POST and yield each server-sent event's JSON `data` (with `_event` set from an `event:`
    line when the server sends one)."""
    async with client.stream("POST", url, json=body, headers=headers, timeout=timeout) as resp:
        if resp.status_code >= 400:
            text = (await resp.aread()).decode(errors="replace")
            raise ProbeError(f"HTTP {resp.status_code}: {text[:300]}")
        event = None
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    try:
                        obj = json.loads(data)
                    except ValueError as e:
                        raise ProbeError(f"bad stream data: {data[:200]}") from e
                    if isinstance(obj, dict):
                        yield {"_event": event, **obj}
                event = None


def _tokens(usage: dict[str, Any] | None) -> int:
    u = usage or {}
    return sum(int(u.get(k) or 0) for k in ("input_tokens", "output_tokens"))


async def _responses_turn(client, url, body, headers, timeout) -> tuple[list[dict], int]:
    """One streamed Responses call -> (output items, tokens reported)."""
    final, items = None, []
    async for ev in _sse(client, url, body, headers, timeout):
        t = ev.get("type") or ev.get("_event")
        if t == "response.output_item.done" and isinstance(ev.get("item"), dict):
            items.append(ev["item"])
        elif t in ("response.completed", "response.failed", "response.incomplete"):
            final = ev.get("response") or {}
        elif t == "error":
            raise ProbeError(f"stream error: {json.dumps(ev.get('error') or ev)[:300]}")
    if final is None:
        raise ProbeError("the stream ended without response.completed")
    if final.get("status") != "completed":
        err = final.get("error") or final.get("incomplete_details")
        raise ProbeError(f"response {final.get('status')}: {json.dumps(err)[:300]}")
    return (final.get("output") or items), _tokens(final.get("usage"))


async def _messages_turn(client, url, body, headers, timeout) -> tuple[list[dict], int]:
    """One streamed Messages call -> (content blocks, tokens reported)."""
    blocks: dict[int, dict[str, Any]] = {}
    partial: dict[int, str] = {}
    tokens, stopped = 0, False
    async for ev in _sse(client, url, body, headers, timeout):
        t = ev.get("type") or ev.get("_event")
        if t == "message_start":
            tokens += _tokens((ev.get("message") or {}).get("usage"))
        elif t == "content_block_start":
            blocks[ev["index"]] = dict(ev.get("content_block") or {})
        elif t == "content_block_delta":
            d, b = ev.get("delta") or {}, blocks.setdefault(ev["index"], {})
            if d.get("type") == "text_delta":
                b["text"] = b.get("text", "") + d.get("text", "")
            elif d.get("type") == "input_json_delta":
                partial[ev["index"]] = partial.get(ev["index"], "") + d.get("partial_json", "")
        elif t == "message_delta":
            tokens += int(((ev.get("usage") or {}).get("output_tokens")) or 0)
        elif t == "message_stop":
            stopped = True
        elif t == "error":
            raise ProbeError(f"stream error: {json.dumps(ev.get('error') or ev)[:300]}")
    if not stopped:
        raise ProbeError("the stream ended without message_stop")
    for i, js in partial.items():
        try:
            blocks[i]["input"] = json.loads(js) if js else {}
        except ValueError as e:
            raise ProbeError(f"tool input is not JSON: {js[:200]}") from e
    return [blocks[i] for i in sorted(blocks)], tokens


async def probe(
    client: httpx.AsyncClient, r: ResolvedModel, key: str | None, types: list[str]
) -> PreflightResult:
    """Turn one asks for a tool call; turn two sends the call back with its result, the way a
    harness does on its first tool use. Both streamed, as the harnesses stream."""
    p = r.provider
    api = "responses" if p.type == "openai_compat" else "messages"
    result = PreflightResult(model=r.qualified, provider=r.provider_name, types=types, api=api,
                             ok=False)
    if not key:
        result.error = f"secret {p.api_key_env} is not set on the controller"
        return result
    timeout = p.timeout_s or DEFAULT_TIMEOUT_S
    t0 = time.monotonic()
    try:
        if p.type == "openai_compat":
            url, headers = f"{p.base_url}/responses", {"Authorization": f"Bearer {key}"}
            body = responses_request(r)
            out, tok1 = await _responses_turn(client, url, body, headers, timeout)
            calls = [x for x in out if x.get("type") == "function_call"]
            result.tool_call_seen = bool(calls)
            if calls:
                body2 = {**body, "input": [*body["input"], *out, *(
                    {"type": "function_call_output", "call_id": c.get("call_id"), "output": TOOL_RESULT}
                    for c in calls)]}
                _, tok2 = await _responses_turn(client, url, body2, headers, timeout)
        else:
            url = f"{p.base_url or ANTHROPIC_DEFAULT_BASE_URL}/v1/messages"
            headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
            body = messages_request(r)
            content, tok1 = await _messages_turn(client, url, body, headers, timeout)
            uses = [b for b in content if b.get("type") == "tool_use"]
            result.tool_call_seen = bool(uses)
            if uses:
                sent = [b for b in content if b.get("type") in ("text", "tool_use")
                        and (b.get("type") != "text" or b.get("text"))]
                body2 = {**body, "messages": [*body["messages"], {"role": "assistant", "content": sent},
                         {"role": "user", "content": [
                             {"type": "tool_result", "tool_use_id": b.get("id"), "content": TOOL_RESULT}
                             for b in uses]}]}
                _, tok2 = await _messages_turn(client, url, body2, headers, timeout)
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
        result.warning = ("the server reported zero token usage on streamed requests; "
                          "flock cannot count this model's cost")
    return result


def models_to_probe(spec: SwarmSpec) -> list[tuple[ResolvedModel, list[str]]]:
    """Distinct provider-backed (provider, model id, reasoning) in the spec, with their types."""
    seen: dict[tuple[str, str, str | None], tuple[ResolvedModel, list[str]]] = {}
    for type_name, ts in spec.types.items():
        r = ts.resolved_model
        if r is None:
            continue
        key = (r.provider_name, r.model_id, r.reasoning)
        seen.setdefault(key, (r, []))[1].append(type_name)
    return list(seen.values())


async def run_preflight(
    spec: SwarmSpec, secrets: Mapping[str, str], client: httpx.AsyncClient
) -> PreflightReport:
    results = await asyncio.gather(*(
        probe(client, r, secrets.get(r.provider.api_key_env), types)
        for r, types in models_to_probe(spec)
    ))
    return PreflightReport(swarm=spec.swarm, ok=all(x.ok for x in results), results=list(results))
