"""Preflight: one small tool-calling request per provider-backed model in a spec.

Run by the controller, which holds the provider keys (the same secrets it hands to jobs, read
by the names in `api_key_env`). Each probe is two streamed turns, the way a harness starts: turn
one asks the model to call a trivial tool; turn two sends the model's output back with the tool's
result and must complete. A server that accepts turn one but fails the follow-up (the HF router's
auto provider choice did, 2026-10-16) fails preflight. Zero reported usage is a warning. `openai_compat` providers are probed on the Responses API,
because that is the only wire Codex 0.162.1 speaks (`wire_api = "chat"` is rejected);
`anthropic` providers on the Messages API, which is what Claude Code speaks. Types whose harness
is `llm` or `pi` are probed on Chat Completions, non-streamed with the alias's `extra`, as that harness
sends them; a tool call the model writes as text (which the harness accepts) passes with a
warning.
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
    Harness,
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
    harness does on its first tool use. Streamed for Responses and Messages, as those harnesses
    stream; not streamed for Chat Completions (the llm harness)."""
    p = r.provider
    api = api or ("responses" if p.type == "openai_compat" else "messages")
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
        if api == "chat_completions":
            tok1 = await _probe_chat(client, r, key, result, timeout, tool_choice)
        elif p.type == "openai_compat":
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
            auth = {"Authorization": f"Bearer {key}"} if p.auth == "bearer" else {"x-api-key": key}
            headers = {**auth, "anthropic-version": ANTHROPIC_VERSION}
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
        result.warning = "; ".join(filter(None, [result.warning, (
            "the server reported zero token usage on streamed requests; "
            "flock cannot count this model's cost")]))
    return result


def api_for(harness: Harness, r: ResolvedModel) -> str:
    """The wire API a type's harness speaks to its provider (`llm` and Pi: chat completions)."""
    if harness in (Harness.llm, Harness.pi):
        return "chat_completions"
    return "responses" if r.provider.type == "openai_compat" else "messages"


def models_to_probe(spec: SwarmSpec) -> list[tuple[ResolvedModel, list[str], str, str | None]]:
    """Distinct provider-backed (provider, model id, reasoning, API) in the spec, with their
    types, the API to probe and the llm tool_choice. `extra` and `tool_choice` are part of the
    key for chat completions (they are sent)."""
    seen: dict[tuple, tuple[ResolvedModel, list[str], str, str | None]] = {}
    for type_name, ts in spec.types.items():
        r = ts.resolved_model
        if r is None:
            continue
        api = api_for(ts.harness, r)
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
