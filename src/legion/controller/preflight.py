"""Preflight: one small tool-calling request per model and inference block in a spec.

Run by the controller, which holds the inference keys (the same secrets it hands to jobs, read
by the names in `api_key_env`). Each probe first checks that the endpoints serve the model (the
inference check's `GET /models`, controller/inference.py), then runs two turns, the way a
harness starts: turn one asks the model to call a trivial tool; turn two sends the model's
output back with the tool's result and must complete. A server that accepts turn one but fails
the follow-up (some HF router providers do) fails preflight. Zero reported usage is a warning.
Models are probed, non-streamed, on the API Pi speaks to them (`inference.api_format`): Chat
Completions, Responses or Anthropic Messages. A tool call the model writes as text is no tool
call: Pi takes only the API's structured calls. A block with `preflight: false` is not probed.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Mapping
from typing import Any

import httpx

from legion.controller.inference import (
    InferenceError,
    aget,
    api_root,
    auth_headers,
    context_window,
)
from legion.models import (
    ApiFormat,
    PreflightReport,
    PreflightResult,
    ResolvedModel,
    SwarmSpec,
)

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


def chat_request(r: ResolvedModel) -> dict[str, Any]:
    """Turn one: non-streamed, temperature 0."""
    return {
        "model": r.request_model,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": [{"type": "function", "function": {"name": "get_time", "description": TOOL_DESCRIPTION,
                                                    "parameters": TOOL_SCHEMA}}],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }


class ProbeError(Exception):
    pass


class NoToolCall(ProbeError):
    """A server answered turn one without a tool call."""


async def _post(client, url, body, headers, timeout) -> dict[str, Any]:
    """One non-streamed call -> its JSON object."""
    resp = await client.post(url, json=body, headers=headers, timeout=timeout)
    if resp.status_code >= 400:
        raise ProbeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        data = resp.json()
    except ValueError as e:
        raise ProbeError(f"not JSON: {resp.text[:200]}") from e
    if not isinstance(data, dict):
        raise ProbeError(f"not a JSON object: {resp.text[:200]}")
    return data


def _tokens(usage: Any, *names: str) -> int:
    u = usage if isinstance(usage, dict) else {}
    return sum(int(u.get(n) or 0) for n in names)


async def _chat_turn(client, url, body, headers, timeout) -> tuple[dict[str, Any], int]:
    """One non-streamed Chat Completions call -> (assistant message, tokens reported)."""
    data = await _post(client, url, body, headers, timeout)
    choices = data.get("choices")
    if not choices:
        raise ProbeError(f"no choices in the reply: {json.dumps(data)[:300]}")
    tokens = _tokens(data.get("usage"), "prompt_tokens", "completion_tokens")
    return choices[0].get("message") or {}, tokens


_THINK = re.compile(r"^\s*(?:<think>)?(.*?)</think>\s*", re.DOTALL)


def visible(content: Any) -> Any:
    """A reply without its inline reasoning (`<think>...</think>answer`, or `...</think>answer`
    when the chat template opened the tag): what goes back in the follow-up turn."""
    if not isinstance(content, str):
        return content
    if m := _THINK.match(content):
        return content[m.end():]
    return "" if content.lstrip().startswith("<think>") else content


async def _probe_all(client, r: ResolvedModel, key: str | None, result: PreflightResult,
                     timeout: float) -> int:
    """Two turns on every server, on the block's API; returns tokens reported. Raises
    ProbeError (NoToolCall: a server made no tool call). With several `urls`, the error names
    the server that failed."""
    urls, total = r.inference.endpoints, 0
    at = PROBES[r.inference.api_format]
    for base in urls:
        try:
            total += await at(client, base, r, key, result, timeout)
        except ProbeError as e:
            if len(urls) == 1:
                raise
            raise type(e)(f"{base}: {e}") from e
    return total


async def _probe_chat_at(client, base: str, r: ResolvedModel, key: str | None,
                         result: PreflightResult, timeout: float) -> int:
    url = f"{base}/chat/completions"
    headers = {**r.headers, **auth_headers(r.inference, key)}
    body = chat_request(r)
    msg, tok1 = await _chat_turn(client, url, body, headers, timeout)
    calls = [c for c in msg.get("tool_calls") or [] if isinstance(c, dict)]
    result.tool_call_seen = bool(calls)
    if not calls:
        raise NoToolCall("the reply has no tool call; agents on this model cannot use tools")
    sent = {"role": "assistant", "content": visible(msg.get("content")), "tool_calls": calls}
    body2 = {**body, "messages": [*body["messages"], sent, *(
        {"role": "tool", "tool_call_id": c.get("id"), "content": TOOL_RESULT} for c in calls)]}
    _, tok2 = await _chat_turn(client, url, body2, headers, timeout)
    return tok1 + tok2


def responses_request(r: ResolvedModel) -> dict[str, Any]:
    """Turn one on the Responses API: non-streamed and not stored, as Pi calls it. No
    temperature: reasoning models refuse one."""
    return {
        "model": r.request_model,
        "input": [{"role": "user", "content": PROMPT}],
        "tools": [{"type": "function", "name": "get_time", "description": TOOL_DESCRIPTION,
                   "parameters": TOOL_SCHEMA}],
        "tool_choice": "auto",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }


def _response_output(data: dict[str, Any]) -> list[dict[str, Any]]:
    if data.get("status") != "completed" or data.get("error") or data.get("incomplete_details"):
        raise ProbeError(f"response did not complete: {json.dumps(data)[:300]}")
    output = data.get("output")
    if not isinstance(output, list) or not all(isinstance(item, dict) for item in output):
        raise ProbeError(f"no output in the reply: {json.dumps(data)[:300]}")
    return output


async def _probe_responses_at(client, base: str, r: ResolvedModel, key: str | None,
                              result: PreflightResult, timeout: float) -> int:
    url = f"{base}/responses"
    headers = {**r.headers, **auth_headers(r.inference, key)}
    body = responses_request(r)
    data = await _post(client, url, body, headers, timeout)
    output = _response_output(data)
    calls = [c for c in output if c.get("type") == "function_call"]
    result.tool_call_seen = bool(calls)
    if not calls:
        raise NoToolCall("the reply has no function call; agents on this model cannot use tools")
    # Stateless replay keeps the ordered response items, including encrypted reasoning and ids.
    body2 = {**body, "input": [*body["input"], *output, *(
        {"type": "function_call_output", "call_id": c.get("call_id"), "output": TOOL_RESULT}
        for c in calls)]}
    data2 = await _post(client, url, body2, headers, timeout)
    _response_output(data2)
    return sum(_tokens(d.get("usage"), "input_tokens", "output_tokens") for d in (data, data2))


def messages_request(r: ResolvedModel) -> dict[str, Any]:
    """Turn one on Anthropic's Messages API: non-streamed."""
    return {
        "model": r.request_model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": [{"name": "get_time", "description": TOOL_DESCRIPTION,
                   "input_schema": TOOL_SCHEMA}],
        "tool_choice": {"type": "auto"},
    }


async def _probe_messages_at(client, base: str, r: ResolvedModel, key: str | None,
                             result: PreflightResult, timeout: float) -> int:
    url = f"{api_root(r.inference, base)}/messages"
    headers = {**r.headers, **auth_headers(r.inference, key)}
    body = messages_request(r)
    data = await _post(client, url, body, headers, timeout)
    content = data.get("content")
    if not isinstance(content, list):
        raise ProbeError(f"no content in the reply: {json.dumps(data)[:300]}")
    calls = [c for c in content if isinstance(c, dict) and c.get("type") == "tool_use"]
    result.tool_call_seen = bool(calls)
    if not calls:
        raise NoToolCall("the reply has no tool_use block; agents on this model cannot use "
                         "tools")
    body2 = {**body, "messages": [*body["messages"], {"role": "assistant", "content": content},
                                  {"role": "user", "content": [
                                      {"type": "tool_result", "tool_use_id": c.get("id"),
                                       "content": TOOL_RESULT} for c in calls]}]}
    data2 = await _post(client, url, body2, headers, timeout)
    if not isinstance(data2.get("content"), list):
        raise ProbeError(f"no content in the reply: {json.dumps(data2)[:300]}")
    return sum(_tokens(d.get("usage"), "input_tokens", "output_tokens") for d in (data, data2))


# `inference.api_format` -> (the API's name in the report, one server's two turns)
APIS: dict[ApiFormat, str] = {
    "openai_chat_completions": "chat_completions",
    "openai_responses": "responses",
    "anthropic_messages": "messages",
}
PROBES = {
    "openai_chat_completions": _probe_chat_at,
    "openai_responses": _probe_responses_at,
    "anthropic_messages": _probe_messages_at,
}


def hint(r: ResolvedModel) -> str:
    """The likely cause of a failed tool call, for the error."""
    if r.inference.hf_router is not None:
        return (f" (try another provider in {r.provider_name}.hf_router.provider: "
                "some fail tool calls)")
    return (" (check the server's tool-calling flags, e.g. vLLM's --enable-auto-tool-choice "
            "--tool-call-parser for the model family)")


async def probe(
    client: httpx.AsyncClient, r: ResolvedModel, key: str | None, types: list[str],
) -> PreflightResult:
    """The model check, then turn one asks for a tool call and turn two sends the call back
    with its result, the way a harness does on its first tool use (on the block's API, not
    streamed). `key` None with a key named: not probed (unverified)."""
    p = r.inference
    result = PreflightResult(model=r.qualified, provider=r.provider_name, types=types,
                             api=APIS[p.api_format], ok=False)
    if not p.preflight:
        result.unverified = True
        result.warning = "unverified: inference.preflight is disabled for this block"
        return result
    if p.key_env and not key:
        result.unverified = True
        result.warning = (f"unverified: secret {p.key_env} is not set on the controller, "
                          "so the model was not probed")
        return result
    timeout = DEFAULT_TIMEOUT_S
    t0 = time.monotonic()
    tok2 = 0
    try:
        context_window(p, r.model_id,
                       {u: await aget(client, p, u, key, r.model_id) for u in p.endpoints},
                       r.provider_name)
        tok1 = await _probe_all(client, r, key, result, timeout)
    except InferenceError as e:
        result.latency_ms = int((time.monotonic() - t0) * 1000)
        result.error = str(e)[:400]
        return result
    except NoToolCall as e:
        result.latency_ms = int((time.monotonic() - t0) * 1000)
        result.error = str(e)[:400] + hint(r)
        return result
    except ProbeError as e:
        result.latency_ms = int((time.monotonic() - t0) * 1000)
        turn = "follow-up turn (tool result sent back)" if result.tool_call_seen else "first turn"
        result.error = f"{turn}: {e}"[:400] + hint(r)
        return result
    except (httpx.HTTPError, ValueError, KeyError) as e:
        result.latency_ms = int((time.monotonic() - t0) * 1000)
        result.error = f"{type(e).__name__}: {e}"[:300]
        return result
    result.latency_ms = int((time.monotonic() - t0) * 1000)
    result.follow_up_ok = result.ok = True
    result.usage_reported = (tok1 + tok2) > 0
    if not result.usage_reported:
        result.warning = "; ".join(filter(None, [result.warning, (
            "the server reported zero token usage on streamed requests; "
            "legion cannot count this model's cost")]))
    return result


def models_to_probe(spec: SwarmSpec) -> list[tuple[ResolvedModel, list[str]]]:
    """Distinct (inference block, model id, reasoning, API) in the spec, with their types."""
    seen: dict[tuple, tuple[ResolvedModel, list[str]]] = {}
    for type_name, ts in spec.types.items():
        r = ts.resolved_model
        if r is None:
            continue
        key = (r.provider_name, r.model_id, r.reasoning, r.inference.api_format)
        seen.setdefault(key, (r, []))[1].append(type_name)
    return list(seen.values())


async def run_preflight(
    spec: SwarmSpec, secrets: Mapping[str, str], client: httpx.AsyncClient
) -> PreflightReport:
    results = await asyncio.gather(*(
        probe(client, r, secrets.get(r.inference.key_env) if r.inference.key_env else None,
              types)
        for r, types in models_to_probe(spec)
    ))
    return PreflightReport(swarm=spec.swarm, ok=all(x.ok or x.unverified for x in results),
                           unverified=sum(x.unverified for x in results), results=list(results))
