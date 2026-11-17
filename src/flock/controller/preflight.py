"""Preflight: one small tool-calling request per model and inference block in a spec.

Run by the controller, which holds the inference keys (the same secrets it hands to jobs, read
by the names in `api_key_env`). Each probe first checks that the endpoints serve the model (the
inference check's `GET /models`, controller/inference.py), then runs two turns, the way a
harness starts: turn one asks the model to call a trivial tool; turn two sends the model's
output back with the tool's result and must complete. A server that accepts turn one but fails
the follow-up (some HF router providers do) fails preflight. Zero reported usage is a warning.
The probe uses Pi's configured Chat Completions, Responses or Messages API. Task-local
endpoints are reported as unverified here because only their Harbor Trial can reach them.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Mapping
from typing import Any

import httpx

from flock.controller.inference import InferenceError, aget, context_window
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


_THINK = re.compile(r"^\s*(?:<think>)?(.*?)</think>\s*", re.DOTALL)


def visible(content: Any) -> Any:
    """A reply without its inline reasoning (`<think>...</think>answer`, or `...</think>answer`
    when the chat template opened the tag): what goes back in the follow-up turn."""
    if not isinstance(content, str):
        return content
    if m := _THINK.match(content):
        return content[m.end():]
    return "" if content.lstrip().startswith("<think>") else content


async def _probe_model(client, r: ResolvedModel, key: str | None, result: PreflightResult,
                      timeout: float) -> int:
    """Two tool-calling turns on every server; returns tokens reported. Raises ProbeError
    (NoToolCall: a server made no tool call). With several `urls`, the error names the server
    that failed."""
    urls, total = r.inference.endpoints, 0
    for base in urls:
        try:
            call = _probe_chat_at if r.inference.api_format == "openai_chat_completions" else _probe_native_at
            total += await call(client, base, r, key, result, timeout)
        except ProbeError as e:
            if len(urls) == 1:
                raise
            raise type(e)(f"{base}: {e}") from e
    return total


async def _probe_chat_at(client, base: str, r: ResolvedModel, key: str | None,
                         result: PreflightResult, timeout: float) -> int:
    url = f"{base}/chat/completions"
    headers = {**r.inference.headers, **({"Authorization": f"Bearer {key}"} if key else {})}
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


async def _probe_native_at(client, base, r, key, result, timeout):
    """Echo a tool call and result through Responses or Messages, retaining provider content."""
    messages_api = r.inference.api_format == "anthropic_messages"
    headers = dict(r.inference.headers)
    if messages_api:
        headers.update({"anthropic-version": "2023-06-01", **({"x-api-key": key} if key else {})})
        url = base.removesuffix("/v1") + "/v1/messages"
        body = {"model": r.request_model, "max_tokens": MAX_OUTPUT_TOKENS,
                "messages": [{"role": "user", "content": PROMPT}],
                "tools": [{"name": "get_time", "description": TOOL_DESCRIPTION, "input_schema": TOOL_SCHEMA}]}
    else:
        headers.update({"Authorization": f"Bearer {key}"} if key else {})
        url = base + "/responses"
        body = {"model": r.request_model, "max_output_tokens": MAX_OUTPUT_TOKENS,
                "input": [{"role": "user", "content": PROMPT}],
                "tools": [{"type": "function", "name": "get_time", "description": TOOL_DESCRIPTION,
                           "parameters": TOOL_SCHEMA}]}

    async def turn(value):
        response = await client.post(url, json=value, headers=headers, timeout=timeout)
        if response.status_code >= 400:
            raise ProbeError(f"HTTP {response.status_code}: {response.text[:200]}")
        reply = response.json()
        if not isinstance(reply, dict) or reply.get("error"):
            raise ProbeError("provider returned an invalid response or an error")
        usage = reply.get("usage") or {}
        tokens = usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
        return reply, tokens

    first, tok1 = await turn(body)
    content = first.get("content" if messages_api else "output") or []
    calls = [item for item in content if isinstance(item, dict)
             and item.get("type") == ("tool_use" if messages_api else "function_call")]
    result.tool_call_seen = bool(calls)
    if not calls:
        raise NoToolCall("the reply has no tool call; agents on this model cannot use tools")
    if messages_api:
        followup = {**body, "messages": [*body["messages"], {"role": "assistant", "content": content},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": c["id"],
                                                  "content": TOOL_RESULT} for c in calls]}]}
    else:
        followup = {**body, "input": [*body["input"], *content,
                    *({"type": "function_call_output", "call_id": c["call_id"], "output": TOOL_RESULT}
                      for c in calls)]}
    _, tok2 = await turn(followup)
    return tok1 + tok2


def hint(r: ResolvedModel) -> str:
    """The likely cause of a failed tool call, for the error."""
    if r.inference.hf_router is not None:
        return (f" (try another provider in {r.provider_name}.hf_router.provider: "
                "some fail tool calls)")
    return (" (check the server's tool-calling flags, e.g. vLLM's --enable-auto-tool-choice "
            "--tool-call-parser for the model family)")


async def probe(
    client: httpx.AsyncClient, r: ResolvedModel, key: str | None, types: list[str],
    api: str | None = None,
) -> PreflightResult:
    """The model check, then turn one asks for a tool call and turn two sends the call back
    with its result through the configured API, without streaming. `key` None with a key
    named: not probed (unverified)."""
    p = r.inference
    api = api or api_for(Harness.harbor, p)
    result = PreflightResult(model=r.qualified, provider=r.provider_name, types=types, api=api,
                             ok=False)
    if p.task_local:
        result.unverified = True
        result.warning = "task-local endpoint: connection must be checked inside its Harbor Trial"
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
        context_window(p, r.model_id, {u: await aget(client, u, key, p) for u in p.endpoints},
                       r.provider_name)
        tok1 = await _probe_model(client, r, key, result, timeout)
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
            "flock cannot count this model's cost")]))
    return result


def api_for(harness: Harness, inference=None) -> str:
    """The type's configured wire API; an unspecified block keeps its harness default."""
    if inference is not None:
        return {"openai_chat_completions": "chat_completions", "openai_responses": "responses",
                "anthropic_messages": "messages"}[inference.api_format]
    return HARNESSES[harness].api


def models_to_probe(spec: SwarmSpec) -> list[tuple[ResolvedModel, list[str], str]]:
    """Distinct (inference block, model id, reasoning, API) in the spec, with their types and
    the API to probe."""
    seen: dict[tuple, tuple[ResolvedModel, list[str], str]] = {}
    for type_name, ts in spec.types.items():
        r = ts.resolved_model
        api = api_for(ts.harness, r.inference)
        key = (r.provider_name, r.model_id, r.reasoning, api)
        seen.setdefault(key, (r, [], api))[1].append(type_name)
    return list(seen.values())


async def run_preflight(
    spec: SwarmSpec, secrets: Mapping[str, str], client: httpx.AsyncClient
) -> PreflightReport:
    results = await asyncio.gather(*(
        probe(client, r, secrets.get(r.inference.key_env) if r.inference.key_env else None,
              types, api)
        for r, types, api in models_to_probe(spec)
    ))
    return PreflightReport(swarm=spec.swarm, ok=all(x.ok or x.unverified for x in results),
                           unverified=sum(x.unverified for x in results), results=list(results))
