"""Preflight: one small tool-calling request per provider-backed model in a spec.

Run by the controller, which holds the provider keys (the same secrets it hands to jobs, read
by the names in `api_key_env`). Each probe asks the model to call a trivial tool and reports
whether a tool call came back. `openai_compat` providers are probed on the Responses API,
because that is the only wire Codex 0.162.1 speaks (`wire_api = "chat"` is rejected);
`anthropic` providers on the Messages API, which is what Claude Code speaks.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
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
TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"timezone": {"type": "string", "description": "IANA timezone, e.g. UTC"}},
    "required": ["timezone"],
}


def responses_request(r: ResolvedModel) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": r.model_id,
        "input": [{"role": "user", "content": PROMPT}],
        "tools": [{"type": "function", "name": "get_time", "description": TOOL_DESCRIPTION,
                   "parameters": TOOL_SCHEMA}],
        "tool_choice": "auto",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
    }
    if r.reasoning is not None:
        body["reasoning"] = {"effort": REASONING_EFFORT[r.reasoning]}
    return body


def messages_request(r: ResolvedModel) -> dict[str, Any]:
    return {
        "model": r.model_id,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "tools": [{"name": "get_time", "description": TOOL_DESCRIPTION, "input_schema": TOOL_SCHEMA}],
        "messages": [{"role": "user", "content": PROMPT}],
    }


def _responses_tool_call(data: dict[str, Any]) -> bool:
    return any(item.get("type") == "function_call" for item in data.get("output") or [])


def _messages_tool_call(data: dict[str, Any]) -> bool:
    return any(block.get("type") == "tool_use" for block in data.get("content") or [])


async def probe(
    client: httpx.AsyncClient, r: ResolvedModel, key: str | None, types: list[str]
) -> PreflightResult:
    p = r.provider
    api = "responses" if p.type == "openai_compat" else "messages"
    result = PreflightResult(model=r.qualified, provider=r.provider_name, types=types, api=api,
                             ok=False)
    if not key:
        result.error = f"secret {p.api_key_env} is not set on the controller"
        return result
    if p.type == "openai_compat":
        url = f"{p.base_url}/responses"
        headers = {"Authorization": f"Bearer {key}"}
        body, seen = responses_request(r), _responses_tool_call
    else:
        url = f"{p.base_url or ANTHROPIC_DEFAULT_BASE_URL}/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
        body, seen = messages_request(r), _messages_tool_call
    t0 = time.monotonic()
    try:
        resp = await client.post(url, json=body, headers=headers, timeout=p.timeout_s or DEFAULT_TIMEOUT_S)
        result.latency_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code >= 400:
            result.error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            return result
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        result.latency_ms = int((time.monotonic() - t0) * 1000)
        result.error = f"{type(e).__name__}: {e}"[:300]
        return result
    result.tool_call_seen = bool(isinstance(data, dict) and seen(data))
    result.ok = result.tool_call_seen
    if not result.ok:
        result.error = "the reply has no tool call; agents on this model cannot use tools"
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
