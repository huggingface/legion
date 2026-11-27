"""The inference check (docs/open-models.md "Model check and context window"): each type's
model is served where its `inference` block says, and the context window it is served with is
pinned into the spec (`resolved_model.context_window`), for Pi's model config and `legion status`.

The controller runs it at apply (`check`) and preflight runs it before its probe
(`context_window`), with `GET <url>/models` on each endpoint as written:
- the HF router lists each model with its providers (`context_length`, `supports_tools`): the
  model must be listed, and a named provider must serve it with tools. A request reaches that
  provider, or under a policy (no provider, or `fastest`, `cheapest`, `preferred`) any provider
  with tools;
- a server you run lists the names it serves, and `model` must be one of them; vLLM and SGLang
  report `max_model_len`. With `api_format: anthropic_messages` the listing is `<url>/v1/models`.

A block with `preflight: false` is not checked: its window is `inference.context_window`, or
Pi's default without one.
The window is the smallest over the servers or providers a request may reach, each one's own,
or `inference.context_window` for one that reports none; without either, the apply is refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from legion.models import HF_ROUTER_URL, HFRouter, Inference, SwarmSpec

TIMEOUT_S = 15.0
# The `check` transport when its caller passes none (None: the network). Tests and their tiny
# runs put a fake model server here (tests/fake_models.py).
TRANSPORT: httpx.BaseTransport | None = None


class InferenceError(Exception):
    pass


def _entries(listing: Any) -> list[dict]:
    data = listing.get("data") if isinstance(listing, dict) else None
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


def _window(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def context_window(inference: Inference, model: str, listings: Mapping[str, Any],
                   block: str = "inference") -> int:
    """The context window `model` is served with on `inference` (its block named `block`), from
    each endpoint's `GET /models` reply (`listings`, by URL): the smallest over the servers or
    router providers a request may reach, `inference.context_window` for one that reports none.
    Raises InferenceError, naming what is served when the model is not, or the server or
    provider that reports no window when there is no fallback."""
    windows = []
    for where, window in served_windows(inference, model, listings):
        window = window or inference.context_window
        if window is None:
            raise InferenceError(f"{where} reports no context window for {model!r}: set "
                                 f"{block}.context_window")
        windows.append(window)
    return min(windows)


def served_windows(inference: Inference, model: str,
                   listings: Mapping[str, Any]) -> list[tuple[str, int | None]]:
    """(where, window or None) of each server, or each router provider a request may reach,
    that serves `model`. Raises InferenceError, naming what is served, when the model is not."""
    if inference.hf_router is not None:
        return _router_windows(inference.hf_router, model, listings[HF_ROUTER_URL])
    out = []
    for url in inference.endpoints:
        entries = _entries(listings[url])
        entry = next((e for e in entries if e.get("id") == model), None)
        if entry is None:
            served = ", ".join(sorted(str(e.get("id")) for e in entries)) or "nothing"
            raise InferenceError(f"{url} does not serve {model!r}; it serves: {served}")
        out.append((url, _window(entry.get("max_model_len"))))
    return out


def _router_windows(router: HFRouter, model: str, listing: Any) -> list[tuple[str, int | None]]:
    entries = _entries(listing)
    entry = next((e for e in entries if e.get("id") == model), None)
    if entry is None:
        near = [e["id"] for e in entries if str(e.get("id", "")).lower() == model.lower()]
        raise InferenceError(f"the HF router does not list {model!r}"
                             + (f" (did you mean {near[0]!r}?)" if near else "")
                             + f": see {HF_ROUTER_URL}/models")
    providers = [p for p in entry.get("providers") or [] if isinstance(p, dict)]
    tools = [p for p in providers if p.get("supports_tools") is True]
    if not router.policy:
        named = next((p for p in providers if p.get("provider") == router.provider), None)
        if named is None or named.get("supports_tools") is not True:
            listed = ", ".join(sorted(str(p.get("provider")) for p in tools)) or "none"
            what = "does not serve" if named is None else "does not list tool support for"
            raise InferenceError(f"HF router provider {router.provider!r} {what} {model!r}; "
                                 f"providers with tool support: {listed}")
        tools = [named]
    elif not tools:
        raise InferenceError(f"no HF router provider lists tool support for {model!r}")
    return [(f"HF router provider {p.get('provider')!r}", _window(p.get("context_length")))
            for p in tools]


ANTHROPIC_VERSION = "2023-06-01"


def api_root(inference: Inference, url: str) -> str:
    """Where the API's paths start on endpoint `url`: Anthropic's are under `/v1`, which its
    URLs leave out (as Pi takes them); the OpenAI APIs' URLs end in it."""
    return f"{url}/v1" if inference.api_format == "anthropic_messages" else url


def auth_headers(inference: Inference, key: str | None) -> dict[str, str]:
    """The key as the block's API takes it: `x-api-key` (and the version header) for
    Anthropic Messages, else `Authorization: Bearer`."""
    if inference.api_format == "anthropic_messages":
        return {"anthropic-version": ANTHROPIC_VERSION, **({"x-api-key": key} if key else {})}
    return {"Authorization": f"Bearer {key}"} if key else {}


def _get(client: httpx.Client, inference: Inference, url: str, key: str | None) -> Any:
    models = f"{api_root(inference, url)}/models"
    try:
        r = client.get(models, headers=auth_headers(inference, key), timeout=TIMEOUT_S)
    except httpx.HTTPError as e:
        raise InferenceError(f"GET {models}: {type(e).__name__}: {e}") from None
    return _reply(models, r)


async def aget(client: httpx.AsyncClient, inference: Inference, url: str,
               key: str | None) -> Any:
    """`GET <url>/models` (preflight's)."""
    models = f"{api_root(inference, url)}/models"
    try:
        r = await client.get(models, headers=auth_headers(inference, key), timeout=TIMEOUT_S)
    except httpx.HTTPError as e:
        raise InferenceError(f"GET {models}: {type(e).__name__}: {e}") from None
    return _reply(models, r)


def _reply(models: str, r: httpx.Response) -> Any:
    if r.status_code >= 400:
        raise InferenceError(f"GET {models}: HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json()
    except ValueError:
        raise InferenceError(f"GET {models}: not JSON: {r.text[:200]}") from None


def check(spec: SwarmSpec, secrets: Mapping[str, str],
          transport: httpx.BaseTransport | None = None,
          previous: SwarmSpec | None = None) -> None:
    """Pin each type's context window into `spec`, or raise InferenceError naming the type. A
    type whose block (but `max_working`, the controller's) and model are those of `previous`
    (the applied spec) keeps its window without a call: a re-apply does not need the servers
    up, and the window stays the one its workers' Trials hold."""
    listings: dict[tuple[str, str, str | None], Any] = {}
    with httpx.Client(transport=transport or TRANSPORT) as client:
        for name, ts in spec.types.items():
            r = ts.resolved_model
            old = previous.types[name].resolved_model if previous and name in previous.types else None
            if (old is not None and old.context_window and (old.inference.held(), old.model_id)
                    == (r.inference.held(), r.model_id)):
                r.context_window = old.context_window
                continue
            if not r.inference.preflight:  # endpoints the controller cannot reach
                r.context_window = r.inference.context_window
                continue
            where = f"types.{name}.model {r.model_id!r} on {r.provider_name}"
            key = secrets.get(r.inference.key_env) if r.inference.key_env else None
            try:
                got = {}
                for url in r.inference.endpoints:
                    at = (url, r.inference.api_format, key)
                    if at not in listings:
                        listings[at] = _get(client, r.inference, url, key)
                    got[url] = listings[at]
                r.context_window = context_window(r.inference, r.model_id, got, r.provider_name)
            except InferenceError as e:
                hint = (f" ({r.inference.key_env} is not set on the controller)"
                        if r.inference.key_env and key is None and "HTTP 401" in str(e) else "")
                raise InferenceError(f"{where}: {e}{hint}") from None
