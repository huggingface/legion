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
  report `max_model_len`.
The window is the smallest over the servers or providers a request may reach, each one's own,
or `inference.context_window` for one that reports none; without either, the apply is refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx

from legion.models import HF_ROUTER_URL, HFRouter, Inference, SwarmSpec

TIMEOUT_S = 15.0
# The `check` transport when its caller passes none (None: the network). Tests and their tiny
# runs put a fake model server here (tests/fake_models.py).
TRANSPORT: httpx.BaseTransport | None = None


ANTHROPIC_VERSION = "2023-06-01"


class InferenceError(Exception):
    pass


def auth_headers(url: str, key: str | None) -> dict[str, str]:
    """How `url`'s API takes a key: Anthropic's own header (and API version), else a bearer."""
    if not key:
        return {}
    if urlsplit(url).hostname == "api.anthropic.com":
        return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
    return {"Authorization": f"Bearer {key}"}


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


def _get(client: httpx.Client, url: str, key: str | None) -> Any:
    try:
        r = client.get(f"{url}/models", headers=auth_headers(url, key), timeout=TIMEOUT_S)
    except httpx.HTTPError as e:
        raise InferenceError(f"GET {url}/models: {type(e).__name__}: {e}") from None
    return _reply(url, r)


async def aget(client: httpx.AsyncClient, url: str, key: str | None) -> Any:
    """`GET <url>/models` (preflight's)."""
    try:
        r = await client.get(f"{url}/models", headers=auth_headers(url, key), timeout=TIMEOUT_S)
    except httpx.HTTPError as e:
        raise InferenceError(f"GET {url}/models: {type(e).__name__}: {e}") from None
    return _reply(url, r)


def _reply(url: str, r: httpx.Response) -> Any:
    if r.status_code >= 400:
        raise InferenceError(f"GET {url}/models: HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json()
    except ValueError:
        raise InferenceError(f"GET {url}/models: not JSON: {r.text[:200]}") from None


def check(spec: SwarmSpec, secrets: Mapping[str, str],
          transport: httpx.BaseTransport | None = None,
          previous: SwarmSpec | None = None) -> None:
    """Pin each type's context window into `spec`, or raise InferenceError naming the type. A
    type whose block (but `max_working`, the controller's) and model are those of `previous`
    (the applied spec) keeps its window without a call: a re-apply does not need the servers
    up, and the window stays the one its workers' Trials hold."""
    listings: dict[tuple[str, str | None], Any] = {}
    with httpx.Client(transport=transport or TRANSPORT) as client:
        for name, ts in spec.types.items():
            r = ts.resolved_model
            old = previous.types[name].resolved_model if previous and name in previous.types else None
            if (old is not None and old.context_window and (old.inference.held(), old.model_id)
                    == (r.inference.held(), r.model_id)):
                r.context_window = old.context_window
                continue
            where = f"types.{name}.model {r.model_id!r} on {r.provider_name}"
            key = secrets.get(r.inference.key_env) if r.inference.key_env else None
            if key is None and ts.harbor and ts.harbor.config.get("harness"):
                # Codex and Claude Code call their provider's API: no keyless or logged-in default
                raise InferenceError(f"{where}: {ts.harbor.config['harness']} needs the secret "
                                     f"{r.inference.key_env}, which is not set on the controller")
            try:
                got = {}
                for url in r.inference.endpoints:
                    if (url, key) not in listings:
                        listings[url, key] = _get(client, url, key)
                    got[url] = listings[url, key]
                r.context_window = context_window(r.inference, r.model_id, got, r.provider_name)
            except InferenceError as e:
                hint = (f" ({r.inference.key_env} is not set on the controller)"
                        if r.inference.key_env and key is None and "HTTP 401" in str(e) else "")
                raise InferenceError(f"{where}: {e}{hint}") from None
