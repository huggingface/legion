"""The inference check (docs/open-models.md "Model check and context window"): each type's
model is served where its `inference` block says, and the context window it is served with is
pinned into the spec (`resolved_model.context_window`), for Pi's model config and `flock status`.

The controller runs it at apply (`check`) and preflight runs it before its probe
(`served_window`), with `GET <url>/models` on each endpoint as written:
- the HF router lists each model with its providers (`context_length`, `supports_tools`): the
  model must be listed, a named provider must serve it with tools, and the window is that
  provider's; under a policy (no provider, or `fastest`, `cheapest`, `preferred`) it is the
  smallest among the providers with tools;
- a server you run lists the names it serves, and `model` must be one of them; vLLM and SGLang
  report `max_model_len`, and the window is the smallest over the servers.
A server that reports no window takes `inference.context_window`; without either, the apply is
refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from flock.models import HF_ROUTER_URL, HFRouter, Inference, SwarmSpec

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


def served_window(inference: Inference, model: str, listings: Mapping[str, Any]) -> int | None:
    """The context window `model` is served with on `inference`, from each endpoint's
    `GET /models` reply (`listings`, by URL); None when no endpoint reports one. Raises
    InferenceError, naming what is served, when the model is not."""
    if inference.hf_router is not None:
        return _router_window(inference.hf_router, model, listings[HF_ROUTER_URL])
    windows = []
    for url in inference.endpoints:
        entries = _entries(listings[url])
        entry = next((e for e in entries if e.get("id") == model), None)
        if entry is None:
            served = ", ".join(sorted(str(e.get("id")) for e in entries)) or "nothing"
            raise InferenceError(f"{url} does not serve {model!r}; it serves: {served}")
        if (w := _window(entry.get("max_model_len"))) is not None:
            windows.append(w)
    return min(windows) if windows else None


def _router_window(router: HFRouter, model: str, listing: Any) -> int | None:
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
    windows = [w for p in tools if (w := _window(p.get("context_length"))) is not None]
    return min(windows) if windows else None


def _get(client: httpx.Client, url: str, key: str | None) -> Any:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        r = client.get(f"{url}/models", headers=headers, timeout=TIMEOUT_S)
    except httpx.HTTPError as e:
        raise InferenceError(f"GET {url}/models: {type(e).__name__}: {e}") from None
    return _reply(url, r)


async def aget(client: httpx.AsyncClient, url: str, key: str | None) -> Any:
    """`GET <url>/models` (preflight's)."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        r = await client.get(f"{url}/models", headers=headers, timeout=TIMEOUT_S)
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
    type whose block and model are those of `previous` (the applied spec) keeps its window
    without a call, so a re-apply does not need the servers up."""
    listings: dict[tuple[str, str | None], Any] = {}
    with httpx.Client(transport=transport or TRANSPORT) as client:
        for name, ts in spec.types.items():
            r = ts.resolved_model
            old = previous.types[name].resolved_model if previous and name in previous.types else None
            if (old is not None and old.context_window
                    and (old.inference, old.model_id) == (r.inference, r.model_id)):
                r.context_window = old.context_window
                continue
            where = f"types.{name}.model {r.model_id!r} on {r.provider_name}"
            key = secrets.get(r.inference.key_env) if r.inference.key_env else None
            try:
                got = {}
                for url in r.inference.endpoints:
                    if (url, key) not in listings:
                        listings[url, key] = _get(client, url, key)
                    got[url] = listings[url, key]
                served = served_window(r.inference, r.model_id, got)
            except InferenceError as e:
                hint = (f" ({r.inference.key_env} is not set on the controller)"
                        if r.inference.key_env and key is None and "HTTP 401" in str(e) else "")
                raise InferenceError(f"{where}: {e}{hint}") from None
            r.context_window = served or r.inference.context_window
            if r.context_window is None:
                raise InferenceError(f"{where}: no server reports the model's context window: "
                                     f"set {r.provider_name}.context_window")
