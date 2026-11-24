"""A fake model server's `GET /v1/models` (and the HF router's), for the controller's inference
check (legion.controller.inference): every test's apply and every tiny run's controller use it
(conftest.py, tests/harbor/fake_worker.py), so no test reaches a real server."""

from __future__ import annotations

import httpx

# The models the fake serves, on every URL and through every router provider below
MODELS = ("m", "m2", "lead", "fake-model", "Qwen/Qwen3.8-27B", "Qwen/Small", "moonshotai/Kimi-K2.5",
          "deepseek-ai/DeepSeek-V4.1-Flash-1017")
PROVIDERS = ("fake", "cerebras", "deepinfra", "novita")
WINDOW = 32768


def listing(models=MODELS, window: int | None = WINDOW, providers=PROVIDERS) -> dict:
    """One reply that reads both ways: a server's (`max_model_len`) and the router's
    (`providers` with `context_length` and tool support)."""
    extra = {"max_model_len": window} if window else {}
    ctx = {"context_length": window} if window else {}
    return {"object": "list", "data": [
        {"id": m, "object": "model", **extra,
         "providers": [{"provider": p, "status": "live", "supports_tools": True, **ctx}
                       for p in providers]}
        for m in models]}


def transport(reply: dict | None = None, seen: list | None = None) -> httpx.MockTransport:
    """`GET .../models` answers `reply` (default: `listing()`); `seen` collects the URLs asked."""
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(str(request.url))
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json=reply or listing())
        return httpx.Response(404, json={"error": "not faked"})
    return httpx.MockTransport(handle)
