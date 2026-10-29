"""Loopback-only accounting proxy; admission closes on success, in-flight calls drain."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


def create_app(upstream: str, key: str, output: Path, agents: int = 1):
    # Release first requests together so slow worker startup cannot turn N-agent runs into N=1.
    ready = asyncio.Event()
    arrivals = set()
    active = set()
    closed = False
    journal = output / "model-requests.jsonl"
    stop_file = output / "stop-model-calls"

    def record(row):
        with journal.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=15),
                                     limits=httpx.Limits(max_connections=128)) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"active": len(active), "closed": closed or stop_file.exists()}

    @app.post("/close")
    async def close():
        nonlocal closed
        closed = True
        ready.set()
        stop_file.touch()
        return {"active": len(active), "closed": True}

    def authorize(request):
        if not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + key):
            raise HTTPException(401, "Invalid proxy credential")

    @app.get("/v1/models")
    async def models(request: Request):
        authorize(request)
        response = await app.state.client.get(upstream + "/models",
                                              headers={"Authorization": "Bearer " + key})
        return JSONResponse(response.json(), status_code=response.status_code)

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        authorize(request)
        body = await request.json()
        sid = request.headers.get("x-session-id")
        if not sid:
            raise HTTPException(400, "X-Session-ID is required")
        arrivals.add(sid)
        if len(arrivals) >= agents or closed:
            ready.set()
        try:
            await asyncio.wait_for(ready.wait(), timeout=300)
        except TimeoutError:
            raise HTTPException(503, "Agent startup barrier timed out") from None
        if closed or stop_file.exists() or (output / "success.json").exists():
            # No upstream call. Finish Pi's loop without admitting further model work.
            completion = {"id": "experiment-ended", "object": "chat.completion",
                "model": body["model"], "created": int(time.time()), "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "The experiment has ended."},
                "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0}}
            if body.get("stream"):
                chunk = {**completion, "object": "chat.completion.chunk", "choices": [{"index": 0,
                         "delta": completion["choices"][0]["message"], "finish_reason": "stop"}]}
                return StreamingResponse(iter(["data: " + json.dumps(chunk) + "\n\n",
                                               "data: [DONE]\n\n"]), media_type="text/event-stream")
            return JSONResponse(completion)
        rid = uuid.uuid4().hex
        active.add(rid)
        started = time.time()
        record({"event": "start", "id": rid, "session_id": sid, "time": started,
                "tools": [t["function"]["name"] for t in body.get("tools", [])],
                "settings": {k: body[k] for k in ["model", "temperature", "top_p", "max_tokens",
                             "max_completion_tokens", "reasoning_effort", "tool_choice"] if k in body}})
        if body.get("stream"):
            body["stream_options"] = {**body.get("stream_options", {}), "include_usage": True}
        usage = None
        status = None
        response = None
        try:
            response = await app.state.client.send(app.state.client.build_request("POST",
                upstream + "/chat/completions", json=body,
                headers={"Authorization": "Bearer " + key, "X-Session-ID": sid}), stream=True)
            status = response.status_code
            if not response.is_success:
                await response.aread()
                record({"event": "end", "id": rid, "session_id": sid, "time": time.time(),
                        "status": status, "usage": None})
                await response.aclose()
                active.discard(rid)
                return JSONResponse({"error": {"message": "Upstream model request failed"}},
                                    status_code=status)
            if not body.get("stream"):
                data = json.loads(await response.aread())
                usage = data.get("usage")
                record({"event": "end", "id": rid, "session_id": sid, "time": time.time(),
                        "status": status, "usage": usage})
                await response.aclose()
                active.discard(rid)
                return JSONResponse(data)
        except (httpx.HTTPError, ValueError):
            if response:
                await response.aclose()
            active.discard(rid)
            record({"event": "end", "id": rid, "session_id": sid, "time": time.time(),
                    "status": status, "usage": None, "error": "upstream_transport"})
            raise HTTPException(502, "Upstream model transport failed") from None

        async def stream():
            nonlocal usage
            try:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line[6:] != "[DONE]":
                        with contextlib.suppress(ValueError):
                            event = json.loads(line[6:])
                            if event.get("usage"):
                                usage = event["usage"]
                    yield line + "\n"
            finally:
                await response.aclose()
                record({"event": "end", "id": rid, "session_id": sid, "time": time.time(),
                        "status": status, "usage": usage})
                active.discard(rid)
        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def summarize(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    starts = {r["id"]: r for r in rows if r["event"] == "start"}
    ends = {r["id"]: r for r in rows if r["event"] == "end"}
    input_tokens = output_tokens = cached = 0
    missing = []
    for rid in starts:
        end = ends.get(rid, {})
        usage = end.get("usage")
        if end.get("status") != 200 or not usage or not all(
            isinstance(usage.get(k), int) and usage[k] >= 0
            for k in ("prompt_tokens", "completion_tokens")
        ):
            missing.append(rid)
            continue
        input_tokens += usage.get("prompt_tokens", 0)
        output_tokens += usage.get("completion_tokens", 0)
        cached += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {"complete": bool(starts) and not missing, "requests": len(starts),
            "missing_or_failed_requests": missing, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "cached_input_tokens": cached,
            "total_tokens": input_tokens + output_tokens,
            "session_ids": sorted({r["session_id"] for r in starts.values()})}


if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--agents", type=int, default=1)
    p.add_argument("--upstream", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    uvicorn.run(create_app(args.upstream.rstrip("/"), os.environ["LOCAL_KEY"], args.output, args.agents),
                host="127.0.0.1", port=args.port, access_log=False)
