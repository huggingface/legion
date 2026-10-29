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


def create_app(upstream: str, key: str, output: Path, agents: int = 1, *, request_timeout: float = 900,
               startup_timeout: float = 300):
    # Release first requests together so slow worker startup cannot turn N-agent runs into N=1.
    ready = asyncio.Event()
    arrivals = set()
    active = set()
    producers = set()
    barrier_opened = None
    startup_started = time.monotonic()
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
            try:
                yield
            finally:
                if producers:
                    await asyncio.gather(*producers, return_exceptions=True)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"active": len(active), "closed": closed or stop_file.exists(),
                "expected_agents": agents, "arrived_session_ids": sorted(arrivals),
                "missing_count": max(0, agents - len(arrivals)),
                "barrier_opened": barrier_opened}

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
        nonlocal barrier_opened
        authorize(request)
        body = await request.json()
        sid = request.headers.get("x-session-id")
        if not sid:
            raise HTTPException(400, "X-Session-ID is required")
        if sid not in arrivals:
            arrivals.add(sid)
            record({"event": "barrier_arrival", "session_id": sid, "time": time.time()})
        if len(arrivals) >= agents and barrier_opened is None and not closed:
            barrier_opened = time.time()
            record({"event": "barrier_opened", "time": barrier_opened,
                    "session_ids": sorted(arrivals)})
            ready.set()
        if closed:
            ready.set()
        try:
            await asyncio.wait_for(ready.wait(), timeout=max(0, startup_timeout -
                                                          (time.monotonic() - startup_started)))
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
        # The producer owns the upstream lifetime. Downstream cancellation only detaches
        # its consumer; it cannot cancel the request before trailing usage arrives.
        queue = asyncio.Queue(maxsize=128)
        detached = asyncio.Event()
        headers_ready = asyncio.get_running_loop().create_future()

        async def deliver(item):
            if detached.is_set():
                return
            try:
                await asyncio.wait_for(queue.put(item), timeout=1)
            except TimeoutError:
                # A stalled downstream must not block accounting or grow memory forever.
                detached.set()
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(None)

        async def produce():
            usage = status = error = None
            try:
                async with asyncio.timeout(request_timeout):
                    async with app.state.client.stream("POST", upstream + "/chat/completions",
                        json=body, headers={"Authorization": "Bearer " + key,
                                            "X-Session-ID": sid}) as response:
                        status = response.status_code
                        headers_ready.set_result(status)
                        if not response.is_success:
                            await response.aread()
                            error = "upstream_http"
                        elif not body.get("stream"):
                            data = json.loads(await response.aread())
                            usage = data.get("usage")
                            await deliver(data)
                        else:
                            async for line in response.aiter_lines():
                                if line.startswith("data: ") and line[6:] != "[DONE]":
                                    with contextlib.suppress(ValueError):
                                        event = json.loads(line[6:])
                                        if event.get("usage"):
                                            usage = event["usage"]
                                await deliver(line + "\n")
            except TimeoutError:
                error = "request_deadline"
            except (httpx.HTTPError, ValueError):
                error = "upstream_transport"
            finally:
                if not headers_ready.done():
                    headers_ready.set_result(502)
                record({"event": "end", "id": rid, "session_id": sid, "time": time.time(),
                        "status": status, "usage": usage, "error": error})
                active.discard(rid)
                await deliver(None)

        task = asyncio.create_task(produce())
        producers.add(task)
        task.add_done_callback(producers.discard)
        try:
            status = await asyncio.shield(headers_ready)
            if status >= 400:
                detached.set()
                return JSONResponse({"error": {"message": "Upstream model request failed"}},
                                    status_code=status)
            if not body.get("stream"):
                data = await queue.get()
                detached.set()
                if data is None:
                    raise HTTPException(502, "Upstream model transport failed")
                return JSONResponse(data)
        except BaseException:
            detached.set()
            raise

        async def stream():
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item
            finally:
                detached.set()
        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def summarize(path: Path):
    rows = []
    malformed = 0
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or "event" not in row:
                raise ValueError("invalid journal row")
            rows.append(row)
        except ValueError:
            malformed += 1
    starts = {r["id"]: r for r in rows if r["event"] == "start"}
    ends = {r["id"]: r for r in rows if r["event"] == "end"}
    input_tokens = output_tokens = cached = 0
    missing = []
    for rid in starts:
        end = ends.get(rid, {})
        usage = end.get("usage")
        if end.get("status") != 200 or end.get("error") or not usage or not all(
            isinstance(usage.get(k), int) and usage[k] >= 0
            for k in ("prompt_tokens", "completion_tokens")
        ):
            missing.append(rid)
            continue
        input_tokens += usage.get("prompt_tokens", 0)
        output_tokens += usage.get("completion_tokens", 0)
        cached += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {"complete": bool(starts) and not missing and not malformed, "requests": len(starts),
            "malformed_records": malformed,
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
