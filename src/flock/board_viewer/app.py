"""Standalone, read-only Space; no controller or board credentials are accepted."""
from __future__ import annotations

import asyncio
import contextlib
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from huggingface_hub import HfApi

from .auth import COOKIE, Access
from .cache import BoardCache


def create_app(cache=None, access=None, *, refresh=True):
    if cache is None or access is None:
        api = HfApi(token=os.environ["HF_TOKEN"])
        cache = cache or BoardCache(api, os.getenv("FLOCK_VIEWER_BUCKET",
            "HuggingFaceSwarms/Experiments"), Path(os.getenv("FLOCK_VIEWER_CACHE", "/tmp/flock-viewer")))
        access = access or Access(api, os.getenv("FLOCK_VIEWER_ORG", "HuggingFaceSwarms"),
            os.environ["OAUTH_CLIENT_ID"], os.environ["OAUTH_CLIENT_SECRET"],
            "https://" + os.environ["SPACE_HOST"])

    @asynccontextmanager
    async def lifespan(app):
        async def update():
            while True:
                await asyncio.to_thread(cache.refresh)
                await asyncio.sleep(30)
        task = asyncio.create_task(update()) if refresh else None
        yield
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.cache, app.state.access = cache, access

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'self' https://huggingface.co")
        return response

    @app.get("/healthz")
    def health():
        return {"ok": True}

    @app.get("/auth/login")
    def login():
        return access.login()

    @app.get("/auth/callback")
    def callback(request: Request, code: str, state: str):
        return access.complete(request, code, state)

    @app.post("/auth/logout")
    def logout(request: Request):
        access.sessions.pop(request.cookies.get(COOKIE, ""), None)
        response = RedirectResponse("/", status_code=303)
        response.delete_cookie(COOKIE, secure=True, httponly=True)
        return response

    @app.get("/", response_class=HTMLResponse)
    @app.get("/boards/{run_id}/", response_class=HTMLResponse)
    def index(request: Request):
        try:
            access.require(request)
        except HTTPException as exc:
            if exc.status_code != 401:
                raise
            return HTMLResponse('<!doctype html><title>Flock boards</title><h1>Flock boards</h1>'
                '<p>Organization access is required to view experiment conversations.</p>'
                '<a href="/auth/login" target="_top">Sign in with Hugging Face</a>')
        return HTMLResponse(Path(__file__).with_name("index.html").read_text())

    protected = [Depends(access.require)]

    @app.get("/viewer.js", dependencies=protected)
    def javascript():
        return FileResponse(Path(__file__).with_name("viewer.js"), media_type="text/javascript")

    @app.get("/api/boards", dependencies=protected)
    def boards():
        return {"boards": cache.summaries(), "refresh_error": cache.error}

    def board(run_id):
        b = cache.boards.get(run_id)
        if not b:
            raise HTTPException(404, "Board not available")
        return b

    @app.get("/api/boards/{run_id}", dependencies=protected)
    def details(run_id: str):
        b = board(run_id)
        return {"manifest": b["manifest"], "refresh_error": b["refresh_error"],
                **{k: b["data"][k] for k in ("channels", "principals", "channel_members")}}

    @app.get("/api/boards/{run_id}/messages", dependencies=protected)
    def messages(run_id: str, channel: int | None = None, author: int | None = None,
                 thread: int | None = None, focus: int | None = None, q: str = Query("", max_length=500),
                 offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)):
        b = board(run_id)["data"]
        rows = [m for m in b["messages"]
                if (channel is None or m["channel_id"] == channel)
                and (author is None or m["sender_id"] == author)
                and (thread is None or (m["thread_id"] or m["id"]) == thread)
                and q.casefold() in m["body"].casefold()]
        rows.sort(key=lambda m: m["id"])
        if focus is not None:
            index = next((i for i, row in enumerate(rows) if row["id"] == focus), None)
            if index is not None:
                offset = index // limit * limit
        page = rows[offset:offset + limit]
        ids = {m["id"] for m in page}
        attachments = {a for m in page for a in m["attachments"]}
        return {"messages": page, "total": len(rows), "offset": offset,
                "reactions": [r for r in b["message_reactions"] if r["message_id"] in ids],
                "attachments": [{k: a[k] for k in ("id", "filename", "size")}
                                for a in b["attachments"] if a["id"] in attachments]}

    @app.get("/api/boards/{run_id}/messages/{message_id}", dependencies=protected)
    def message(run_id: str, message_id: int):
        b = board(run_id)["data"]
        row = next((m for m in b["messages"] if m["id"] == message_id), None)
        if row is None:
            raise HTTPException(404, "Message not found")
        return row

    @app.get("/api/boards/{run_id}/attachments/{attachment_id}", dependencies=protected)
    def attachment(run_id: str, attachment_id: str):
        b = board(run_id)
        a = next((a for a in b["data"]["attachments"] if a["id"] == attachment_id), None)
        if a is None:
            raise HTTPException(404, "Attachment not found")
        # Force download, including HTML/SVG, instead of executing same-origin user content.
        filename = Path((a["filename"] or "attachment").replace("\\", "/")).name
        return FileResponse(b["path"] / a["file"], media_type="application/octet-stream",
                            filename=filename)

    return app


def main():
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.getenv("PORT", "7860")),
                access_log=False)  # OAuth callback query strings contain one-time codes.


if __name__ == "__main__":
    main()
