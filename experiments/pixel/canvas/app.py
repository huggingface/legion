"""FastAPI app. Endpoints are sync (run in the threadpool); World holds one lock."""

from __future__ import annotations

import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from .auth import NAME_RE, Auth, AuthError
from .config import HERE, Settings, parse_hex
from .world import PaintError, RegionConflict, World, hexc, png_bytes

REPLAY_DIR = HERE / "replay"


class Pixel(BaseModel):
    x: int
    y: int
    r: int | None = None
    g: int | None = None
    b: int | None = None
    hex: str | None = None


class PaintBody(BaseModel):
    pixels: list[Pixel]
    turn: str | None = None


class ClaimBody(BaseModel):
    n: int | None = None


class ReleaseBody(BaseModel):
    pixels: list[Pixel] | None = None


class RegionBody(BaseModel):
    x: int
    y: int
    w: int
    h: int
    ttl: float | None = None


class RegionRef(BaseModel):
    id: int | None = None


class GrantBody(BaseModel):
    agent: str
    id: int | None = None


class RegisterBody(BaseModel):
    name: str


def _region_response(world: World, arr, x: int, y: int, fmt: str):
    h, w = arr.shape[:2]
    if fmt == "png":
        return Response(png_bytes(arr), media_type="image/png")
    if fmt == "raw":
        return Response(
            arr.tobytes(),
            media_type="application/octet-stream",
            headers={"X-Width": str(w), "X-Height": str(h)},
        )
    if fmt != "json":
        raise PaintError("format must be json, png or raw")
    if w * h > world.cfg.region_json_max:
        raise PaintError(
            f"region too big for json ({w * h} px > {world.cfg.region_json_max}); "
            "use a smaller w*h or format=png"
        )
    rows = [" ".join(hexc(p) for p in row) for row in arr]
    return {"x": x, "y": y, "w": w, "h": h, "rows": rows}


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or Settings.from_env()
    world = World(cfg)
    auth = Auth(cfg.token, cfg.admin_token, cfg.data_dir)
    stop = threading.Event()

    def snapshot_loop():
        while not stop.wait(cfg.snapshot_seconds):
            try:
                world.snapshot()
            except Exception as e:  # noqa: BLE001
                print(f"canvas: snapshot failed: {e}", flush=True)

    @asynccontextmanager
    async def lifespan(app):
        th = threading.Thread(target=snapshot_loop, daemon=True)
        if cfg.data_dir:
            th.start()
        yield
        stop.set()
        world.close()

    app = FastAPI(title="flock pixel canvas", lifespan=lifespan)
    app.state.world = world
    app.state.auth = auth

    @app.exception_handler(PaintError)
    async def _paint_err(request, exc: PaintError):
        code = 409 if isinstance(exc, RegionConflict) else 400
        return JSONResponse({"error": str(exc)}, status_code=code)

    @app.exception_handler(AuthError)
    async def _auth_err(request, exc: AuthError):
        return JSONResponse({"error": exc.msg}, status_code=exc.status)

    def reader(request: Request) -> None:
        if cfg.read_auth:
            auth.check_read(request.headers)

    def writer(request: Request) -> str:
        agent = auth.identify(request.headers)
        wait = world.check_rate(agent)
        if wait > 0:
            raise _RateLimited(wait, cfg.rate_limit)
        return agent

    class _RateLimited(Exception):
        def __init__(self, wait: float, rate: float):
            self.wait, self.rate = wait, rate

    @app.exception_handler(_RateLimited)
    async def _rl(request, exc: _RateLimited):
        return JSONResponse(
            {
                "error": f"rate limited: max {exc.rate:g} calls/s per agent; "
                f"retry in {exc.wait:.1f}s",
                "retry_after": round(exc.wait, 2),
            },
            status_code=429,
            headers={"Retry-After": str(max(1, round(exc.wait)))},
        )

    def req_id(request: Request) -> str:
        return request.headers.get("x-request-id") or uuid.uuid4().hex[:16]

    # ------------------------------------------------------------------ target

    @app.get("/target/info")
    def target_info(request: Request):
        reader(request)
        import numpy as np

        flat = world.target.reshape(-1, 3)
        key = (
            (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
        )
        u, c = np.unique(key, return_counts=True)
        top = np.argsort(c)[::-1][:16]
        return {
            "width": world.w,
            "height": world.h,
            "background": hexc(world.bg),
            "blank": hexc(world.blank),
            "colours": len(u),
            "top_colours": [[f"{int(u[i]):06x}", int(c[i])] for i in top],
            "work_pixels": world.work_total,
            "background_is_work": cfg.background_is_work,
            "partial_alpha_pixels": (
                0 if world.alpha is None else int(((world.alpha > 0) & (world.alpha < 255)).sum())
            ),
            "max_pixels_per_call": cfg.max_pixels_per_call,
            "max_claim": cfg.max_claim,
        }

    @app.get("/target/region")
    def target_region(
        request: Request, x: int = 0, y: int = 0, w: int = 16, h: int = 16, format: str = "json"
    ):
        reader(request)
        return _region_response(world, world.region(world.target, x, y, w, h), x, y, format)

    @app.get("/target/pixel")
    def target_pixel(request: Request, x: int, y: int):
        reader(request)
        t = world.region(world.target, x, y, 1, 1)[0, 0]
        c = world.region(world.canvas, x, y, 1, 1)[0, 0]
        return {
            "x": x,
            "y": y,
            "hex": hexc(t),
            "r": int(t[0]),
            "g": int(t[1]),
            "b": int(t[2]),
            "current": hexc(c),
            "ok": bool(world.match[y * world.w + x]),
        }

    # ------------------------------------------------------------------ canvas

    @app.get("/region")
    def region(
        request: Request, x: int = 0, y: int = 0, w: int = 16, h: int = 16, format: str = "json"
    ):
        reader(request)
        return _region_response(world, world.region(world.canvas, x, y, w, h), x, y, format)

    @app.get("/canvas.png")
    def canvas_png(request: Request):
        reader(request)
        with world.lock:
            arr = world.canvas.copy()
        return Response(
            png_bytes(arr), media_type="image/png", headers={"Cache-Control": "no-store"}
        )

    @app.get("/target.png")
    def target_png(request: Request):
        reader(request)
        return Response(png_bytes(world.target), media_type="image/png")

    # ------------------------------------------------------------------ writes

    @app.post("/claim")
    def claim(request: Request, body: ClaimBody | None = None):
        agent = writer(request)
        return world.claim(agent, body.n if body else None)

    @app.post("/release")
    def release(request: Request, body: ReleaseBody | None = None):
        agent = writer(request)
        pts = None if body is None or body.pixels is None else [(p.x, p.y) for p in body.pixels]
        return {"released": world.release(agent, pts)}

    @app.post("/paint")
    def paint(request: Request, body: PaintBody):
        agent = writer(request)
        px = []
        for p in body.pixels:
            if p.hex is not None:
                try:
                    r, g, b = parse_hex(p.hex)
                except ValueError:
                    raise PaintError(f"pixel ({p.x},{p.y}) bad hex {p.hex!r}: want RRGGBB")
            elif None in (p.r, p.g, p.b):
                raise PaintError(f"pixel ({p.x},{p.y}) needs r,g,b or hex")
            else:
                r, g, b = p.r, p.g, p.b
            px.append((p.x, p.y, r, g, b))
        turn = body.turn or request.headers.get("x-flock-turn") or None
        rid = req_id(request)
        out = world.paint(agent, px, turn=turn, req=rid)
        out["agent"] = agent
        return out

    # ----------------------------------------------------------- region leases

    @app.post("/claim/region")
    def claim_region(request: Request, body: RegionBody):
        agent = writer(request)
        return world.claim_region(agent, body.x, body.y, body.w, body.h, body.ttl)

    @app.post("/claim/region/grant")
    def grant_region(request: Request, body: GrantBody):
        agent = writer(request)
        if not NAME_RE.match(body.agent):
            raise PaintError("bad agent name: use 1-64 chars of [A-Za-z0-9._:@-]")
        return {"regions": world.grant_region(agent, body.agent, body.id)}

    @app.post("/release/region")
    def release_region(request: Request, body: RegionRef | None = None):
        agent = writer(request)
        return {"released": world.release_region(agent, body.id if body else None)}

    @app.get("/regions")
    def regions(request: Request):
        reader(request)
        return {"regions": world.list_regions(), "enforce": cfg.region_enforce}

    @app.post("/agents/register")
    def register(request: Request, body: RegisterBody):
        tok = auth.register(request.headers, body.name)
        return {"name": body.name, "token": tok}

    # ----------------------------------------------------------- score/status

    @app.get("/score")
    def score(request: Request, agents: bool = True):
        reader(request)
        return world.score(agents=agents)

    @app.get("/status")
    def status(request: Request):
        reader(request)
        out = world.status()
        out["auth"] = {
            "shared": bool(cfg.token),
            "per_agent": bool(cfg.admin_token),
            "open": auth.open,
            "read_auth": cfg.read_auth,
        }
        return out

    @app.get("/cli/canvas")
    def cli_source():
        """The CLI itself, so a worker without the repo can fetch it (stdlib only)."""
        return FileResponse(HERE / "cli" / "canvas_cli.py", media_type="text/x-python")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    # ------------------------------------------------------------------ log/replay

    @app.get("/log")
    def log(request: Request, after: int = 0, limit: int = Query(100, ge=1, le=5000)):
        reader(request)
        lines = world.read_log(after, limit)
        return PlainTextResponse(
            "\n".join(lines) + ("\n" if lines else ""), media_type="application/x-ndjson"
        )

    @app.get("/replay")
    def replay_page():
        return FileResponse(REPLAY_DIR / "index.html", media_type="text/html")

    @app.get("/replay/frame")
    def replay_frame(request: Request, t: int = 0):
        reader(request)
        return Response(
            png_bytes(world.frame(t)),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600" if t <= world.seq else "no-store"},
        )

    @app.get("/")
    def index():
        return {"service": "flock pixel canvas", "replay": "/replay", "docs": "/docs"}

    return app
