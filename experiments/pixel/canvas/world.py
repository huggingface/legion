"""The canvas world: target, live canvas, claims, paint log, scores, replay frames.

One process, one lock. Every mutation (claim, release, paint) takes ``World.lock``; a paint
call touches at most ``max_pixels_per_call`` pixels, so the critical section is tiny. The
append-only paint log (``paint.log.jsonl`` in the data dir) is the source of truth: on start
the canvas and all counters are rebuilt by folding over it.

Per-pixel outcomes (evaluated in commit order, against the canvas *before* this pixel):
  useful       colour == target and the pixel did not match before
  duplicate    colour == target and the pixel already matched (no-op)
  wrong        colour != target and the pixel did not match before
  overwritten  colour != target and the pixel matched before (destroyed correct work)
  refused      enforced claim mode only: pixel leased by another agent; not applied
"""

from __future__ import annotations

import io
import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings, parse_hex

OUTCOMES = ("useful", "duplicate", "wrong", "overwritten", "refused")
LOG_NAME = "paint.log.jsonl"
SNAP_NAME = "claims.npz"


class PaintError(ValueError):
    """Client error with a short, actionable message."""


class RegionConflict(PaintError):
    """409: the region is held by someone else."""


def load_target(path: Path, background: tuple[int, int, int]) -> np.ndarray:
    """Load the target as uint8 HxWx3. Pixels with alpha == 0 become ``background``.

    Partially transparent pixels are composited over ``background`` so the target is exact RGB.
    """
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P", "PA"):
        im = im.convert("RGBA")
        a = np.asarray(im).astype(np.float32)
        alpha = a[..., 3:4] / 255.0
        bg = np.array(background, dtype=np.float32)
        rgb = np.rint(a[..., :3] * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        return np.ascontiguousarray(rgb)
    return np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))


def hexc(rgb) -> str:
    return f"{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


class _Grow:
    """Append-only numpy buffer (amortised doubling). Written rows are never mutated."""

    def __init__(self, dtype, width: int | None = None):
        shape = (1024,) if width is None else (1024, width)
        self.buf = np.zeros(shape, dtype=dtype)
        self.n = 0

    def extend(self, rows) -> None:
        rows = np.asarray(rows, dtype=self.buf.dtype)
        k = len(rows)
        if self.n + k > len(self.buf):
            new = np.zeros(
                (max(2 * len(self.buf), self.n + k),) + self.buf.shape[1:], dtype=self.buf.dtype
            )
            new[: self.n] = self.buf[: self.n]
            self.buf = new
        self.buf[self.n : self.n + k] = rows
        self.n += k

    def view(self) -> np.ndarray:
        return self.buf[: self.n]


@dataclass
class AgentStats:
    calls: int = 0
    claims: int = 0
    pixels: int = 0
    useful: int = 0
    duplicate: int = 0
    wrong: int = 0
    overwritten: int = 0
    refused: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0


class _Bucket:
    __slots__ = ("t", "tokens")

    def __init__(self, burst: float):
        self.tokens = burst
        self.t = time.monotonic()


class World:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.bg = parse_hex(cfg.background)
        self.blank = parse_hex(cfg.blank) if cfg.blank else self.bg
        self.target = load_target(cfg.target, self.bg)
        self.h, self.w = self.target.shape[:2]
        self.n = self.h * self.w
        self._tflat = self.target.reshape(-1, 3)
        self._blank_frame = np.empty_like(self.target)
        self._blank_frame[:] = self.blank
        self.work_mask = ~(self.target == np.array(self.blank, np.uint8)).all(-1).reshape(-1)
        self.work_total = int(self.work_mask.sum())
        self.started = time.time()
        self._reset_state()
        self._buckets: dict[str, _Bucket] = {}
        self._rate_lock = threading.Lock()
        self._req_times: dict[int, int] = {}
        self._kf: OrderedDict[int, np.ndarray] = OrderedDict()
        self._kf_lock = threading.Lock()
        self._log_fh = None
        if cfg.data_dir:
            cfg.data_dir.mkdir(parents=True, exist_ok=True)
            self._load()
            self._log_fh = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115

    # ------------------------------------------------------------------ state

    def _reset_state(self) -> None:
        self.canvas = self._blank_frame.copy()
        self._cflat = self.canvas.reshape(-1, 3)
        self.match = (self._cflat == self._tflat).all(-1)
        self.matched = int(self.match.sum())
        self.painted = np.zeros(self.n, dtype=bool)
        diff = self.canvas.astype(np.int64) - self.target.astype(np.int64)
        self.sse = int((diff * diff).sum())
        self.lease_until = np.zeros(self.n, dtype=np.float64)
        self.lease_owner = np.full(self.n, -1, dtype=np.int32)
        self.agent_ids: dict[str, int] = {}
        self.agent_names: list[str] = []
        self.agent_leases: dict[int, list[int]] = {}
        self.stats: dict[str, AgentStats] = {}
        self.totals = {k: 0 for k in OUTCOMES}
        self.cursor = 0
        # Region leases (leader -> children). region_map holds the lease id per pixel (-1 none).
        self.regions: dict[int, dict] = {}
        self.region_map = np.full(self.n, -1, dtype=np.int32)
        self.next_region = 1
        self.seq = 0
        self.first_ts: float | None = None
        self.last_ts: float | None = None
        # Columnar replay index: applied paints in commit order, and per-entry end offsets.
        self.r_idx = _Grow(np.int32)
        self.r_rgb = _Grow(np.uint8, 3)
        self.r_end = _Grow(np.int64)
        self.r_end.extend([0])  # entry 0 = blank canvas
        self.r_ts = _Grow(np.float64)
        self.r_ts.extend([0.0])
        self._offsets: list[tuple[int, int]] = []  # (seq before entry, byte offset) per 1000
        self._log_bytes = 0
        self._memlog: list[tuple[int, str]] = []

    @property
    def log_path(self) -> Path:
        assert self.cfg.data_dir is not None
        return self.cfg.data_dir / LOG_NAME

    def _aid(self, name: str) -> int:
        aid = self.agent_ids.get(name)
        if aid is None:
            aid = len(self.agent_names)
            self.agent_ids[name] = aid
            self.agent_names.append(name)
        return aid

    def _stat(self, name: str, ts: float) -> AgentStats:
        st = self.stats.get(name)
        if st is None:
            st = self.stats[name] = AgentStats(first_seen=ts)
        st.last_seen = ts
        return st

    # ------------------------------------------------------------ rate limit

    def check_rate(self, agent: str) -> float:
        """Token bucket per agent. Returns 0 if allowed, else seconds to wait."""
        rate, burst = self.cfg.rate_limit, self.cfg.rate_burst
        now = time.monotonic()
        sec = int(now)
        with self._rate_lock:
            self._req_times[sec] = self._req_times.get(sec, 0) + 1
            if len(self._req_times) > 120:
                for k in [k for k in self._req_times if k < sec - 60]:
                    del self._req_times[k]
            if rate <= 0:
                return 0.0
            b = self._buckets.get(agent)
            if b is None:
                b = self._buckets[agent] = _Bucket(burst)
            b.tokens = min(burst, b.tokens + (now - b.t) * rate)
            b.t = now
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return 0.0
            return (1.0 - b.tokens) / rate

    def req_per_s(self, window: int = 10) -> float:
        now = int(time.monotonic())
        with self._rate_lock:
            total = sum(v for k, v in self._req_times.items() if now - window <= k < now)
        return total / window

    # ----------------------------------------------------------------- claims

    def _release_ids(self, aid: int, ids) -> int:
        ids = np.asarray(list(ids), dtype=np.int64)
        if not len(ids):
            return 0
        own = self.lease_owner[ids] == aid
        ids = ids[own]
        self.lease_until[ids] = 0.0
        self.lease_owner[ids] = -1
        return len(ids)

    def claim(self, agent: str, n: int | None = None, now: float | None = None) -> dict:
        n = self.cfg.max_claim if n is None else int(n)
        if n < 1:
            raise PaintError("n must be >= 1")
        n = min(n, self.cfg.max_claim)
        now = time.time() if now is None else now
        with self.lock:
            aid = self._aid(agent)
            # One lease set per agent: a new claim drops the previous one (no hoarding).
            self._release_ids(aid, self.agent_leases.pop(aid, []))
            blocked = self._blocked_regions(agent, now)
            got: list[np.ndarray] = []
            need = n
            chunk = 1 << 16
            start = self.cursor
            scanned = 0
            while need > 0 and scanned < self.n:
                lo = (start + scanned) % self.n
                hi = min(lo + chunk, self.n, lo + self.n - scanned)
                free = ~self.match[lo:hi] & (self.lease_until[lo:hi] <= now)
                if len(blocked):
                    free &= ~np.isin(self.region_map[lo:hi], blocked)
                hits = np.flatnonzero(free)[:need] + lo
                if len(hits):
                    got.append(hits)
                    need -= len(hits)
                    self.cursor = int(hits[-1] + 1) % self.n
                scanned += hi - lo
            ids = np.concatenate(got) if got else np.zeros(0, dtype=np.int64)
            until = now + self.cfg.claim_ttl
            self.lease_until[ids] = until
            self.lease_owner[ids] = aid
            self.agent_leases[aid] = ids.tolist()
            st = self._stat(agent, now)
            st.claims += 1
            remaining = self.n - self.matched
        pixels = []
        for i in ids.tolist():
            p = {"x": i % self.w, "y": i // self.w}
            if self.cfg.reveal_target_in_claim:
                p["hex"] = hexc(self._tflat[i])
            pixels.append(p)
        return {
            "pixels": pixels,
            "runs": _runs(ids, self.w),
            "lease_s": self.cfg.claim_ttl,
            "expires": until,
            "remaining": remaining,
            "done": remaining == 0,
        }

    def release(self, agent: str, pixels: list[tuple[int, int]] | None = None) -> int:
        with self.lock:
            aid = self._aid(agent)
            if pixels is None:
                ids = self.agent_leases.pop(aid, [])
            else:
                ids = [y * self.w + x for x, y in pixels if 0 <= x < self.w and 0 <= y < self.h]
            return self._release_ids(aid, ids)

    # ---------------------------------------------------------- region leases

    def _paint_region(self, r: dict, rid: int) -> None:
        m = self.region_map.reshape(self.h, self.w)
        m[r["y"] : r["y"] + r["h"], r["x"] : r["x"] + r["w"]] = rid

    def _expire_regions(self, now: float) -> None:
        for rid in [k for k, r in self.regions.items() if r["until"] <= now]:
            self._paint_region(self.regions.pop(rid), -1)

    def _blocked_regions(self, agent: str, now: float) -> list[int]:
        """Region ids this agent may not paint or pixel-claim in (REGION_ENFORCE=1 only)."""
        if not self.cfg.region_enforce or not self.regions:
            return []
        self._expire_regions(now)
        return [
            k for k, r in self.regions.items() if r["owner"] != agent and agent not in r["grants"]
        ]

    def claim_region(
        self,
        agent: str,
        x: int,
        y: int,
        w: int,
        h: int,
        ttl: float | None = None,
        now: float | None = None,
    ) -> dict:
        if w < 1 or h < 1 or x < 0 or y < 0 or x + w > self.w or y + h > self.h:
            raise PaintError(f"region ({x},{y},{w}x{h}) outside canvas {self.w}x{self.h}")
        ttl = self.cfg.region_ttl if ttl is None else float(ttl)
        if not 0 < ttl <= self.cfg.region_ttl_max:
            raise PaintError(f"ttl must be in (0, {self.cfg.region_ttl_max:g}] seconds")
        now = time.time() if now is None else now
        with self.lock:
            self._expire_regions(now)
            for r in self.regions.values():
                if (
                    x < r["x"] + r["w"]
                    and r["x"] < x + w
                    and y < r["y"] + r["h"]
                    and r["y"] < y + h
                ):
                    raise RegionConflict(
                        f"region overlaps lease {r['id']} held by {r['owner']} "
                        f"({r['x']},{r['y']},{r['w']}x{r['h']}); pick another area or wait"
                    )
            rid = self.next_region
            self.next_region += 1
            r = {
                "id": rid,
                "owner": agent,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "until": now + ttl,
                "grants": [],
            }
            self.regions[rid] = r
            self._paint_region(r, rid)
            self._stat(agent, now)
            return dict(r)

    def _owned(self, agent: str, rid: int | None) -> list[dict]:
        mine = [r for r in self.regions.values() if r["owner"] == agent]
        if rid is None:
            return mine
        r = self.regions.get(rid)
        if r is None:
            raise PaintError(f"no live region lease {rid}")
        if r["owner"] != agent:
            raise RegionConflict(f"region {rid} is held by {r['owner']}, not you")
        return [r]

    def grant_region(
        self, agent: str, grantee: str, rid: int | None = None, now: float | None = None
    ) -> list[dict]:
        now = time.time() if now is None else now
        with self.lock:
            self._expire_regions(now)
            rs = self._owned(agent, rid)
            if not rs:
                raise PaintError("you hold no region lease; POST /claim/region first")
            for r in rs:
                if grantee not in r["grants"]:
                    r["grants"].append(grantee)
            return [dict(r) for r in rs]

    def release_region(self, agent: str, rid: int | None = None, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self.lock:
            self._expire_regions(now)
            rs = self._owned(agent, rid)
            for r in rs:
                self._paint_region(self.regions.pop(r["id"]), -1)
            return len(rs)

    def list_regions(self, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        with self.lock:
            self._expire_regions(now)
            return [dict(r) for r in self.regions.values()]

    def claims_outstanding(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        return int((self.lease_until > now).sum())

    # ------------------------------------------------------------------ paint

    def validate(self, pixels: list[tuple[int, int, int, int, int]]) -> None:
        cap = self.cfg.max_pixels_per_call
        if not pixels:
            raise PaintError("no pixels given")
        if len(pixels) > cap:
            raise PaintError(
                f"too many pixels: {len(pixels)} > {cap} per call; split into calls of {cap}"
            )
        for x, y, r, g, b in pixels:
            if not (0 <= x < self.w and 0 <= y < self.h):
                raise PaintError(f"pixel ({x},{y}) out of bounds: canvas is {self.w}x{self.h}")
            if not all(0 <= c <= 255 for c in (r, g, b)):
                raise PaintError(f"pixel ({x},{y}) colour out of range: r,g,b must be 0..255")

    def paint(
        self,
        agent: str,
        pixels: list[tuple[int, int, int, int, int]],
        turn: str | None = None,
        req: str | None = None,
        now: float | None = None,
    ) -> dict:
        self.validate(pixels)
        now = time.time() if now is None else now
        with self.lock:
            entry = self._apply(agent, pixels, turn, req, now, enforce=True)
            self._write_log(entry)
            score = self._score_brief()
        return {
            "seq": entry["seq"],
            "req": req,
            "results": [self._result(p) for p in entry["px"]],
            "counts": _count(entry["px"]),
            "score": score,
        }

    def _result(self, p) -> dict:
        r = {"x": p[0], "y": p[1], "outcome": p[3]}
        if p[3] in ("wrong", "overwritten"):
            r["want"] = hexc(self._tflat[p[1] * self.w + p[0]])
        return r

    def _apply(self, agent, pixels, turn, req, ts, enforce: bool, outcomes=None) -> dict:
        """Apply one paint call under the lock. When ``outcomes`` is given (log replay) the
        recorded outcomes are trusted for 'refused' so replay is deterministic."""
        ts = round(ts, 4)  # as written to the log, so a rebuild is bit-identical
        aid = self._aid(agent)
        st = self._stat(agent, ts)
        st.calls += 1
        enforced = enforce and self.cfg.claim_mode == "enforced"
        blocked = self._blocked_regions(agent, ts) if enforce else []
        px_out = []
        applied_idx = []
        applied_rgb = []
        for k, (x, y, r, g, b) in enumerate(pixels):
            i = y * self.w + x
            if outcomes is not None:
                refused = outcomes[k] == "refused"
            else:
                refused = (
                    enforced and self.lease_owner[i] not in (-1, aid) and self.lease_until[i] > ts
                ) or (len(blocked) > 0 and int(self.region_map[i]) in blocked)
            if refused:
                out = "refused"
            else:
                t = self._tflat[i]
                before = self._cflat[i]
                was = bool(self.match[i])
                ok = int(t[0]) == r and int(t[1]) == g and int(t[2]) == b
                if ok:
                    out = "duplicate" if was else "useful"
                else:
                    out = "overwritten" if was else "wrong"
                old = (
                    (int(before[0]) - int(t[0])) ** 2
                    + (int(before[1]) - int(t[1])) ** 2
                    + (int(before[2]) - int(t[2])) ** 2
                )
                new = (r - int(t[0])) ** 2 + (g - int(t[1])) ** 2 + (b - int(t[2])) ** 2
                self.sse += new - old
                self._cflat[i] = (r, g, b)
                self.match[i] = ok
                self.matched += int(ok) - int(was)
                self.painted[i] = True
                if ok or self.lease_owner[i] == aid:
                    self.lease_owner[i] = -1
                    self.lease_until[i] = 0.0
                applied_idx.append(i)
                applied_rgb.append((r, g, b))
            px_out.append([x, y, f"{r:02x}{g:02x}{b:02x}", out])
            self.totals[out] += 1
            setattr(st, out, getattr(st, out) + 1)
            st.pixels += 1
        self.seq += 1
        if applied_idx:
            self.r_idx.extend(applied_idx)
            self.r_rgb.extend(applied_rgb)
        self.r_end.extend([self.r_idx.n])
        self.r_ts.extend([ts])
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts
        return {
            "seq": self.seq,
            "ts": ts,
            "agent": agent,
            "turn": turn,
            "req": req,
            "px": px_out,
        }

    # -------------------------------------------------------------- the log

    def _write_log(self, entry: dict) -> None:
        line = json.dumps(entry, separators=(",", ":"))
        if self._log_fh is None:
            self._memlog.append((entry["seq"], line))
            return
        if entry["seq"] % 1000 == 1:
            self._offsets.append((entry["seq"] - 1, self._log_bytes))
        data = line + "\n"
        self._log_fh.write(data)
        self._log_fh.flush()
        self._log_bytes += len(data.encode())

    def read_log(self, after: int = 0, limit: int = 100) -> list[str]:
        """Return raw JSONL lines with seq > after (from disk, or rebuilt if memory-only)."""
        limit = max(1, min(int(limit), 5000))
        if self.cfg.data_dir is None:
            return self._mem_log(after, limit)
        out: list[str] = []
        with open(self.log_path, encoding="utf-8") as fh:
            off = self._seek_hint(after)
            fh.seek(off)
            for line in fh:
                if not line.endswith("\n"):
                    break
                seq = _seq_of(line)
                if seq > after:
                    out.append(line.rstrip("\n"))
                    if len(out) >= limit:
                        break
        return out

    def _seek_hint(self, after: int) -> int:
        # Byte offsets of every 1000th entry let /log skip ahead without scanning the file.
        best = 0
        for s, off in self._offsets:
            if s <= after:
                best = off
            else:
                break
        return best

    def _mem_log(self, after: int, limit: int) -> list[str]:
        # Memory-only mode (tests / no CANVAS_DATA): keep a bounded in-memory copy.
        return [ln for s, ln in self._memlog if s > after][:limit]

    # ------------------------------------------------------------- persistence

    def _load(self) -> None:
        path = self.log_path
        if path.exists():
            good = 0
            with open(path, "rb") as fh:
                for raw in fh:
                    if not raw.endswith(b"\n"):
                        break  # torn last line from a crash: drop it
                    try:
                        e = json.loads(raw)
                    except json.JSONDecodeError:
                        break
                    if e["seq"] % 1000 == 1:
                        self._offsets.append((e["seq"] - 1, good))
                    px = [(p[0], p[1], *parse_hex(p[2])) for p in e["px"]]
                    self._apply(
                        e["agent"],
                        px,
                        e.get("turn"),
                        e.get("req"),
                        e["ts"],
                        enforce=False,
                        outcomes=[p[3] for p in e["px"]],
                    )
                    good += len(raw)
            if good != path.stat().st_size:
                with open(path, "r+b") as fh:
                    fh.truncate(good)
            self._log_bytes = good
        snap = self.cfg.data_dir / SNAP_NAME
        if snap.exists():
            try:
                z = np.load(snap, allow_pickle=False)
                if int(z["seq"]) == self.seq and len(z["lease_until"]) == self.n:
                    names = json.loads(str(z["names"]))
                    for nm in names:
                        self._aid(nm)
                    remap = np.array([self.agent_ids[nm] for nm in names] + [-1], np.int32)
                    owner = z["lease_owner"]
                    self.lease_owner = remap[np.where(owner < 0, len(names), owner)]
                    self.lease_until = z["lease_until"].copy()
                    self.cursor = int(z["cursor"])
                    for r in json.loads(str(z["regions"])) if "regions" in z else []:
                        self.regions[r["id"]] = r
                        self._paint_region(r, r["id"])
                        self.next_region = max(self.next_region, r["id"] + 1)
                    for i in np.flatnonzero(self.lease_owner >= 0).tolist():
                        self.agent_leases.setdefault(int(self.lease_owner[i]), []).append(i)
            except Exception as e:  # noqa: BLE001 - claims are soft state; log is the truth
                print(f"canvas: ignoring unreadable claims snapshot: {e}", flush=True)

    def snapshot(self) -> None:
        """Write claims + a canvas PNG (convenience). Canvas truth is the log."""
        if self.cfg.data_dir is None:
            return
        with self.lock:
            if self._log_fh is not None:
                self._log_fh.flush()
                os.fsync(self._log_fh.fileno())
            payload = {
                "seq": np.int64(self.seq),
                "cursor": np.int64(self.cursor),
                "lease_until": self.lease_until.copy(),
                "lease_owner": self.lease_owner.copy(),
                "names": np.array(json.dumps(self.agent_names)),
                "regions": np.array(json.dumps(list(self.regions.values()))),
            }
            canvas = self.canvas.copy()
        tmp = self.cfg.data_dir / (SNAP_NAME + ".tmp.npz")
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, self.cfg.data_dir / SNAP_NAME)
        (self.cfg.data_dir / "canvas.png.tmp").write_bytes(png_bytes(canvas))
        os.replace(self.cfg.data_dir / "canvas.png.tmp", self.cfg.data_dir / "canvas.png")

    def close(self) -> None:
        self.snapshot()
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    # ------------------------------------------------------------------ scores

    def _score_brief(self) -> dict:
        return {
            "exact_match_pct": round(100.0 * self.matched / self.n, 4),
            "todo": self.n - self.matched,
        }

    def score(self, agents: bool = True) -> dict:
        with self.lock:
            mse = self.sse / (self.n * 3)
            work_done = int((self.match & self.work_mask).sum())
            out = {
                "width": self.w,
                "height": self.h,
                "seq": self.seq,
                "exact_match_pct": round(100.0 * self.matched / self.n, 4),
                "coverage_pct": round(100.0 * work_done / max(1, self.work_total), 4),
                "painted_pct": round(100.0 * int(self.painted.sum()) / self.n, 4),
                "matched": self.matched,
                "todo": self.n - self.matched,
                "work_total": self.work_total,
                "work_done": work_done,
                "mse": round(mse, 4),
                "psnr": round(10 * math.log10(255.0**2 / mse), 3) if mse > 0 else None,
                "paints": dict(self.totals),
            }
            if agents:
                out["agents"] = {k: asdict(v) for k, v in self.stats.items()}
        return out

    def status(self) -> dict:
        now = time.time()
        with self.lock:
            active = sum(1 for s in self.stats.values() if now - s.last_seen < 60)
            out = {
                "uptime_s": round(now - self.started, 1),
                "seq": self.seq,
                "agents_seen": len(self.stats),
                "agents_active_60s": active,
                "claims_outstanding": self.claims_outstanding(now),
                "region_leases": len(self.regions),
                "todo": self.n - self.matched,
                "exact_match_pct": round(100.0 * self.matched / self.n, 4),
                "first_ts": self.first_ts,
                "last_ts": self.last_ts,
            }
        out["req_per_s_10s"] = self.req_per_s()
        out["config"] = {
            "max_pixels_per_call": self.cfg.max_pixels_per_call,
            "max_claim": self.cfg.max_claim,
            "claim_ttl_s": self.cfg.claim_ttl,
            "claim_mode": self.cfg.claim_mode,
            "rate_limit_per_s": self.cfg.rate_limit,
            "keyframe_every": self.cfg.keyframe_every,
            "region_enforce": self.cfg.region_enforce,
        }
        return out

    # ------------------------------------------------------------------ replay

    def frame(self, t: int) -> np.ndarray:
        """Canvas state after log entry ``t`` (0 = blank), folded from the replay index.

        Keyframes every ``keyframe_every`` entries are cached (LRU) so a scrubber touching
        nearby positions only re-applies the delta since the previous keyframe.
        """
        with self.lock:
            t = max(0, min(int(t), self.seq))
            ends = self.r_end.view()
            idx = self.r_idx.view()
            rgb = self.r_rgb.view()
        kfe = max(1, self.cfg.keyframe_every)
        k = (t // kfe) * kfe
        with self._kf_lock:
            base = self._keyframe(k, ends, idx, rgb)
        frame = base.copy()
        _fold(frame, idx, rgb, int(ends[k]), int(ends[t]))
        return frame

    def _keyframe(self, k: int, ends, idx, rgb) -> np.ndarray:
        if k == 0:
            return self._blank_frame
        hit = self._kf.get(k)
        if hit is not None:
            self._kf.move_to_end(k)
            return hit
        prev = max((c for c in list(self._kf) if c < k), default=0)
        base = self._blank_frame if prev == 0 else self._kf[prev]
        frame = base.copy()
        _fold(frame, idx, rgb, int(ends[prev]), int(ends[k]))
        self._kf[k] = frame
        while len(self._kf) > max(1, self.cfg.keyframe_cache):
            self._kf.popitem(last=False)
        return frame

    def region(self, src: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        if w < 1 or h < 1 or x < 0 or y < 0 or x + w > self.w or y + h > self.h:
            raise PaintError(f"region ({x},{y},{w}x{h}) outside canvas {self.w}x{self.h}")
        with self.lock:
            return src[y : y + h, x : x + w].copy()


def _fold(frame: np.ndarray, idx, rgb, lo: int, hi: int) -> None:
    """Apply paints [lo, hi) to ``frame``; for repeated pixels the last write wins."""
    if hi <= lo:
        return
    seg = idx[lo:hi]
    rev = seg[::-1]
    u, first = np.unique(rev, return_index=True)
    frame.reshape(-1, 3)[u] = rgb[lo:hi][len(seg) - 1 - first]


def _runs(ids: np.ndarray, w: int) -> list[dict]:
    runs: list[dict] = []
    for i in ids.tolist():
        x, y = i % w, i // w
        if runs and runs[-1]["y"] == y and runs[-1]["x"] + runs[-1]["len"] == x:
            runs[-1]["len"] += 1
        else:
            runs.append({"x": x, "y": y, "len": 1})
    return runs


def _count(px) -> dict:
    c: dict[str, int] = {}
    for p in px:
        c[p[3]] = c.get(p[3], 0) + 1
    return c


def _seq_of(line: str) -> int:
    # Entries start with {"seq":N, so avoid a full json parse when skipping.
    j = line.index(",")
    return int(line[7:j])
