"""Bucket sync: per agent, `<BUCKET>/<swarm>/<type>/<agent>/` holds

    workdir/<path relative to the workdir>        (if persist_workdir)
    sessions/<agent>.jsonl, <agent>.state.json    (pi session log + worker state)
    harness/<path relative to the harness home>   (CLAUDE_CONFIG_DIR, CODEX_HOME or PI_CODING_AGENT_DIR)

The workdir goes through `sync_bucket` (the Python form of `hf buckets sync`, as in the
resume-gate experiment), upward with delete so the bucket mirrors the workdir. The handful of
session and harness files go through `batch_bucket_files`, only those whose (mtime, size)
changed since the last push by this worker. `restore` brings all three areas back to the
same absolute paths (workdir and work root are fixed, harness homes are fixed by the image)
before the first resume.

Buckets store bytes, not file modes: a restored file comes back 0644, so an executable script
would stop running after a move (pixel run 1, 2026-10-19: `./canvas: Permission denied`). Each
push therefore writes `<workdir>/.flock-modes.json` ({relative path: octal mode} for every file
with an executable bit) when it changed, and restore re-applies it after `sync_down`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import stat
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from flock.worker.agents import AgentRuntime

log = logging.getLogger(__name__)

UPLOAD_CHUNK = 200
MODES_FILE = ".flock-modes.json"


def collect_modes(workdir: Path) -> dict[str, str]:
    """{relative posix path: "0o755"} for regular files with any executable bit (symlinks and
    the manifest itself excluded)."""
    out: dict[str, str] = {}
    for root, dirs, files in os.walk(workdir):
        dirs.sort()
        for f in files:
            p = Path(root) / f
            try:
                st = p.lstat()
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode) and st.st_mode & 0o111:
                rel = p.relative_to(workdir).as_posix()
                if rel != MODES_FILE:
                    out[rel] = oct(stat.S_IMODE(st.st_mode))
    return dict(sorted(out.items()))


def write_modes(workdir: Path) -> bool:
    """(Re)write the manifest if its content changed, remove it when nothing is executable.
    Returns whether the file changed."""
    path = workdir / MODES_FILE
    modes = collect_modes(workdir)
    old = path.read_text() if path.is_file() else None
    if not modes:
        if old is not None:
            path.unlink()
            return True
        return False
    new = json.dumps(modes, indent=0, sort_keys=True) + "\n"
    if new == old:
        return False
    path.write_text(new)
    return True


def apply_modes(workdir: Path) -> int:
    """chmod the files listed in the manifest (inside the workdir, regular files only)."""
    path = workdir / MODES_FILE
    try:
        modes = json.loads(path.read_text())
    except (OSError, ValueError):
        return 0
    n = 0
    root = workdir.resolve()
    for rel, mode in (modes.items() if isinstance(modes, dict) else []):
        p = workdir / rel
        try:
            if ".." in Path(rel).parts or not p.resolve().is_relative_to(root) or p.is_symlink() \
                    or not p.is_file():
                continue
            os.chmod(p, int(mode, 8) & 0o7777)
            n += 1
        except (OSError, ValueError, TypeError):
            continue
    return n


@dataclass
class BucketStats:
    """Bucket traffic counters for the worker log: store operations (`ops`), the Hub HTTP
    requests behind them (`http`, by method and route; xet CAS transfers run in hf_xet and are
    not seen here) and HTTP error statuses (`errors`, e.g. 429). Thread-safe."""

    ops: Counter = field(default_factory=Counter)
    http: Counter = field(default_factory=Counter)
    errors: Counter = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, kind: str, key: str, n: int = 1) -> None:
        with self._lock:
            getattr(self, kind)[key] += n

    def snapshot(self) -> dict[str, Counter]:
        with self._lock:
            return {"ops": Counter(self.ops), "http": Counter(self.http),
                    "errors": Counter(self.errors)}

    def since(self, before: dict[str, Counter]) -> dict[str, dict[str, int]]:
        now = self.snapshot()
        return {k: dict(now[k] - before[k]) for k in now}


_BUCKET_ROUTE = re.compile(r"^/api/buckets/[^/]+/[^/]+(?:/([^/]+))?")


def http_route(method: str, host: str, path: str) -> str:
    """'GET /api/buckets/ns/b/tree/x/y' -> 'GET buckets/tree'; other hosts by first segment."""
    m = _BUCKET_ROUTE.match(path)
    if m:
        return f"{method} buckets/{m.group(1) or ''}".rstrip("/")
    first = path.strip("/").split("/", 1)[0]
    if host.endswith("huggingface.co"):
        return f"{method} {first or '/'}"
    return f"{method} {host}/{first}"


def count_hub_http(stats: BucketStats) -> None:
    """Count every request huggingface_hub's shared client makes (one install per process)."""
    import huggingface_hub
    from huggingface_hub.utils._http import default_client_factory

    def on_request(request) -> None:
        stats.add("http", http_route(request.method, request.url.host, request.url.path))

    def on_response(response) -> None:
        if response.status_code >= 400:
            req = response.request
            route = http_route(req.method, req.url.host, req.url.path)
            stats.add("errors", f"{response.status_code} {route}")

    def factory():
        client = default_client_factory()
        client.event_hooks["request"].append(on_request)
        client.event_hooks.setdefault("response", []).append(on_response)
        return client

    huggingface_hub.set_client_factory(factory)


class BucketStore(Protocol):
    """Blocking bucket operations on paths relative to the flock root."""

    def upload(self, files: list[tuple[Path, str]]) -> None: ...
    def list(self, prefix: str) -> list[str]: ...
    def download(self, files: list[tuple[str, Path]]) -> None: ...
    def sync_up(self, local: Path, prefix: str) -> None:
        """Mirror a local directory to `prefix` (delete remote files missing locally)."""
        ...

    def sync_down(self, prefix: str, local: Path) -> None: ...


def parse_bucket(url: str) -> tuple[str, str]:
    """'hf://buckets/ns/name/some/prefix' -> ('ns/name', 'some/prefix/')."""
    rest = url.removeprefix("hf://").removeprefix("buckets/").strip("/")
    parts = rest.split("/")
    if len(parts) < 2:
        raise ValueError(f"bad bucket url {url!r}; expected hf://buckets/<namespace>/<name>[/prefix]")
    prefix = "/".join(parts[2:])
    return f"{parts[0]}/{parts[1]}", (prefix + "/" if prefix else "")


class HubBucketStore:
    """huggingface_hub bucket API (batch_bucket_files / list_bucket_tree / download_bucket_files)."""

    def __init__(self, url: str, token: str | None = None, stats: BucketStats | None = None) -> None:
        from huggingface_hub import HfApi

        self.bucket_id, self.base = parse_bucket(url)
        self.api = HfApi(token=token)
        self.stats = stats or BucketStats()

    def upload(self, files: list[tuple[Path, str]]) -> None:
        for i in range(0, len(files), UPLOAD_CHUNK):
            self.stats.add("ops", "batch_bucket_files")
            chunk = files[i : i + UPLOAD_CHUNK]
            self.api.batch_bucket_files(self.bucket_id, add=[(str(p), self.base + r) for p, r in chunk])

    def list(self, prefix: str) -> list[str]:
        from huggingface_hub import BucketFile

        self.stats.add("ops", "list_bucket_tree")
        full = self.base + prefix
        out = []
        for f in self.api.list_bucket_tree(self.bucket_id, prefix=full, recursive=True):
            if isinstance(f, BucketFile) and f.path.startswith(full):
                out.append(f.path[len(self.base):])
        return out

    def download(self, files: list[tuple[str, Path]]) -> None:
        self.stats.add("ops", "download_bucket_files")
        for _, local in files:
            local.parent.mkdir(parents=True, exist_ok=True)
        self.api.download_bucket_files(self.bucket_id, [(self.base + r, str(p)) for r, p in files])

    def _url(self, prefix: str) -> str:
        return f"hf://buckets/{self.bucket_id}/{self.base}{prefix}".rstrip("/")

    def sync_up(self, local: Path, prefix: str) -> None:
        self.stats.add("ops", "sync_bucket_up")
        self.api.sync_bucket(str(local), self._url(prefix), delete=True, quiet=True)

    def sync_down(self, prefix: str, local: Path) -> None:
        self.stats.add("ops", "sync_bucket_down")
        local.mkdir(parents=True, exist_ok=True)
        self.api.sync_bucket(self._url(prefix), str(local), quiet=True)


@dataclass
class RestoreResult:
    workdir: int = 0
    sessions: int = 0
    harness: int = 0


class BucketSync:
    def __init__(self, store: BucketStore, swarm: str) -> None:
        self.store = store
        self.swarm = swarm
        self.stats: BucketStats = getattr(store, "stats", None) or BucketStats()
        self._seen: dict[Path, tuple[int, int]] = {}

    def prefix(self, rt: AgentRuntime) -> str:
        return f"{self.swarm}/{rt.type_name}/{rt.name}/"

    def collect(self, rt: AgentRuntime) -> list[tuple[Path, str]]:
        """The agent's session-log, state and harness files, with their bucket paths."""
        pre = self.prefix(rt)
        out: list[tuple[Path, str]] = []
        for p in (rt.log.path, rt.state_path):
            if p.is_file():
                out.append((p, f"{pre}sessions/{p.name}"))
        home = Path(rt.adapter.home)
        for p in rt.harness_files():
            try:
                rel = p.relative_to(home)
            except ValueError:
                log.warning("%s: harness file %s is outside %s; not synced", rt.name, p, home)
                continue
            out.append((p, f"{pre}harness/{rel.as_posix()}"))
        return out

    def _changed(self, files: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
        changed = []
        for p, r in files:
            try:
                st = p.stat()
            except OSError:
                continue
            sig = (st.st_mtime_ns, st.st_size)
            if self._seen.get(p) != sig:
                changed.append((p, r))
        return changed

    def _mark(self, files: list[tuple[Path, str]]) -> None:
        for p, _ in files:
            try:
                st = p.stat()
                self._seen[p] = (st.st_mtime_ns, st.st_size)
            except OSError:
                pass

    async def push(self, rt: AgentRuntime) -> int:
        try:
            return await self._push(rt)
        except Exception:
            self.stats.add("errors", "push_failed")
            raise

    async def _push(self, rt: AgentRuntime) -> int:
        if rt.spec.persist_workdir and rt.workdir.is_dir():
            await asyncio.to_thread(write_modes, rt.workdir)
            await asyncio.to_thread(self.store.sync_up, rt.workdir, self.prefix(rt) + "workdir")
        files = self._changed(self.collect(rt))
        if files:
            await asyncio.to_thread(self.store.upload, files)
            self._mark(files)
        return len(files)

    async def restore(self, rt: AgentRuntime) -> RestoreResult:
        try:
            return await self._restore(rt)
        except Exception:
            self.stats.add("errors", "restore_failed")
            raise

    async def _restore(self, rt: AgentRuntime) -> RestoreResult:
        pre = self.prefix(rt)
        remote = await asyncio.to_thread(self.store.list, pre)
        res = RestoreResult()
        plan: list[tuple[str, Path]] = []
        home = Path(rt.adapter.home)
        for r in remote:
            rel = r[len(pre):]
            area, _, sub = rel.partition("/")
            if not sub or ".." in Path(sub).parts:
                continue
            if area == "workdir":
                res.workdir += 1
            elif area == "sessions":
                plan.append((r, rt.sessions_dir / sub))
                res.sessions += 1
            elif area == "harness":
                plan.append((r, home / sub))
                res.harness += 1
        if res.workdir and rt.spec.persist_workdir:
            await asyncio.to_thread(self.store.sync_down, pre + "workdir", rt.workdir)
            if n := await asyncio.to_thread(apply_modes, rt.workdir):
                log.info("%s: restored the mode of %d file(s)", rt.name, n)
        if plan:
            await asyncio.to_thread(self.store.download, plan)
            self._mark([(p, r) for r, p in plan])
        log.info("%s: restored %s", rt.name, res)
        return res
