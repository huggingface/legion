"""Bucket sync: per agent, `<BUCKET>/<swarm>/<type>/<agent>/` holds

    workdir/<path relative to the workdir>        (if persist_workdir)
    meta.json                                     ({files, bytes, ts} of the mirrored workdir when it last changed)
    sessions/<agent>.jsonl, <agent>.state.json    (pi session log + worker state)
    harness/<path relative to the harness home>   (CLAUDE_CONFIG_DIR, CODEX_HOME or PI_CODING_AGENT_DIR)

The workdir goes through `sync_bucket` (the Python form of `hf buckets sync`, as in the
resume-gate experiment), upward with delete so the bucket mirrors the workdir. The handful of
session and harness files go through `batch_bucket_files`, only those whose (mtime, size)
changed since the last push by this worker. `restore` brings all three areas back to the
same absolute paths (workdir and work root are fixed, harness homes are fixed by the image)
before the first resume.

Restore lists the agent's prefix once, then downloads every file it found in batches
(`restore_batch()`: env `FLOCK_RESTORE_BATCH`, default 1000, minimum 1, an invalid value
means the default), each to a `.flock-part` sibling renamed into place on success. A batch that
fails is retried file by file, with backoff (`RESTORE_BACKOFF`) between rounds: one unreadable
file is skipped and reported in `RestoreResult.failed` instead of aborting the whole restore. A
permanent "not found" (HTTP 404, e.g. xet "CAS Client Error: ... 404 Not Found" on the
reconstruction: the entry's data is gone) is skipped at once, without the backoff rounds (an
already-corrupt copy with ~42k such entries took 636 s to restore with them). Skipped files are
logged as one summary line per agent. Only a failed listing makes `restore` raise.

Buckets store bytes, not file modes: a restored file comes back 0644, so an executable script
would stop running after a move (pixel run 1, 2026-10-19: `./canvas: Permission denied`). Each
push therefore writes `<workdir>/.flock-modes.json` ({relative path: octal mode} for every file
with an executable bit) when it changed, and restore re-applies it after the workdir download.

Push safety (run 4, 2026-10-27: periodic syncs mirrored half-restored workdirs over 31 agents'
bucket copies, deleting entries the in-flight restore had not fetched yet):

- No push before setup: `AgentRuntime.sync_now` does nothing until the agent's `setup()` has
  finished its restore (every caller: the periodic sync, drain, preempt, unassign), and
  `BucketSync.push` refuses an agent whose restore is in flight. The bucket copy is the truth
  until the restore is done.
- Delete-mirror only from a full baseline: the workdir push deletes remote files missing
  locally only when this worker's restore of the agent was complete (the listing succeeded and
  no file failed; an empty prefix counts as complete) or when no restore was attempted (a new
  agent). After a failed listing, a skipped file or an interrupted restore, pushes upload
  changed files but never delete, for as long as this worker holds the agent: a file it could
  not fetch stays in the bucket for the next restore.
- `*.flock-part` (restore temporaries) is never uploaded nor deleted by a push; restore removes
  stray ones left by a crash.
- Regenerable caches (`workdir_excludes`: `__pycache__/`, `*.pyc`, `node_modules/`, `.venv/`, ...; env
  FLOCK_SYNC_EXCLUDE adds globs, FLOCK_SYNC_EXCLUDE_DEFAULTS=0 drops the defaults) are neither
  uploaded nor deleted by the mirror, and restore skips them, also those already in the bucket.
  `.git/` and worktrees are never excluded by default.
- The type's `workdir.ephemeral` dirs (root-relative) are excluded like the caches; the agent's
  setup empties them (`reset_ephemeral`), so they never come back after a move.
- Each push's modes walk also measures what is mirrored (`WorkdirStats`): above the type's
  `workdir.soft_*` the agent gets a note, above `max_*` a `workdir_over_quota` event (each at
  most hourly, `BucketSync._quota`). A push is never refused for size.
- Restore sets each workdir file's mtime to the bucket's (the listing carries it), so the first
  push after a restore uploads only real changes; session and harness files are marked as
  pushed (`_mark`).
"""

from __future__ import annotations

import asyncio
import fnmatch
import functools
import json
import logging
import os
import re
import shutil
import stat
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from flock.models import WorkdirSpec

if TYPE_CHECKING:
    from flock.models import TypeSpec
    from flock.worker.agents import AgentRuntime

log = logging.getLogger(__name__)

UPLOAD_CHUNK = 200
MODES_FILE = ".flock-modes.json"
RESTORE_BATCH = 1000  # default of FLOCK_RESTORE_BATCH
RESTORE_THREADS = 4  # default of FLOCK_RESTORE_THREADS
# The type's `worker:` values for this module (sync_exclude, restore_batch, restore_threads),
# set live by `Worker.apply_knobs`; each one present wins over its env variable.
SPEC_KNOBS: dict[str, object] = {}


def _env_int(name: str, default: int) -> int:
    try:
        n = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return n if n >= 1 else default


def restore_batch() -> int:
    """Files per restore download batch: the spec's, else env FLOCK_RESTORE_BATCH (>= 1), else
    RESTORE_BATCH."""
    return SPEC_KNOBS.get("restore_batch") or _env_int("FLOCK_RESTORE_BATCH", RESTORE_BATCH)


def restore_threads() -> int:
    """Batches of one agent's restore downloaded in parallel: the spec's, else env
    FLOCK_RESTORE_THREADS (>= 1), else RESTORE_THREADS."""
    return SPEC_KNOBS.get("restore_threads") or _env_int("FLOCK_RESTORE_THREADS", RESTORE_THREADS)


RESTORE_BACKOFF = (1.0, 4.0, 10.0)  # seconds before each per-file retry round
PART_SUFFIX = ".flock-part"
PUSH_EXCLUDE = ["*" + PART_SUFFIX]  # sync_bucket patterns, matched on relative paths
# Regenerable caches left out of the workdir mirror and of restore (module docstring). A trailing
# "/" names a directory at any depth (`workdir_excludes`); other entries are plain globs.
# Not build/, dist/ or target/: sometimes source.
DEFAULT_WORKDIR_EXCLUDE = ["__pycache__/", "*.pyc", ".pytest_cache/", ".mypy_cache/",
                           ".ruff_cache/", ".hypothesis/", "*.egg-info/", ".cache/",
                           "node_modules/", ".venv/", "venv/", ".tox/", ".nox/",
                           ".ipynb_checkpoints/", ".gradle/", ".next/", "*.o"]


def _expand(pattern: str) -> list[str]:
    """`name/` -> `name/*` and `*/name/*` (the directory at the root and below any directory);
    anything else is returned as is. `sync_bucket` globs are fnmatch on the whole relative path,
    where `*` also crosses `/`."""
    if pattern.endswith("/"):
        d = pattern.rstrip("/")
        return [f"{d}/*", f"*/{d}/*"]
    return [pattern]


def workdir_excludes() -> list[str]:
    """Globs for the workdir mirror, relative to the workdir, always including `*.flock-part`:
    the defaults (unless env FLOCK_SYNC_EXCLUDE_DEFAULTS=0) plus the spec's `sync_exclude`, else
    the comma-separated FLOCK_SYNC_EXCLUDE. `.git/` is never in the defaults."""
    pats = list(PUSH_EXCLUDE)
    if os.environ.get("FLOCK_SYNC_EXCLUDE_DEFAULTS", "1").strip() != "0":
        pats += [x for d in DEFAULT_WORKDIR_EXCLUDE for x in _expand(d)]
    extra = SPEC_KNOBS.get("sync_exclude")
    if extra is None:
        extra = os.environ.get("FLOCK_SYNC_EXCLUDE", "").split(",")
    for item in extra:
        if item.strip():
            pats += _expand(item.strip())
    return pats


@functools.lru_cache(maxsize=64)
def _matcher(patterns: tuple[str, ...]):
    """One compiled regex for a pattern list (a walk tests every file of a workdir)."""
    if not patterns:
        return lambda rel: None
    return re.compile("|".join(fnmatch.translate(p.replace("\\", "/")) for p in patterns)).match


def is_excluded(rel: str, patterns: list[str]) -> bool:
    """Does the workdir-relative posix path match one of the patterns (as `sync_bucket` does)?"""
    return _matcher(tuple(patterns))(rel) is not None


@dataclass
class WorkdirStats:
    """What a push mirrors: regular files and their bytes, excluded paths left out, and the
    top-level directories with the most files ((name, files), largest first, up to 5)."""

    files: int = 0
    bytes: int = 0
    top_dirs: list[tuple[str, int]] = field(default_factory=list)

    def summary(self) -> str:
        """'182k files, 3100 MB; largest: snaps/ 140k, src/ 2k' (`{workspace}` in loop prompts)."""
        top = ", ".join(f"{d}/ {_k(n)}" for d, n in self.top_dirs[:3])
        return (f"{_k(self.files)} files, {self.bytes / 2**20:.0f} MB"
                + (f"; largest: {top}" if top else ""))


def _k(n: int) -> str:
    return f"{n / 1000:.0f}k" if n >= 1000 else str(n)


QUOTA_NOTE_EVERY = 3600.0  # s, per agent: the over-soft-limit note and workdir_over_quota


def quota_note(ws: WorkdirStats, w: WorkdirSpec) -> str | None:
    """The note for an agent whose workdir is above its soft limit, else None."""
    if ws.files <= w.soft_files and ws.bytes / 2**20 <= w.soft_mb:
        return None
    where = f", or keep throwaway clones in {w.ephemeral[0]}" if w.ephemeral else ""
    return (f"your workspace has {ws.summary()} (soft limit {_k(w.soft_files)} files, "
            f"{w.soft_mb:.0f} MB); delete clones you no longer need{where}")


def over_hard_limit(ws: WorkdirStats, w: WorkdirSpec) -> bool:
    return (ws.files > (w.max_files or 5 * w.soft_files)
            or ws.bytes / 2**20 > (w.max_mb or 5 * w.soft_mb))


def ephemeral_excludes(spec: TypeSpec) -> list[str]:
    """The type's `workdir.ephemeral` dirs as root-anchored globs (never synced nor restored)."""
    return [d + "*" for d in (spec.workdir.ephemeral if spec.workdir else [])]


def reset_ephemeral(workdir: Path, spec: TypeSpec) -> None:
    """Empty (or create) the type's ephemeral dirs: run at every agent setup, so a move or a
    restart never brings back their content. Blocking."""
    root = workdir.resolve()
    for d in spec.workdir.ephemeral if spec.workdir else []:
        p = workdir / d
        try:
            if not p.parent.resolve().is_relative_to(root):  # a symlinked parent: leave it
                log.warning("ephemeral dir %s is outside %s; not touched", p, root)
                continue
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            p.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.warning("ephemeral dir %s: %s", p, e)


def collect_modes(workdir: Path, stats: WorkdirStats | None = None,
                  exclude: list[str] | None = None) -> dict[str, str]:
    """{relative posix path: "0o755"} for regular files with any executable bit (symlinks and
    the manifest itself excluded). `exclude`: paths left out of the walk (excluded directories
    are not entered). `stats` gets the walk's counts (`WorkdirStats`)."""
    out: dict[str, str] = {}
    excl = _matcher(tuple(exclude or ()))
    per_dir: Counter[str] = Counter()
    for root, dirs, files in os.walk(workdir):
        base = Path(root).relative_to(workdir).as_posix()
        base = "" if base == "." else base + "/"
        dirs[:] = sorted(d for d in dirs if excl(f"{base}{d}/") is None)
        for f in files:
            rel = base + f
            if excl(rel) is not None:
                continue
            try:
                st = os.lstat(os.path.join(root, f))
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if stats is not None:
                stats.files += 1
                stats.bytes += st.st_size
                if base:
                    per_dir[base.split("/", 1)[0]] += 1
            if st.st_mode & 0o111 and rel != MODES_FILE:
                out[rel] = oct(stat.S_IMODE(st.st_mode))
    if stats is not None:
        stats.top_dirs = per_dir.most_common(5)
    return dict(sorted(out.items()))


def write_modes(workdir: Path, stats: WorkdirStats | None = None,
                exclude: list[str] | None = None) -> bool:
    """(Re)write the manifest if its content changed, remove it when nothing is executable.
    Returns whether the file changed."""
    path = workdir / MODES_FILE
    modes = collect_modes(workdir, stats, exclude)
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
    # optional: `list_info(prefix) -> list[tuple[str, float | None]]`, paths with their mtime
    # (POSIX seconds) when the store keeps one; restore uses it to set local mtimes
    def download(self, files: list[tuple[str, Path]]) -> None: ...
    def sync_up(self, local: Path, prefix: str, *, delete: bool = True,
                exclude: list[str] | None = None) -> None:
        """Upload new/changed files of a local directory to `prefix`; with `delete`, also
        delete remote files missing locally. `exclude`: fnmatch patterns on relative paths,
        neither uploaded nor deleted."""
        ...

    def sync_down(self, prefix: str, local: Path) -> None: ...
    def delete(self, paths: list[str]) -> None:
        """Delete files (paths relative to the flock root). Used by the controller's state."""
        ...


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
        return [r for r, _ in self.list_info(prefix)]

    def list_info(self, prefix: str) -> list[tuple[str, float | None]]:
        from huggingface_hub import BucketFile

        self.stats.add("ops", "list_bucket_tree")
        full = self.base + prefix
        out: list[tuple[str, float | None]] = []
        for f in self.api.list_bucket_tree(self.bucket_id, prefix=full, recursive=True):
            if isinstance(f, BucketFile) and f.path.startswith(full):
                mt = f.mtime.timestamp() if isinstance(f.mtime, datetime) else None
                out.append((f.path[len(self.base):], mt))
        return out

    def download(self, files: list[tuple[str, Path]]) -> None:
        self.stats.add("ops", "download_bucket_files")
        for _, local in files:
            local.parent.mkdir(parents=True, exist_ok=True)
        self.api.download_bucket_files(self.bucket_id, [(self.base + r, str(p)) for r, p in files])

    def delete(self, paths: list[str]) -> None:
        for i in range(0, len(paths), UPLOAD_CHUNK):
            self.stats.add("ops", "batch_bucket_files")
            chunk = paths[i : i + UPLOAD_CHUNK]
            self.api.batch_bucket_files(self.bucket_id, delete=[self.base + r for r in chunk])

    def _url(self, prefix: str) -> str:
        return f"hf://buckets/{self.bucket_id}/{self.base}{prefix}".rstrip("/")

    def sync_up(self, local: Path, prefix: str, *, delete: bool = True,
                exclude: list[str] | None = None) -> None:
        self.stats.add("ops", "sync_bucket_up")
        self.api.sync_bucket(str(local), self._url(prefix), delete=delete, exclude=exclude or None,
                             quiet=True)

    def sync_down(self, prefix: str, local: Path) -> None:
        self.stats.add("ops", "sync_bucket_down")
        local.mkdir(parents=True, exist_ok=True)
        self.api.sync_bucket(self._url(prefix), str(local), quiet=True)


@dataclass
class RestoreResult:
    workdir: int = 0
    sessions: int = 0
    harness: int = 0
    # bucket paths (relative to the flock root) that could not be downloaded, after retries
    failed: list[str] = field(default_factory=list)


_NOT_FOUND = re.compile(r"\b404\b|not found", re.IGNORECASE)


def _permanent(e: BaseException) -> bool:
    """A download error that retrying cannot fix: the entry's data is gone (HTTP 404)."""
    resp = getattr(e, "response", None)
    if getattr(resp, "status_code", None) == 404:
        return True
    return bool(_NOT_FOUND.search(str(e)))


def _part(p: Path) -> Path:
    return p.with_name(p.name + PART_SUFFIX)


def download_resilient(store: BucketStore, files: list[tuple[str, Path]], *,
                       backoff: tuple[float, ...] | None = None, sleep=None,
                       progress=None) -> tuple[list[tuple[str, Path]], list[str]]:
    """Download (remote, local) pairs; returns (downloaded, failed remote paths). Never raises
    for a file: batches first, then the files of a failed batch one by one, retried after each
    delay in `backoff` unless the error is a permanent "not found" (`_permanent`). Each file lands in a `.flock-part` sibling, renamed into place only
    when it arrived, so a failed file never leaves a truncated copy at its real path.
    `progress(n)` is called with the number of files each attempt brought in."""

    gone: list[str] = []  # permanent "not found": skipped, never retried

    def attempt(batch: list[tuple[str, Path]]) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
        ok, bad = [], []
        try:
            store.download([(r, _part(p)) for r, p in batch])
        except Exception as e:  # noqa: BLE001 - sorted out per file below
            err = e
        else:
            err = None
        for r, p in batch:
            part = _part(p)
            if err is None and part.is_file():
                try:
                    os.replace(part, p)
                    ok.append((r, p))
                    continue
                except OSError:
                    pass
            part.unlink(missing_ok=True)
            bad.append((r, p))
        if ok and progress is not None:
            progress(len(ok))
        if err is not None and len(batch) == 1:
            log.debug("download of %s failed: %s", batch[0][0], _short(err))
            if bad and _permanent(err):
                gone.append(bad[0][0])
                return ok, []
        return ok, bad

    backoff = RESTORE_BACKOFF if backoff is None else backoff
    sleep = sleep or time.sleep
    done: list[tuple[str, Path]] = []
    pending: list[tuple[str, Path]] = []
    size = restore_batch()
    for i in range(0, len(files), size):
        ok, bad = attempt(files[i : i + size])
        done += ok
        pending += bad
    if pending:  # each file alone, so one bad file costs only itself
        retry, pending = pending, []
        for f in retry:
            ok, bad = attempt([f])
            done += ok
            pending += bad
    for delay in backoff:
        if not pending:
            break
        sleep(delay)
        retry, pending = pending, []
        for f in retry:
            ok, bad = attempt([f])
            done += ok
            pending += bad
    # the caller logs the skipped files (one summary line per agent, `BucketSync._restore_files`)
    return done, gone + [r for r, _ in pending]


def _clean_parts(root: Path) -> int:
    """Remove stray `.flock-part` files under `root` (a crash mid-download leaves them)."""
    n = 0
    if root.is_dir():
        for p in root.rglob("*" + PART_SUFFIX):
            try:
                if p.is_file() and not p.is_symlink():
                    p.unlink()
                    n += 1
            except OSError:
                pass
    return n


def _short(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e)[:300]}"


class BucketSync:
    def __init__(self, store: BucketStore, swarm: str) -> None:
        self.store = store
        self.swarm = swarm
        self.stats: BucketStats = getattr(store, "stats", None) or BucketStats()
        self._seen: dict[Path, tuple[int, int]] = {}
        # per agent prefix: may the workdir push delete remote files? Absent: no restore was
        # attempted (a new agent), deletes allowed. False from the start of a restore until it
        # completes with every file; stays False after an incomplete one (module docstring).
        self._mirror: dict[str, bool] = {}
        self._restoring: set[str] = set()  # agent prefixes with a restore in flight
        self._meta: dict[str, dict] = {}  # per agent prefix: the meta.json last pushed
        self._quota_at: dict[str, float] = {}  # per agent prefix: monotonic time of the last note
        # per agent prefix with a restore in flight: [files done, files listed, started (UTC)]
        self.restore_progress: dict[str, list] = {}

    def mirror_deletes(self, rt: AgentRuntime) -> bool:
        return self._mirror.get(self.prefix(rt), True)

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
        pre = self.prefix(rt)
        if pre in self._restoring:  # never mirror a half-restored agent (module docstring)
            log.warning("%s: push skipped, its restore is still running", rt.name)
            return 0
        meta: list[tuple[Path, str]] = []
        if rt.spec.persist_workdir and rt.workdir.is_dir():
            exclude = workdir_excludes() + ephemeral_excludes(rt.spec)
            ws = WorkdirStats()
            await asyncio.to_thread(write_modes, rt.workdir, ws, exclude)
            rt.workdir_files, rt.workdir_stats = ws.files, ws
            self._quota(rt, ws)
            stats = {"files": ws.files, "bytes": ws.bytes}
            if self._meta.get(pre) != stats:  # the workdir's size: predicts a restore's time
                path = rt.sessions_dir / f"{rt.name}.meta.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({**stats, "ts": datetime.now(UTC).isoformat()}))
                meta = [(path, pre + "meta.json")]
            delete = self.mirror_deletes(rt)
            if not delete:
                log.info("%s: workdir push without remote deletes (last restore incomplete)",
                         rt.name)
            await asyncio.to_thread(self.store.sync_up, rt.workdir, pre + "workdir",
                                    delete=delete, exclude=exclude)
        files = self._changed(self.collect(rt)) + meta
        if files:
            await asyncio.to_thread(self.store.upload, files)
            self._mark(files)
            if meta:
                self._meta[pre] = stats
        return len(files)

    def _quota(self, rt: AgentRuntime, ws: WorkdirStats) -> None:
        """Above the soft limit: a note in front of the agent's next prompt, and above the hard
        one a `workdir_over_quota` event, each at most hourly. The push itself goes on."""
        w = rt.spec.workdir or WorkdirSpec()
        note, pre, now = quota_note(ws, w), self.prefix(rt), time.monotonic()
        if note is None or now - self._quota_at.get(pre, -QUOTA_NOTE_EVERY) < QUOTA_NOTE_EVERY:
            return
        self._quota_at[pre] = now
        rt._note(note)
        if over_hard_limit(ws, w):
            rt.emit("workdir_over_quota", {"files": ws.files, "bytes": ws.bytes,
                                           "top_dirs": [list(t) for t in ws.top_dirs]})

    async def restore(self, rt: AgentRuntime) -> RestoreResult:
        try:
            return await self._restore(rt)
        except Exception:
            self.stats.add("errors", "restore_failed")
            raise

    async def _restore(self, rt: AgentRuntime) -> RestoreResult:
        pre = self.prefix(rt)
        self._mirror[pre] = False  # until this restore completes with every file
        self._restoring.add(pre)
        self.restore_progress[pre] = [0, 0, datetime.now(UTC)]
        try:
            res = await self._restore_files(rt, pre)
        finally:
            self._restoring.discard(pre)
            self.restore_progress.pop(pre, None)
        self._mirror[pre] = not res.failed
        return res

    async def _restore_files(self, rt: AgentRuntime, pre: str) -> RestoreResult:
        lister = getattr(self.store, "list_info", None)
        if lister is not None:  # raises: the caller starts fresh
            listed = await asyncio.to_thread(lister, pre)
        else:
            listed = [(r, None) for r in await asyncio.to_thread(self.store.list, pre)]
        mtimes = {r: mt for r, mt in listed}
        res = RestoreResult()
        work: list[tuple[str, Path]] = []
        plan: list[tuple[str, Path]] = []
        home = Path(rt.adapter.home)
        n = sum([await asyncio.to_thread(_clean_parts, d) for d in (rt.workdir, rt.sessions_dir)])
        if n:
            log.info("%s: removed %d stray %s file(s)", rt.name, n, PART_SUFFIX)
        skip = workdir_excludes() + ephemeral_excludes(rt.spec)
        for r, _ in listed:
            rel = r[len(pre):]
            area, _, sub = rel.partition("/")
            if rel == "meta.json":  # known, the first push after a move need not resend it
                plan.append((r, rt.sessions_dir / f"{rt.name}.meta.json"))
                continue
            if not sub or ".." in Path(sub).parts:
                continue
            if area == "workdir":
                if rt.spec.persist_workdir and not is_excluded(sub, skip):
                    work.append((r, rt.workdir / sub))
            elif area == "sessions":
                plan.append((r, rt.sessions_dir / sub))
            elif area == "harness":
                plan.append((r, home / sub))
        prog = self.restore_progress.get(pre)
        if prog is not None:
            prog[1] = len(work) + len(plan)

        def progress(n: int) -> None:
            if prog is not None:
                prog[0] += n

        if work:
            rt.workdir.mkdir(parents=True, exist_ok=True)
            got, failed = await asyncio.to_thread(download_resilient, self.store, work,
                                                  progress=progress)
            res.workdir, res.failed = len(got), res.failed + failed
            await asyncio.to_thread(_set_mtimes, got, mtimes)
            if n := await asyncio.to_thread(apply_modes, rt.workdir):
                log.info("%s: restored the mode of %d file(s)", rt.name, n)
        if plan:
            for _, p in plan:
                _part(p).unlink(missing_ok=True)
            got, failed = await asyncio.to_thread(download_resilient, self.store, plan,
                                                  progress=progress)
            res.failed += failed
            await asyncio.to_thread(_set_mtimes, got, mtimes)
            for r, p in got:
                if r == pre + "meta.json":
                    try:
                        self._meta[pre] = {k: v for k, v in json.loads(p.read_text()).items()
                                           if k in ("files", "bytes")}
                    except (OSError, ValueError, AttributeError):
                        pass
                elif r.startswith(pre + "sessions/"):
                    res.sessions += 1
                else:
                    res.harness += 1
            self._mark([(p, r) for r, p in got])
        if res.failed:
            self.stats.add("errors", "restore_file_failed", len(res.failed))
            state = f"{pre}sessions/{rt.state_path.name}"
            if state in res.failed:
                log.warning("%s: state file %s not restored; the agent starts a fresh session",
                            rt.name, state)
            log.warning("%s: restored workdir=%d sessions=%d harness=%d; %d file(s) skipped "
                        "(not found or still failing after retries), e.g. %s", rt.name,
                        res.workdir, res.sessions, res.harness, len(res.failed),
                        ", ".join(res.failed[:10]))
        else:
            log.info("%s: restored %s", rt.name, res)
        return res


def _set_mtimes(files: list[tuple[str, Path]], mtimes: dict[str, float | None]) -> None:
    """Give restored files the bucket's mtime, so the next `sync_up` sees them unchanged."""
    for r, p in files:
        mt = mtimes.get(r)
        if mt is None:
            continue
        try:
            os.utime(p, (mt, mt), follow_symlinks=False)
        except (OSError, NotImplementedError):
            pass
