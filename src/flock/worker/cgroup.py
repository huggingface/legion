"""The worker's memory as its cgroup sees it, and per-agent process memory, for the heartbeat
and the periodic memory line (telemetry only; nothing here acts on it).

`CgroupMem` finds the worker's cgroup from `/proc/self/cgroup` and reads v2
`memory.{current,stat,events,pressure}` (v1 `memory.{usage_in_bytes,stat,oom_control}` as a
fallback). The limit is the smallest finite `memory.max` (v1 `memory.limit_in_bytes`) on the way
up to the root, since the job's limit usually sits on a parent; FLOCK_MEMORY_LIMIT_MB stands in
when none is found. Without a limit the reader is not `armed`.

Non-reclaimable memory is anon + shmem + slab_unreclaimable + kernel_stack + pagetables
(page cache can be dropped, so `memory.current` overstates the pressure). The slope is a least
squares fit over the last hour of samples; `eta_limit_s` is the time to the limit at that slope.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping
from pathlib import Path

log = logging.getLogger(__name__)

MB = 1024 * 1024
UNLIMITED = 1 << 60  # v1 reports "no limit" as a huge number
SLOPE_WINDOW_S = 3600.0
SLOPE_MIN_SPAN_S = 300.0  # no slope from less than 5 min of samples
NONRECLAIM = ("anon", "shmem", "slab_unreclaimable", "kernel_stack", "pagetables")
NONRECLAIM_V1 = ("total_rss", "total_shmem")  # v1 has no slab split


def _read(p: Path) -> str | None:
    try:
        return p.read_text()
    except OSError:
        return None


def _int(p: Path) -> int | None:
    s = _read(p)
    try:
        return int(s.strip()) if s is not None else None
    except ValueError:
        return None  # "max"


def _kv(p: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in (_read(p) or "").splitlines():
        k, _, v = line.partition(" ")
        try:
            out[k] = int(v)
        except ValueError:
            pass
    return out


class CgroupMem:
    def __init__(self, root: str | Path = "/", env: Mapping[str, str] = os.environ,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.root = Path(root)
        self.clock = clock
        self.samples: deque[tuple[float, float]] = deque()
        self.version, self.path = self._find()
        self.limit_mb: float | None = self._limit()
        if self.limit_mb is None and env.get("FLOCK_MEMORY_LIMIT_MB"):
            try:
                self.limit_mb = float(env["FLOCK_MEMORY_LIMIT_MB"])
            except ValueError:
                log.warning("FLOCK_MEMORY_LIMIT_MB=%r is not a number", env["FLOCK_MEMORY_LIMIT_MB"])

    @property
    def armed(self) -> bool:
        return self.limit_mb is not None

    def _find(self) -> tuple[int, Path | None]:
        """(cgroup version, the worker's cgroup directory) or (0, None) when there is none."""
        base = self.root / "sys/fs/cgroup"
        v1 = None
        for line in (_read(self.root / "proc/self/cgroup") or "").splitlines():
            _, ctrl, rel = line.split(":", 2) if line.count(":") >= 2 else ("", "", "")
            rel = rel.strip().lstrip("/")
            if ctrl == "" and (base / "cgroup.controllers").exists():
                return 2, base / rel
            if "memory" in ctrl.split(","):
                v1 = base / "memory" / rel
        return (1, v1) if v1 is not None else (0, None)

    def _limit(self) -> float | None:
        """The smallest finite limit from the worker's cgroup up to the root."""
        if self.path is None:
            return None
        top = self.root / "sys/fs/cgroup" / ("memory" if self.version == 1 else "")
        name = "memory.max" if self.version == 2 else "memory.limit_in_bytes"
        best, d = None, self.path
        while True:
            v = _int(d / name)
            if v is not None and v < UNLIMITED and (best is None or v < best):
                best = v
            if d == top or top not in d.parents:
                break
            d = d.parent
        return round(best / MB, 1) if best is not None else None

    def sample(self) -> dict:
        """One reading (MB), with the slope over the last hour; record it for the slope."""
        d = self.path
        if d is None:
            return {"limit_mb": self.limit_mb}
        if self.version == 2:
            current = _int(d / "memory.current")
            stat = _kv(d / "memory.stat")
            keys = NONRECLAIM
            oom = _kv(d / "memory.events").get("oom_kill")
            psi = None
            for line in (_read(d / "memory.pressure") or "").splitlines():
                if line.startswith("some "):
                    psi = next((float(f[6:]) for f in line.split() if f.startswith("avg10=")), None)
        else:
            current = _int(d / "memory.usage_in_bytes")
            stat = _kv(d / "memory.stat")
            keys = NONRECLAIM_V1 if "total_rss" in stat else ("rss", "shmem")
            oom = _kv(d / "memory.oom_control").get("oom_kill")
            psi = None
        nonreclaim = sum(stat.get(k, 0) for k in keys) / MB if stat else None
        now = self.clock()
        if nonreclaim is not None:
            self.samples.append((now, nonreclaim))
        while self.samples and self.samples[0][0] < now - SLOPE_WINDOW_S:
            self.samples.popleft()
        slope = self.slope_mb_h()
        eta = None
        if slope and slope > 0 and self.limit_mb is not None and nonreclaim is not None:
            eta = max(0.0, (self.limit_mb - nonreclaim) / slope * 3600)
        return {"current_mb": None if current is None else round(current / MB, 1),
                "limit_mb": self.limit_mb,
                "nonreclaim_mb": None if nonreclaim is None else round(nonreclaim, 1),
                "slope_mb_h": None if slope is None else round(slope, 1),
                "eta_limit_s": None if eta is None else round(eta), "psi_some_avg10": psi,
                "oom_kills": oom}

    def slope_mb_h(self) -> float | None:
        s = self.samples
        if len(s) < 2 or s[-1][0] - s[0][0] < SLOPE_MIN_SPAN_S:
            return None
        n = len(s)
        mt = sum(t for t, _ in s) / n
        mv = sum(v for _, v in s) / n
        den = sum((t - mt) ** 2 for t, _ in s)
        return sum((t - mt) * (v - mv) for t, v in s) / den * 3600 if den else None


# ---------------------------------------------------------------------- per-agent processes


def _rollup(pid: int) -> dict[str, int]:
    """/proc/<pid>/smaps_rollup in kB ({} when the process is gone or unreadable)."""
    out: dict[str, int] = {}
    for line in (_read(Path(f"/proc/{pid}/smaps_rollup")) or "").splitlines():
        k, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[k] = int(parts[0])
    return out


def _comm(pid: int) -> str:
    return (_read(Path(f"/proc/{pid}/comm")) or "?").strip()


def du_mb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total / MB


def agent_memory(pid: int | None, descendants: Callable[[int], list[tuple[int, str]]],
                 tmp_dir: Path | None = None) -> dict:
    """PSS of an agent's harness process and its tools (by `comm`), its sandbox's tmpfs root
    and its tmp dir, in MB. `pid` is the process the adapter started (None: none running);
    under isolation that is `unshare`, and the harness is its first descendant."""
    out = {} if pid is None else _process_memory(pid, descendants)
    if tmp_dir is not None and tmp_dir.is_dir():
        out["tmp_mb"] = round(du_mb(tmp_dir), 1)
    return out


def _process_memory(pid: int, descendants: Callable[[int], list[tuple[int, str]]]) -> dict:
    kids = [p for p, _ in descendants(pid)]
    harness = kids.pop(0) if _comm(pid) == "unshare" and kids else pid
    roll = _rollup(harness)
    tools: Counter[str] = Counter()
    for p in kids:
        if (kb := _rollup(p).get("Pss")) and p != harness:
            tools[_comm(p)] += kb
    try:
        st = os.statvfs(f"/proc/{harness}/root/")
        tmpfs = (st.f_blocks - st.f_bfree) * st.f_frsize / MB
    except PermissionError:  # another user namespace: the shmem it maps is the next best thing
        tmpfs = roll.get("Pss_Shmem", 0) / 1024
    except OSError:
        tmpfs = None
    return {"pi_pss_mb": round(roll.get("Pss", 0) / 1024, 1),
            "tools_pss_mb": round(sum(tools.values()) / 1024, 1),
            "tools_by_comm": {c: round(kb / 1024, 1) for c, kb in tools.most_common(5)},
            "tmpfs_root_mb": None if tmpfs is None else round(tmpfs, 1)}


TMPFS_TYPES = ("tmpfs", "ramfs")


def fs_type(path: Path, mounts: str | None = None) -> str | None:
    """The filesystem type of the mount holding `path`, from /proc/self/mounts."""
    if mounts is None:
        mounts = _read(Path("/proc/self/mounts")) or ""
    p = str(Path(path).resolve())
    best, kind = "", None
    for line in mounts.splitlines():
        f = line.split()
        if len(f) < 3:
            continue
        mp = f[1].replace("\\040", " ")
        if (p == mp or p.startswith(mp.rstrip("/") + "/")) and len(mp) >= len(best):
            best, kind = mp, f[2]
    return kind
