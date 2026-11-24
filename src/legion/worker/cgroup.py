"""The worker's memory as its cgroup sees it, for the heartbeat, the periodic memory line and
the worker's relief ladder (`MemThresholds`, docs/slurm.md
"Memory"; the worker acts on these readings, nothing here does).

`CgroupMem` finds the worker's cgroup from `/proc/self/cgroup` and reads v2
`memory.{current,stat,events,pressure}` (v1 `memory.{usage_in_bytes,stat,oom_control}` as a
fallback). The limit is the smallest finite `memory.max` (v1 `memory.limit_in_bytes`) on the way
up to the root, since the job's limit usually sits on a parent; LEGION_MEMORY_LIMIT_MB stands in
when none is found. Without a limit the reader is not `armed`.

Non-reclaimable memory is anon + shmem + slab_unreclaimable + kernel_stack + pagetables
(page cache can be dropped, so `memory.current` overstates the pressure). The slope is a least
squares fit over the last hour of samples; `eta_limit_s` is the time to the limit at that slope.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path

log = logging.getLogger(__name__)

MB = 1024 * 1024
UNLIMITED = 1 << 60  # v1 reports "no limit" as a huge number
SLOPE_WINDOW_S = 3600.0
SLOPE_MIN_SPAN_S = 300.0  # no slope from less than 5 min of samples
NONRECLAIM = ("anon", "shmem", "slab_unreclaimable", "kernel_stack", "pagetables")
NONRECLAIM_V1 = ("total_rss", "total_shmem")  # v1 has no slab split


@dataclass
class MemThresholds:
    """The worker's memory relief ladder, as fractions of the cgroup limit on nonreclaimable
    memory, from LEGION_MEM_* (docs/slurm.md "Memory")."""

    rotate: float = 0.75  # LEGION_MEM_ROTATE: wants_rotation (the controller may drain us)
    drain: float = 0.85  # LEGION_MEM_DRAIN: bounded drain, exit 0 (planned restart)
    preempt: float = 0.93  # LEGION_MEM_PREEMPT: preempt("memory"), exit 75
    drain_deadline_s: float = 300.0  # LEGION_MEM_DRAIN_DEADLINE_S: turns still running are cut

    @classmethod
    def read(cls, knob: Callable[[str, float], float]) -> MemThresholds:
        """Each field through `knob(LEGION_MEM_<NAME>, default)`: Worker.mem_knob (the spec's
        `worker.memory`, then the env variable, then the default)."""
        return cls(**{f.name: knob(f"LEGION_MEM_{f.name.upper()}", f.default) for f in fields(cls)})


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
        for var in ("SLURM_MEM_PER_NODE", "LEGION_MEMORY_LIMIT_MB"):  # MB; Slurm sets the first in-job
            if self.limit_mb is None and env.get(var):
                try:
                    self.limit_mb = float(env[var])
                except ValueError:
                    log.warning("%s=%r is not a number", var, env[var])

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
                # a host-namespace path (/kubepods/...) is not mounted here: the base is ours
                return 2, base / rel if (base / rel).is_dir() else base
            if "memory" in ctrl.split(","):
                v1 = base / "memory" / rel
                if not v1.is_dir():
                    v1 = base / "memory"
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

    def oom_group(self) -> str | None:
        """memory.oom.group of the worker's cgroup: "1" means an OOM kill takes every process
        in it (the whole job step), whatever their oom_score_adj."""
        s = _read(self.path / "memory.oom.group") if self.path is not None else None
        return s.strip() if s else None

    def slope_mb_h(self) -> float | None:
        s = self.samples
        if len(s) < 2 or s[-1][0] - s[0][0] < SLOPE_MIN_SPAN_S:
            return None
        n = len(s)
        mt = sum(t for t, _ in s) / n
        mv = sum(v for _, v in s) / n
        den = sum((t - mt) ** 2 for t, _ in s)
        return sum((t - mt) * (v - mv) for t, v in s) / den * 3600 if den else None


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
