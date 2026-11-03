"""`<data-dir>/current`: how a second terminal finds a local run `flock launch` started (docs/local.md).

A JSON file (mode 0600) with the run's controller URL, its generated token, the run dir and the
pid of its background process (flock.controller.localrun). Written when the controller is up, removed at exit; a file whose
pid is gone is ignored. No controller dependencies: the plain CLI reads it.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

CURRENT_FILE = "current"


def write_current(data_dir: Path, info: dict[str, Any]) -> Path:
    path = Path(data_dir) / CURRENT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return path


def clear_current(data_dir: Path, pid: int) -> None:
    """Remove `current` if it still describes this process's run."""
    path = Path(data_dir) / CURRENT_FILE
    with contextlib.suppress(OSError, ValueError):
        if json.loads(path.read_text(encoding="utf-8")).get("pid") == pid:
            path.unlink()


def read_current(path: Path) -> dict[str, Any] | None:
    """URL and token of a live local run, or None (no file, unreadable, or its pid is gone)."""
    try:
        info = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pid = info.get("pid") if isinstance(info, dict) else None
    if not isinstance(pid, int) or not info.get("url") or not info.get("token"):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # alive, someone else's
    return info
