"""A home directory that works in containers without one (Enroot/Pyxis on Slurm, docs/slurm.md).

The cluster runs flock with no HOME, or a HOME that is not writable, and in its containers
`/tmp` is a RAM tmpfs that counts against the job's memory. Nothing on the hot path may expand
`~` or default to `/tmp` then: `homeless_defaults(root)` points HOME, HF_HOME, XDG_CACHE_HOME,
UV_CACHE_DIR and TMPDIR under `root` (a worker: the parent of WORK_ROOT, e.g. /work), only for
names that are not set already, and `home()` is $HOME when it is a writable directory, else
`<parent of WORK_ROOT>/home`, else a per-user dir under the temp dir.
"""

import os
import tempfile
from collections.abc import MutableMapping
from pathlib import Path

HOMELESS_VARS = {"HOME": "home", "HF_HOME": "hf", "XDG_CACHE_HOME": "cache", "UV_CACHE_DIR": "uv",
                 "TMPDIR": "tmp"}


def _writable_dir(path: str | os.PathLike | None) -> bool:
    return bool(path) and os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK)


def usable_home(env: MutableMapping[str, str] = os.environ) -> bool:
    return _writable_dir(env.get("HOME"))


def home(env: MutableMapping[str, str] = os.environ) -> Path:
    """$HOME if it is a writable directory, else <parent of WORK_ROOT>/home, else
    <tmp>/flock-home-<uid> (created)."""
    if usable_home(env):
        return Path(env["HOME"])
    if env.get("WORK_ROOT"):
        fallback = Path(env["WORK_ROOT"]).parent / "home"
    else:
        fallback = Path(tempfile.gettempdir()) / f"flock-home-{os.getuid()}"
    fallback.mkdir(mode=0o700, parents=True, exist_ok=True)
    return fallback


def homeless_defaults(root: Path, env: MutableMapping[str, str] = os.environ) -> dict[str, str]:
    """With no usable HOME, set the unset (or unusable) HOME, HF_HOME, XDG_CACHE_HOME,
    UV_CACHE_DIR and TMPDIR to `<root>/<name>` (created). Returns what it set. TMPDIR takes
    effect for this process's `tempfile` too."""
    if usable_home(env):
        return {}
    out = {}
    for var, sub in HOMELESS_VARS.items():
        if var == "HOME" or not env.get(var):
            path = Path(root) / sub
            path.mkdir(parents=True, exist_ok=True)
            env[var] = out[var] = str(path)
    if "TMPDIR" in out:
        tempfile.tempdir = None  # re-read TMPDIR
    return out


def cache_defaults(base: Path | None = None,
                   env: MutableMapping[str, str] = os.environ) -> dict[str, str]:
    """`homeless_defaults` for processes with no work root (the controller under its data dir,
    a world): under $FLOCK_HOMELESS_ROOT if set (a container: /work), else `base`, else the
    temp dir's flock-<uid>."""
    if env.get("FLOCK_HOMELESS_ROOT"):
        root = Path(env["FLOCK_HOMELESS_ROOT"])
    elif base is not None:
        root = Path(base)
    else:
        root = Path(tempfile.gettempdir()) / f"flock-{os.getuid()}"
    return homeless_defaults(root, env)
