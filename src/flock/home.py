"""A home directory that works in containers without one (Enroot/Pyxis on Slurm, docs/slurm.md).

The cluster runs flock with no HOME, or a HOME that is not writable. Nothing on the hot path
may expand `~` then: `home()` is $HOME when it is a writable directory, else a per-user dir
under the temp dir, and `cache_defaults` points HF_HOME, XDG_CACHE_HOME and UV_CACHE_DIR at a
writable base when HOME is unusable (only names that are not set already).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import MutableMapping
from pathlib import Path

CACHE_VARS = {"HF_HOME": "hf", "XDG_CACHE_HOME": "cache", "UV_CACHE_DIR": "uv"}


def _writable_dir(path: str | os.PathLike | None) -> bool:
    return bool(path) and os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK)


def usable_home(env: MutableMapping[str, str] = os.environ) -> bool:
    return _writable_dir(env.get("HOME"))


def home(env: MutableMapping[str, str] = os.environ) -> Path:
    """$HOME if it is a writable directory, else <tmp>/flock-home-<uid> (created)."""
    if usable_home(env):
        return Path(env["HOME"])
    fallback = Path(tempfile.gettempdir()) / f"flock-home-{os.getuid()}"
    fallback.mkdir(mode=0o700, parents=True, exist_ok=True)
    return fallback


def cache_defaults(base: Path | None = None,
                   env: MutableMapping[str, str] = os.environ) -> dict[str, str]:
    """With no usable HOME, set the cache variables that are unset to `<base>/<name>` (default
    base: `home()`). Returns what it set."""
    if usable_home(env):
        return {}
    root = Path(base) if base is not None else home(env)
    out = {}
    for var, sub in CACHE_VARS.items():
        if not env.get(var):
            path = root / sub
            path.mkdir(parents=True, exist_ok=True)
            env[var] = out[var] = str(path)
    return out
