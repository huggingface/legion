"""Tools bundles: the folder as a deterministic tar.gz, versioned by its content hash.

`pack` walks the folder (sorted; `__pycache__`, `*.pyc`, `.git`, `.venv` and dotfiles left out,
symlinks refused), adds the manifest the loader produced as `.flock-manifest.json`, and writes
a tar.gz with fixed metadata (mtime 0, uid/gid 0, mode 644/755), so the same folder always gives
the same bytes. The version is the first 16 hex digits of the archive's sha256. `extract`
checks the hash and every member (regular files and directories only, no absolute paths, no
`..`) before writing anything, then renames into place, so a version directory is complete or
absent. Limits: MAX_BUNDLE_BYTES compressed, MAX_FILES files, MAX_UNPACKED_BYTES unpacked.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from flock.tools.loader import Manifest, ToolsError, describe

MANIFEST_NAME = ".flock-manifest.json"
REQUIREMENTS = "requirements.txt"
MAX_BUNDLE_BYTES = 10 * 1024 * 1024
MAX_UNPACKED_BYTES = 50 * 1024 * 1024
MAX_FILES = 2000
VERSION_CHARS = 16
EXCLUDE_DIRS = frozenset({"__pycache__", ".git", ".venv", "venv", ".mypy_cache", ".pytest_cache",
                          ".ruff_cache"})


@dataclass
class Bundle:
    version: str
    data: bytes
    manifest: Manifest


def version_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:VERSION_CHARS]


def _files(folder: Path) -> list[Path]:
    out: list[Path] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith("."))
        for d in dirs:
            if (Path(root) / d).is_symlink():
                raise ToolsError(f"{Path(root, d)}: symlinks are not allowed in the tools folder")
        for f in sorted(files):
            p = Path(root) / f
            if f.startswith(".") or f.endswith((".pyc", ".pyo")):
                continue
            if p.is_symlink():
                raise ToolsError(f"{p}: symlinks are not allowed in the tools folder")
            out.append(p)
    return out


def _info(name: str, size: int, mode: int) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.size, ti.mode, ti.mtime = size, mode, 0
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    return ti


def pack(folder: str | Path, manifest: Manifest | None = None) -> Bundle:
    """The bundle of `folder`; `manifest` defaults to `describe(folder)` (imports the modules in
    a subprocess). Raises ToolsError for a bad module, a symlink, or a folder over the limits."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ToolsError(f"tools folder {folder} does not exist")
    files = _files(folder)
    if manifest is None:
        manifest = describe(folder)
    if len(files) > MAX_FILES:
        raise ToolsError(f"tools folder {folder} has {len(files)} files; the limit is {MAX_FILES}")
    raw = io.BytesIO()
    total = 0
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for p in files:
            data = p.read_bytes()
            total += len(data)
            if total > MAX_UNPACKED_BYTES:
                raise ToolsError(f"tools folder {folder} is over {MAX_UNPACKED_BYTES} bytes")
            mode = 0o755 if os.access(p, os.X_OK) else 0o644
            tar.addfile(_info(p.relative_to(folder).as_posix(), len(data), mode), io.BytesIO(data))
        m = json.dumps(manifest.to_json(), sort_keys=True, indent=1).encode()
        tar.addfile(_info(MANIFEST_NAME, len(m), 0o644), io.BytesIO(m))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, compresslevel=9, filename="") as gz:
        gz.write(raw.getvalue())
    data = out.getvalue()
    if len(data) > MAX_BUNDLE_BYTES:
        raise ToolsError(f"tools bundle is {len(data)} bytes compressed; the limit is {MAX_BUNDLE_BYTES}")
    return Bundle(version_of(data), data, manifest)


def _members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = tar.getmembers()
    if len(members) > MAX_FILES + 1:
        raise ToolsError(f"bundle has {len(members)} members; the limit is {MAX_FILES}")
    total = 0
    for m in members:
        p = PurePosixPath(m.name)
        if p.is_absolute() or ".." in p.parts or not m.name:
            raise ToolsError(f"bundle member {m.name!r} escapes the folder")
        if not (m.isfile() or m.isdir()):
            raise ToolsError(f"bundle member {m.name!r} is not a regular file")
        total += m.size
        if total > MAX_UNPACKED_BYTES:
            raise ToolsError(f"bundle unpacks to over {MAX_UNPACKED_BYTES} bytes")
    return members


def read_manifest(data: bytes) -> Manifest:
    """The manifest inside a bundle (checks the archive; raises ToolsError)."""
    if len(data) > MAX_BUNDLE_BYTES:
        raise ToolsError(f"tools bundle is {len(data)} bytes; the limit is {MAX_BUNDLE_BYTES}")
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            _members(tar)
            f = tar.extractfile(MANIFEST_NAME)
            if f is None:
                raise KeyError(MANIFEST_NAME)
            return Manifest.from_json(json.loads(f.read()))
    except KeyError as e:
        raise ToolsError(f"bundle has no {MANIFEST_NAME} (made by `flock apply`?)") from e
    except (tarfile.TarError, OSError, EOFError, ValueError, TypeError) as e:
        raise ToolsError(f"not a tools bundle: {type(e).__name__}: {e}") from e


def extract(data: bytes, dest: str | Path, version: str | None = None) -> Path:
    """Unpack into `dest` (created atomically; an existing `dest` is kept as is). Checks the
    content hash against `version` when given."""
    dest = Path(dest)
    if version is not None and version_of(data) != version:
        raise ToolsError(f"bundle content hash {version_of(data)} does not match version {version}")
    if dest.is_dir():
        return dest
    read_manifest(data)  # validates the archive before anything is written
    tmp = dest.with_name(f".{dest.name}.tmp-{uuid.uuid4().hex[:8]}")
    tmp.mkdir(parents=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for m in _members(tar):
                target = tmp / m.name
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                assert src is not None
                target.write_bytes(src.read())
                target.chmod(0o755 if m.mode & 0o111 else 0o644)
        try:
            os.rename(tmp, dest)
        except OSError:
            if not dest.is_dir():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


def load_extracted(dest: str | Path) -> Manifest:
    return Manifest.from_json(json.loads((Path(dest) / MANIFEST_NAME).read_text()))
