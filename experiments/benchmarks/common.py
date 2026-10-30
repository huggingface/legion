"""Small shared utilities. No environment or model calls at import time."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNTIME = ROOT / '.flock/deps/benchmarks-venv/bin/python'
ESCAPE_REVISION = '03742c79f48646e0130886e18e3e79798cf8edd7'


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.chmod(0o600)
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Symlinks are not supported in task snapshots: {p}')
        if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts:
            h.update(p.relative_to(path).as_posix().encode() + b'\0' + p.read_bytes())
    return h.hexdigest()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items()
            if k in {'PATH', 'HOME', 'USER', 'LANG', 'TMPDIR', 'DOCKER_HOST'} or k.startswith('LC_')}


def stop(process, grace=20) -> None:
    if process is None:
        return
    # The group can outlive its leader.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)
