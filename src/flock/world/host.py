"""Controller side of a world (docs/world.md): start a `command` world as a child process
(`compute: local`), fetch and check its manifest at every apply, derive the shared secret.

At `PUT /swarms/{swarm}` the controller calls `resolve`: it starts (or keeps, or restarts) the
world, waits for `GET /manifest`, validates it, checks every type's `world:<tool>` entries
against it, and writes `world.endpoint` and `world.manifest` into the spec it stores. Workers
get that block with every assign and heartbeat; the secret goes to them as a job secret
(`FLOCK_WORLD_TOKEN`), never in the spec.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import httpx

from flock.models import COMMS_TOOLS, SwarmSpec, WorldSpec, comms_name, comms_tools, world_tools
from flock.world.protocol import (
    HOST_ENV,
    PORT_ENV,
    SWARM_ENV,
    TOKEN_ENV,
    TOKEN_HEADER,
    WorldManifest,
    manifest_error,
)

log = logging.getLogger("flock.world")

START_TIMEOUT_S = 30.0
MANIFEST_TIMEOUT_S = 10.0
STOP_GRACE_S = 5.0
LOG_TAIL = 1500


class WorldError(ValueError):
    """Why the world cannot be applied (the message says what to fix)."""


def derived_token(controller_token: str, swarm: str) -> str:
    """The secret of a world flock starts without `token_env`: stable for the controller's
    token and the swarm, so a relaunched worker and a restarted world agree without storing it."""
    return hmac.new(controller_token.encode(), f"flock-world:{swarm}".encode(),
                    hashlib.sha256).hexdigest()


def world_token(spec: SwarmSpec, controller_token: str, secrets: Mapping[str, str]) -> str | None:
    """The secret flock sends as X-World-Token for this swarm's world (None: no secret)."""
    w = spec.world
    if w is None:
        return None
    if w.token_env:
        return secrets.get(w.token_env) or None
    if w.command is not None and spec.compute.kind == "local" and controller_token:
        return derived_token(controller_token, spec.swarm)
    return None


def fetch_manifest(url: str, token: str | None, *, transport: httpx.BaseTransport | None = None,
                   timeout: float = MANIFEST_TIMEOUT_S) -> WorldManifest:
    """GET <url>/manifest and validate it (raises WorldError)."""
    headers = {TOKEN_HEADER: token} if token else {}
    try:
        with httpx.Client(transport=transport, timeout=timeout) as http:
            r = http.get(f"{url}/manifest", headers=headers)
    except httpx.HTTPError as e:
        raise WorldError(f"world: cannot reach {url}/manifest: {type(e).__name__}: {e}") from None
    if r.status_code == 401:
        raise WorldError(f"world: {url} refused the token (401): check world.token_env")
    if r.status_code != 200:
        raise WorldError(f"world: GET {url}/manifest answered {r.status_code}: {r.text[:300]}")
    try:
        return WorldManifest.model_validate(r.json())
    except ValueError as e:
        raise WorldError(f"world: invalid manifest from {url}: {manifest_error(e)}") from None


def check_tools(spec: SwarmSpec, folder: Callable[[str], list[str]]) -> None:
    """Every `world:<tool>` a type lists exists in the manifest, and no world tool a type gets
    has the name of one of its folder or comms tools (raises WorldError)."""
    w = spec.world
    assert w is not None and w.manifest is not None
    have = w.manifest.names()
    for name, ts in spec.types.items():
        names = world_tools(ts, w)
        unknown = [n for n in names if n not in have]
        if unknown:
            raise WorldError(f"types.{name}.tools: unknown world tool(s) {', '.join(unknown)}; "
                             f"world {w.manifest.name} has: {', '.join(have) or 'none'}")
        others = set(folder(name)) | {comms_name(e) for e in comms_tools(ts, spec.comms)
                                      if e in COMMS_TOOLS}
        if clash := sorted(set(names) & others):
            raise WorldError(f"types.{name}: world tool(s) {', '.join(clash)} have the name of a "
                             "folder or comms tool the type gets; rename one, or list the world "
                             "tools the type should get as world:<tool>")


def _free_port(host: str) -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def py_fingerprint(folder: Path) -> str:
    """The *.py files under `folder` (path, size, mtime): a re-apply restarts a started world
    when it changed. Dot directories, __pycache__ and virtualenvs are skipped."""
    h = hashlib.sha256()
    skip = {"__pycache__", "node_modules", "venv"}
    files = []
    for p in folder.rglob("*.py"):
        rel = p.relative_to(folder)
        if any(part.startswith(".") or part in skip for part in rel.parts[:-1]):
            continue
        with contextlib.suppress(OSError):
            st = p.stat()
            files.append(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}")
    for f in sorted(files)[:5000]:
        h.update(f.encode() + b"\n")
    return h.hexdigest()[:16]


@dataclass
class _Proc:
    popen: subprocess.Popen
    key: tuple
    url: str
    log_path: Path | None
    log_file: IO[bytes] | None

    def alive(self) -> bool:
        return self.popen.poll() is None


class WorldHost:
    """The worlds this controller started, one per swarm (`compute: local`). Thread-safe: the
    API's apply runs in a thread pool."""

    def __init__(self, log_dir: Path | None = None, *, host: str = "127.0.0.1",
                 start_timeout: float = START_TIMEOUT_S) -> None:
        self.log_dir = Path(log_dir) if log_dir else None
        self.host = host
        self.start_timeout = start_timeout
        self._procs: dict[str, _Proc] = {}
        self._lock = threading.RLock()

    def url(self, swarm: str) -> str | None:
        p = self._procs.get(swarm)
        return p.url if p is not None and p.alive() else None

    def ensure(self, swarm: str, w: WorldSpec, token: str | None) -> tuple[str, bool]:
        """The url of the swarm's started world: the running one when its command, dir and
        Python files are unchanged, else a new process. Returns (url, started now)."""
        assert w.command is not None
        cwd = Path(w.dir).resolve() if w.dir else Path.cwd()
        if not cwd.is_dir():
            raise WorldError(f"world.dir {cwd} does not exist")
        key = (tuple(w.command), str(cwd), token, py_fingerprint(cwd))
        with self._lock:
            p = self._procs.get(swarm)
            if p is not None and p.alive() and p.key == key:
                return p.url, False
            if p is not None:
                log.info("world for %s: %s; restarting", swarm,
                         "changed" if p.alive() else f"exited ({p.popen.returncode})")
                self.stop(swarm)
            return self._start(swarm, w.command, cwd, key, token), True

    def _start(self, swarm: str, command: list[str], cwd: Path, key: tuple,
               token: str | None) -> str:
        port = _free_port(self.host)
        argv = [x.replace("{port}", str(port)).replace("{python}", sys.executable)
                for x in command]
        env = {k: v for k, v in os.environ.items() if not k.startswith("FLOCK_")}
        env.update({PORT_ENV: str(port), HOST_ENV: self.host, SWARM_ENV: swarm})
        if token:
            env[TOKEN_ENV] = token
        log_path = log_file = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"world-{swarm}.log"
            log_file = log_path.open("ab")
        try:
            popen = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log_file, stderr=subprocess.STDOUT if log_file else None,
                                     start_new_session=True)
        except OSError as e:
            if log_file:
                log_file.close()
            raise WorldError(f"world.command: cannot start {argv[0]!r} in {cwd}: {e}") from None
        url = f"http://{self.host}:{port}"
        proc = _Proc(popen, key, url, log_path, log_file)
        self._procs[swarm] = proc
        log.info("world for %s: started %s (pid %d) at %s", swarm, " ".join(argv), popen.pid, url)
        deadline = time.monotonic() + self.start_timeout
        while True:
            if not proc.alive():
                self.stop(swarm)
                raise WorldError(f"world.command exited with {popen.returncode} before serving "
                                 f"GET /manifest{self._tail(log_path)}")
            try:
                r = httpx.get(f"{url}/manifest", timeout=2,
                              headers={TOKEN_HEADER: token} if token else {})
                if r.status_code < 500:
                    return url
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                self.stop(swarm)
                raise WorldError(f"world.command did not serve GET /manifest on port {port} "
                                 f"within {self.start_timeout:g} s (it gets the port as "
                                 f"${PORT_ENV} or {{port}}){self._tail(log_path)}")
            time.sleep(0.1)

    @staticmethod
    def _tail(path: Path | None) -> str:
        if path is None or not path.is_file():
            return ""
        text = path.read_bytes()[-LOG_TAIL:].decode(errors="replace").strip()
        return f"; its log ({path}) ends with:\n{text}" if text else ""

    def stop(self, swarm: str) -> None:
        with self._lock:
            p = self._procs.pop(swarm, None)
        if p is None:
            return
        if p.alive():
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.popen.pid, signal.SIGTERM)
            try:
                p.popen.wait(STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(p.popen.pid, signal.SIGKILL)
                p.popen.wait()
            log.info("world for %s: stopped", swarm)
        if p.log_file is not None:
            p.log_file.close()

    def stop_all(self) -> None:
        for swarm in list(self._procs):
            self.stop(swarm)


def resolve(spec: SwarmSpec, *, host: WorldHost | None, controller_token: str,
            secrets: Mapping[str, str], folder: Callable[[str], list[str]],
            transport: httpx.BaseTransport | None = None) -> None:
    """Fill `spec.world.endpoint` and `spec.world.manifest` (raises WorldError). Starts or
    restarts a `command` world with `compute: local`; values a client sent are replaced."""
    w = spec.world
    if w is None:
        return
    w.endpoint = w.manifest = None
    token = world_token(spec, controller_token, secrets)
    if w.token_env and not token:
        raise WorldError(f"world.token_env: the secret {w.token_env} is not set on the controller "
                         "(it is sent to the world and handed to the workers)")
    started = False
    if spec.compute.kind == "local" and w.command is not None:
        if host is None:
            raise WorldError("world.command: this controller does not start worlds; give "
                             "world.url instead")
        url, started = host.ensure(spec.swarm, w, token)
    else:
        assert w.url is not None
        url = w.url
    try:
        w.manifest = fetch_manifest(url, token, transport=transport)
        w.endpoint = url
        check_tools(spec, folder)
    except WorldError:
        if started and host is not None:
            host.stop(spec.swarm)
        raise
