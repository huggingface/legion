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
    ADMIN_TOKEN_ENV,
    HOST_ENV,
    PORT_ENV,
    SWARM_ENV,
    TOKEN_ENV,
    TOKEN_HEADER,
    WORLD_HOST_FILE_ENV,
    WORLD_HOST_VAR,
    WorldManifest,
    credential_id,
    manifest_error,
)

log = logging.getLogger("flock.world")

START_TIMEOUT_S = 30.0
MANIFEST_TIMEOUT_S = 10.0
STOP_GRACE_S = 5.0  # for the whole process group, then SIGKILL
KILL_WAIT_S = 2.0
LOG_TAIL = 1500


class WorldError(ValueError):
    """Why the world cannot be applied (the message says what to fix)."""


class WorldParked(WorldError):
    """The world answers but is parked: it could not build itself and retries (docs/world.md)."""


def derived_token(controller_token: str, swarm: str) -> str:
    """The secret of a world flock starts without `token_env`: stable for the controller's
    token and the swarm, so a relaunched worker and a restarted world agree without storing it."""
    return hmac.new(controller_token.encode(), f"flock-world:{swarm}".encode(),
                    hashlib.sha256).hexdigest()


def credential_origin(spec: SwarmSpec, endpoint: str) -> str:
    """What a world's secret is bound to: the endpoint, or `local:<swarm>` for a world flock
    starts (every process it starts for the swarm gets the same secret, on 127.0.0.1)."""
    w = spec.world
    if w is not None and w.command is not None and spec.compute.kind == "local":
        return f"local:{spec.swarm}"
    if w is not None and w.url is not None and WORLD_HOST_VAR in w.url:
        return w.url  # bound to the template: a world that moves host keeps its credential
    return endpoint


def world_endpoint(url: str, env: Mapping[str, str] = os.environ) -> str:
    """world.url with ${WORLD_HOST} replaced by the first word of FLOCK_WORLD_HOST_FILE."""
    if WORLD_HOST_VAR not in url:
        return url
    path = env.get(WORLD_HOST_FILE_ENV)
    if not path:
        raise WorldError(f"world.url {url} needs {WORLD_HOST_FILE_ENV} on the controller")
    try:
        words = Path(path).read_text().split()
    except OSError as e:
        raise WorldError(f"world host file {path}: {e}") from None
    if not words:
        raise WorldError(f"world host file {path} is empty (is the world job up?)")
    return url.replace(WORLD_HOST_VAR, words[0])


def follow_world_host(spec: SwarmSpec, controller_token: str, secrets: Mapping[str, str], *,
                      env: Mapping[str, str] = os.environ,
                      transport: httpx.BaseTransport | None = None) -> SwarmSpec | None:
    """A swarm whose world.url has ${WORLD_HOST}: if the host file now names another host and
    the world answers there, the spec with the new endpoint (and manifest), to store; workers
    get it with their next heartbeat. None: nothing to change (or the new host does not
    answer yet: try again later)."""
    w = spec.world
    if w is None or w.url is None or WORLD_HOST_VAR not in w.url:
        return None
    try:
        url = world_endpoint(w.url, env)
    except WorldError as e:
        log.warning("world %s: %s", spec.swarm, e)
        return None
    if url == w.endpoint:
        return None
    token = world_token(spec, controller_token, secrets)
    try:
        manifest = fetch_manifest(url, token, transport=transport)
    except WorldError as e:
        log.warning("world of %s moved to %s but does not answer yet: %s", spec.swarm, url, e)
        return None
    new = spec.model_copy(deep=True)
    assert new.world is not None
    log.info("world of %s moved: %s -> %s", spec.swarm, w.endpoint, url)
    new.world.endpoint, new.world.manifest = url, manifest
    return new


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


def world_admin_token(spec: SwarmSpec, controller_token: str,
                      secrets: Mapping[str, str]) -> str | None:
    """The secret of the world's /admin routes (docs/world.md, "Admin API"): for a world flock
    starts, derived from the controller's token and the swarm (the world gets it as
    $FLOCK_WORLD_ADMIN_TOKEN); otherwise $FLOCK_WORLD_ADMIN_TOKEN from the controller's secrets.
    Never derived from the world secret: every worker holds that one."""
    w = spec.world
    if w is None:
        return None
    if w.command is not None and spec.compute.kind == "local" and controller_token:
        return derived_token(controller_token, f"flock-world-admin:{spec.swarm}")
    return secrets.get(ADMIN_TOKEN_ENV) or None


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
    if r.status_code == 503:
        body = None
        with contextlib.suppress(ValueError):
            body = r.json()
        if isinstance(body, dict) and body.get("parked") is True:
            raise WorldParked(f"world parked at {url} (attempt {body.get('attempt')}, next "
                              f"retry in {body.get('next_retry_s')} s): {body.get('error')}")
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
    API's apply runs in a thread pool.

    A changed world is replaced in two steps: `ensure` starts the candidate next to the running
    one (the incumbent), and only `commit` (after the apply passed every check and was stored)
    stops the incumbent; `discard` drops a refused candidate and leaves the incumbent as it was."""

    def __init__(self, log_dir: Path | None = None, *, host: str = "127.0.0.1",
                 start_timeout: float = START_TIMEOUT_S, stop_grace: float = STOP_GRACE_S) -> None:
        self.log_dir = Path(log_dir) if log_dir else None
        self.host = host
        self.start_timeout = start_timeout
        self.stop_grace = stop_grace
        self._procs: dict[str, _Proc] = {}
        self._pending: dict[str, _Proc] = {}
        self._lock = threading.RLock()

    def url(self, swarm: str) -> str | None:
        p = self._procs.get(swarm)
        return p.url if p is not None and p.alive() else None

    def ensure(self, swarm: str, w: WorldSpec, token: str | None,
               admin: str | None = None) -> tuple[str, bool]:
        """The url of the swarm's started world: the running one when its command, dir, secret
        and Python files are unchanged, else a new candidate process (the running one is left
        alone until `commit`). Returns (url, started now)."""
        assert w.command is not None
        cwd = Path(w.dir).resolve() if w.dir else Path.cwd()
        if not cwd.is_dir():
            raise WorldError(f"world.dir {cwd} does not exist")
        key = (tuple(w.command), str(cwd), token, admin, py_fingerprint(cwd))
        with self._lock:
            p = self._procs.get(swarm)
            if p is not None and p.alive() and p.key == key:
                self.discard(swarm)
                return p.url, False
            if p is not None:
                log.info("world for %s: %s; starting a replacement", swarm,
                         "changed" if p.alive() else f"exited ({p.popen.returncode})")
            self.discard(swarm)
            proc = self._start(swarm, w.command, cwd, key, token, admin)
            self._pending[swarm] = proc
            return proc.url, True

    def commit(self, swarm: str) -> None:
        """The apply was stored: the candidate (if any) becomes the swarm's world and the one it
        replaces stops."""
        with self._lock:
            new = self._pending.pop(swarm, None)
            if new is None:
                return
            old = self._procs.get(swarm)
            self._procs[swarm] = new
        if old is not None:
            self._terminate(old)
            log.info("world for %s: replaced", swarm)

    def discard(self, swarm: str) -> None:
        """Drop the candidate of a refused apply; the running world is untouched."""
        with self._lock:
            p = self._pending.pop(swarm, None)
        if p is not None:
            self._terminate(p)

    def _start(self, swarm: str, command: list[str], cwd: Path, key: tuple,
               token: str | None, admin: str | None = None) -> _Proc:
        port = _free_port(self.host)
        argv = [x.replace("{port}", str(port)).replace("{python}", sys.executable)
                for x in command]
        env = {k: v for k, v in os.environ.items() if not k.startswith("FLOCK_")}
        env.update({PORT_ENV: str(port), HOST_ENV: self.host, SWARM_ENV: swarm})
        if token:
            env[TOKEN_ENV] = token
        if admin:
            env[ADMIN_TOKEN_ENV] = admin
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
        log.info("world for %s: started %s (pid %d) at %s", swarm, " ".join(argv), popen.pid, url)
        deadline = time.monotonic() + self.start_timeout
        while True:
            if not proc.alive():
                self._terminate(proc)
                raise WorldError(f"world.command exited with {popen.returncode} before serving "
                                 f"GET /manifest{self._tail(log_path)}")
            try:
                r = httpx.get(f"{url}/manifest", timeout=2,
                              headers={TOKEN_HEADER: token} if token else {})
                if r.status_code < 500:
                    return proc
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                self._terminate(proc)
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
        """Stop the swarm's world (and a candidate, if one is pending)."""
        with self._lock:
            p = self._procs.pop(swarm, None)
            q = self._pending.pop(swarm, None)
        for proc in (p, q):
            if proc is not None:
                self._terminate(proc)
                log.info("world for %s: stopped", swarm)

    def stop_all(self) -> None:
        for swarm in {*self._procs, *self._pending}:
            self.stop(swarm)

    def _terminate(self, p: _Proc) -> None:
        """SIGTERM the world's whole process group (it runs in its own session), wait up to
        `stop_grace` for the leader and every other member, then SIGKILL what is left: a helper
        that ignores SIGTERM does not outlive its world."""
        pgid = p.popen.pid
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + self.stop_grace
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.popen.wait(max(0.0, deadline - time.monotonic()))
        while _group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if p.alive() or _group_alive(pgid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
            p.popen.wait()
            end = time.monotonic() + KILL_WAIT_S  # SIGKILL lands asynchronously
            while _group_alive(pgid) and time.monotonic() < end:
                time.sleep(0.02)
        if p.log_file is not None:
            p.log_file.close()


def _group_alive(pgid: int) -> bool:
    """Any process left in the group, zombies of exited members excepted."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    proc = Path("/proc")
    if not proc.is_dir():  # no procfs: trust killpg
        return True
    for stat in proc.glob("[0-9]*/stat"):
        with contextlib.suppress(OSError, IndexError, ValueError):
            fields = stat.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                return True
    return False


def resolve(spec: SwarmSpec, *, host: WorldHost | None, controller_token: str,
            secrets: Mapping[str, str], folder: Callable[[str], list[str]],
            transport: httpx.BaseTransport | None = None,
            previous: WorldSpec | None = None) -> None:
    """Fill `spec.world.endpoint`, `manifest` and the credential's origin and id (raises
    WorldError); values a client sent are replaced. A `command` world with `compute: local`
    that is new or changed is started as a candidate: the caller stores the spec, then
    `host.commit(swarm)`; on any refusal the candidate is dropped and the running world kept.
    A parked world at the endpoint of `previous` (the stored world block) keeps its last
    manifest: the apply goes through and the world serves again once it is built."""
    w = spec.world
    if w is None:
        return
    w.endpoint = w.manifest = w.credential_origin = w.credential_id = None
    token = world_token(spec, controller_token, secrets)
    if w.token_env and not token:
        raise WorldError(f"world.token_env: the secret {w.token_env} is not set on the controller "
                         "(it is sent to the world and handed to the workers)")
    started = False
    if spec.compute.kind == "local" and w.command is not None:
        if host is None:
            raise WorldError("world.command: this controller does not start worlds; give "
                             "world.url instead")
        url, started = host.ensure(spec.swarm, w, token,
                                   world_admin_token(spec, controller_token, secrets))
    else:
        assert w.url is not None
        url = world_endpoint(w.url)
    try:
        try:
            w.manifest = fetch_manifest(url, token, transport=transport)
        except WorldParked as e:
            if previous is None or previous.manifest is None or previous.endpoint != url:
                raise WorldParked(f"{e}; apply again once it serves its manifest") from None
            log.warning("%s; keeping the last manifest (version %s)", e, previous.manifest.version)
            w.manifest = previous.manifest
        w.endpoint = url
        check_tools(spec, folder)
    except WorldError:
        if started and host is not None:
            host.discard(spec.swarm)  # the running world stays as it was
        raise
    if token:
        w.credential_origin = credential_origin(spec, url)
        w.credential_id = credential_id(token, w.credential_origin)
