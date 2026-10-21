"""The worker's tools versions (docs/tools.md, "Versions").

The controller names the swarm's tools version in each assign and in every heartbeat response
(with the worker type's selection). `want` records it; a version not on disk yet is fetched
from the controller (`GET /swarms/{swarm}/tools/{version}`), checked against its hash,
extracted to `<root>/<version>/` and, when the bundle has a `requirements.txt`, given its own
venv (`<root>/<version>.venv`, uv when on PATH, else venv + pip). Only then does `current()`
switch to it, together with the selection that came with it: a turn that starts meanwhile
keeps the old version, and a turn that already took a Toolset keeps it to the end. Old versions
stay on disk for the worker's lifetime (bundles are small; a running call may still use one).
A fetch or install failure keeps the previous version and is reported in `Toolset.error`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import sysconfig
from collections.abc import Awaitable, Callable
from pathlib import Path

from flock.tools.bundle import REQUIREMENTS, extract, load_extracted
from flock.tools.loader import Manifest, ToolsError
from flock.tools.toolset import Toolset

log = logging.getLogger(__name__)

INSTALL_TIMEOUT_S = 600.0
Fetch = Callable[[str], Awaitable[bytes]]


def _requirements(folder: Path) -> list[str]:
    f = folder / REQUIREMENTS
    if not f.is_file():
        return []
    return [ln.strip() for ln in f.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]


async def _run(cmd: list[str], timeout: float, cwd: Path | None = None) -> None:
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(cwd) if cwd else None,
                                                stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolsError(f"{cmd[0]} {cmd[1]} took over {timeout:g}s") from None
    if proc.returncode != 0:
        raise ToolsError(f"`{' '.join(cmd[:3])} ...` failed (exit {proc.returncode}): "
                         f"{out.decode(errors='replace').strip()[-800:]}")


async def make_venv(folder: Path, venv: Path, *, timeout: float = INSTALL_TIMEOUT_S) -> str:
    """A venv for the bundle's requirements that also sees the worker's own packages (flock
    included, so `from flock.tools import Context` works): a .pth file adds the worker's
    site-packages after the venv's own. Returns the venv's python."""
    py = venv / "bin" / "python"
    if py.is_file() and (venv / ".flock-ready").is_file():
        return str(py)
    shutil.rmtree(venv, ignore_errors=True)
    uv = shutil.which("uv")
    if uv:
        await _run([uv, "venv", "--quiet", "--python", sys.executable, str(venv)], timeout)
    else:
        await _run([sys.executable, "-m", "venv", str(venv)], timeout)
    purelib = Path(sysconfig.get_paths(vars={"base": str(venv), "platbase": str(venv)})["purelib"])
    if not purelib.is_dir():  # the layout of the venv's own interpreter
        purelib = next(venv.glob("lib/python*/site-packages"))
    # addsitedir (not a bare path line) so the worker's own .pth files (an editable flock) apply
    parent = sorted({sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]})
    (purelib / "_flock_worker.pth").write_text(
        "".join(f"import site; site.addsitedir({p!r})\n" for p in parent))
    # from the extracted bundle: `./dep.whl` or `-e ./pkg` in requirements.txt is the bundle's
    if uv:
        await _run([uv, "pip", "install", "--quiet", "--python", str(py), "-r", REQUIREMENTS],
                   timeout, cwd=folder)
    else:
        await _run([str(py), "-m", "pip", "install", "--quiet", "-r", REQUIREMENTS], timeout,
                   cwd=folder)
    (venv / ".flock-ready").write_text("ok\n")
    return str(py)


class ToolsManager:
    def __init__(self, root: Path, *, swarm: str, type_name: str, fetch: Fetch | None,
                 data_dir: Path | None = None) -> None:
        self.root = Path(root)
        self.swarm, self.type = swarm, type_name
        self.fetch = fetch
        self.data_dir = Path(data_dir) if data_dir else self.root.parent / "_tools_data"
        self.version: str | None = None  # the version current() serves
        self.select: list[str] | str | None = None
        self._manifest: Manifest | None = None
        self._python: str | None = None
        self.wanted: tuple[str | None, list[str] | str | None] | None = None
        self.error: str | None = None
        self._task: asyncio.Task | None = None
        self._ready: dict[str, tuple[Manifest, str | None]] = {}
        self._lock = asyncio.Lock()

    def current(self) -> Toolset:
        """What a turn starting now gets (selection applied; unknown names dropped, see `error`)."""
        tools = ()
        if self._manifest is not None and self.select is not None:
            names = self._manifest.names() if self.select == "all" else self.select
            tools = tuple(t for n in names if (t := self._manifest.get(n)) is not None)
        return Toolset(version=self.version, dir=self.root / self.version if self.version else None,
                       swarm=self.swarm, type=self.type, data_dir=self.data_dir,
                       python=self._python, tools=tools, error=self.error)

    def want(self, version: str | None, select: list[str] | str | None) -> asyncio.Task | None:
        """Record what the controller says is current. Same version: the selection applies at
        once. A new version: fetched in the background; returns that task (or None)."""
        self.wanted = (version, select)
        if version is None:
            self.version, self.select, self._manifest, self._python, self.error = None, None, None, None, None
            return None
        if version == self.version:
            self.select = select
            return None
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._switch(), name="tools-fetch")
        return self._task

    async def ensure(self, version: str | None, select: list[str] | str | None) -> None:
        """`want`, then wait for the fetch (on assign: before the agent's first turn)."""
        task = self.want(version, select)
        if task is not None:
            await asyncio.shield(task)

    async def _switch(self) -> None:
        while self.wanted is not None and self.wanted[0] not in (None, self.version):
            version = self.wanted[0]
            try:
                manifest, python = await self._prepare(version)
            except Exception as e:  # noqa: BLE001 - keep serving the old version
                self.error = f"tools version {version}: {e}"
                log.warning("%s", self.error)
                return
            if self.wanted is not None and self.wanted[0] == version:
                self.version, self._manifest, self._python = version, manifest, python
                self.select, self.error = self.wanted[1], None
                log.info("tools version %s ready (%d tools)", version, len(manifest.tools))
            else:
                self._ready[version] = (manifest, python)

    async def _prepare(self, version: str) -> tuple[Manifest, str | None]:
        async with self._lock:
            if version in self._ready:
                return self._ready[version]
            dest = self.root / version
            if not dest.is_dir():
                if self.fetch is None:
                    raise ToolsError("no controller to fetch it from")
                data = await self.fetch(version)
                self.root.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(extract, data, dest, version)
            manifest = load_extracted(dest)
            python = await make_venv(dest, self.root / f"{version}.venv") if _requirements(dest) else None
            self._ready[version] = (manifest, python)
            return manifest, python


def tools_root(work_root: Path) -> Path:
    """Outside every agent workdir (agent names start with a letter)."""
    return Path(os.environ.get("FLOCK_TOOLS_ROOT") or Path(work_root) / "_tools")
