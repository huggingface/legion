"""A Harbor run's tools and resources folders on the worker (flock.worker.harbor).

`flock launch` packs both folders as one tools bundle (flock.harbor.spec.pack_folders) and
uploads it to the controller; `folder(version)` fetches that version once
(`GET /swarms/{swarm}/tools/{version}`), checks it against its hash and extracts it to
`<root>/<version>/`.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from flock.tools.bundle import extract
from flock.tools.loader import ToolsError

Fetch = Callable[[str], Awaitable[bytes]]


class ToolsManager:
    def __init__(self, root: Path, *, fetch: Fetch | None) -> None:
        self.root = Path(root)
        self.fetch = fetch
        self._lock = asyncio.Lock()

    async def folder(self, version: str) -> Path:
        """Version `version` unpacked on this worker (fetched once)."""
        async with self._lock:
            dest = self.root / version
            if not dest.is_dir():
                if self.fetch is None:
                    raise ToolsError("no controller to fetch it from")
                data = await self.fetch(version)
                self.root.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(extract, data, dest, version)
            return dest


def tools_root(work_root: Path) -> Path:
    """Outside every agent workdir (agent names start with a letter)."""
    return Path(os.environ.get("FLOCK_TOOLS_ROOT") or Path(work_root) / "_tools")
