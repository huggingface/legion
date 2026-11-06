"""Fakes for worker tests: a scripted harness adapter and an in-memory bucket."""

from __future__ import annotations

import asyncio
import fnmatch
import shutil
import threading
import time
from pathlib import Path

from flock.models import TokenUsage, TypeSpec
from flock.worker.adapters.base import AdapterEvent, EventQueue, ResumeError
from flock.worker.agents import AgentRuntime
from flock.worker.events import Outbox
from flock.worker.sync import BucketSync


def spec(**kw) -> TypeSpec:
    base = {"harness": "claude", "model": "m-1", "idle_grace": "200ms", "turn_timeout": "2s",
            "drain_window": "1s"}
    return TypeSpec.model_validate({**base, **kw})


class ScriptedAdapter:
    """Pretends to be a harness: each forwarded prompt outside a turn starts a turn that ends
    `turn_time` later with one assistant message and cumulative usage; prompts during a turn
    are folded into it. `hang=True` turns never end. A resume needs `<home>/<sid>.jsonl`."""

    harness = "claude"

    def __init__(self, home: Path, *, start_delay: float = 0.0, turn_time: float = 0.05,
                 hang: bool = False, log: list | None = None) -> None:
        self.home = Path(home)
        self.start_delay, self.turn_time, self.hang = start_delay, turn_time, hang
        self.starts: list[str | None] = []
        self.envs: list[dict] = []
        self.sent: list[str] = []
        self.stops = 0
        self.live = False
        self.sid: str | None = None
        self.log = log if log is not None else []
        self._q = EventQueue()
        self._turn: asyncio.Task | None = None
        self._cum = TokenUsage()

    async def start(self, workdir, session_id, spec, *, env=None):
        self.log.append(("start", session_id))
        self.starts.append(session_id)
        self.envs.append(dict(env or {}))
        if session_id and not (self.home / f"{session_id}.jsonl").exists():
            raise ResumeError(f"no transcript for {session_id}")
        await asyncio.sleep(self.start_delay)
        self._q = EventQueue()
        self.live = True
        self.sid = session_id or f"s-{len(self.starts)}"
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / f"{self.sid}.jsonl").write_text("{}\n")
        return self.sid

    async def send(self, text):
        if not self.live:
            raise RuntimeError("not running")
        self.sent.append(text)
        self.log.append(("send", text))
        if self._turn is None:
            self._q.put(AdapterEvent("turn_started"))
            self._turn = asyncio.create_task(self._finish())
        self._q.put(AdapterEvent("message", message={"role": "user", "content": text}))

    async def _finish(self):
        if self.hang:
            await asyncio.Event().wait()
        await asyncio.sleep(self.turn_time)
        self._q.put(AdapterEvent("message", message={"role": "assistant", "model": "m-1",
                                                     "content": [{"type": "text", "text": "ok"}]}))
        self._cum = self._cum + TokenUsage(input=10, output=5, cache_read=100)
        self._q.put(AdapterEvent("usage", model="m-1", usage=self._cum, cumulative=True, cost_usd=0.01))
        self._q.put(AdapterEvent("turn_ended", status="completed"))
        self._turn = None

    async def stop(self):
        if not self.live:
            return
        self.live = False
        self.stops += 1
        self.log.append(("stop", None))
        if self._turn:
            self._turn.cancel()
            self._turn = None
        self._q.put(AdapterEvent("exited"))

    async def events(self):
        async for ev in self._q:
            yield ev

    def session_files(self, session_id):
        p = self.home / f"{session_id}.jsonl"
        return [p] if session_id and p.exists() else []


class MemoryStore:
    """BucketStore over a dict of {remote path: bytes}, with each file's mtime (seconds) as the
    Hub keeps it. `sync_up` behaves like `sync_bucket`: uploads files whose size or mtime
    differ, deletes remote files missing locally only with `delete`, skips `exclude` matches."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.mtimes: dict[str, float] = {}
        self.uploads: list[list[str]] = []
        self.synced: list[list[str]] = []  # per sync_up call: the paths it uploaded

    def upload(self, files):
        self.uploads.append([r for _, r in files])
        for p, r in files:
            self.files[r] = Path(p).read_bytes()
            self.mtimes[r] = Path(p).stat().st_mtime

    def list(self, prefix):
        return sorted(k for k in self.files if k.startswith(prefix))

    def first(self, prefix):
        return next(iter(self.list(prefix)), None)

    def list_info(self, prefix):
        return [(k, self.mtimes.get(k)) for k in self.list(prefix)]

    def download(self, files):
        for r, p in files:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_bytes(self.files[r])

    def sync_up(self, local, prefix, *, delete=True, exclude=None):
        prefix = prefix.rstrip("/") + "/"
        keep = lambda rel: not any(fnmatch.fnmatchcase(rel, pat) for pat in exclude or ())
        mine = {prefix + p.relative_to(local).as_posix(): p for p in Path(local).rglob("*")
                if p.is_file() and keep(p.relative_to(local).as_posix())}
        if delete:
            for k in [k for k in self.files if k.startswith(prefix) and k not in mine
                      and keep(k[len(prefix):])]:
                del self.files[k]
                self.mtimes.pop(k, None)
        sent = []
        for k, p in mine.items():
            st = p.stat()
            if (k in self.files and len(self.files[k]) == st.st_size
                    and abs(self.mtimes.get(k, 0) - st.st_mtime) <= 1e-3):
                continue
            self.files[k] = p.read_bytes()
            self.mtimes[k] = st.st_mtime
            sent.append(k)
        self.synced.append(sent)

    def sync_down(self, prefix, local):
        prefix = prefix.rstrip("/") + "/"
        self.download([(k, Path(local) / k[len(prefix):]) for k in self.list(prefix)])

    def delete(self, paths):
        for p in paths:
            self.files.pop(p, None)
            self.mtimes.pop(p, None)


class RecordingSync(BucketSync):
    def __init__(self, store: MemoryStore, swarm: str = "sw", log: list | None = None) -> None:
        super().__init__(store, swarm)
        self.pushed: list[str] = []
        self.restored: list[str] = []
        self.log = log if log is not None else []

    async def push(self, rt, **kw):
        self.pushed.append(rt.name)
        self.log.append(("push", rt.name))
        return await super().push(rt, **kw)

    async def restore(self, rt):
        self.restored.append(rt.name)
        self.log.append(("restore", rt.name))
        return await super().restore(rt)


def runtime(tmp_path: Path, name: str = "researcher-0001", *, adapter=None, sync=None, **spec_kw):
    adapter = adapter or ScriptedAdapter(tmp_path / "home")
    rt = AgentRuntime(name, "researcher", spec(**spec_kw), swarm="sw", work_root=tmp_path / "agents",
                      adapter=adapter, outbox=Outbox(), sync=sync)
    return rt, adapter


async def until(cond, timeout: float = 5.0, step: float = 0.01) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(step)


def event_types(rt: AgentRuntime) -> list[str]:
    return [e.type for e in list(rt.outbox._q)]


def copy_fixture(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


class GatedStore(MemoryStore):
    """Downloads block until `gate` is set (a long restore wave), counting the waiters."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.waiting = 0

    def download(self, files):
        self.waiting += 1
        if not self.gate.wait(10):
            raise RuntimeError("gate never opened")
        super().download(files)


class HangStore(MemoryStore):
    """`sync_up` of a prefix containing `hang` blocks until `gate` is set (a hung xet commit);
    `order` records each call ("upload", "sync_up:<prefix>")."""

    def __init__(self, hang: str = "") -> None:
        super().__init__()
        self.hang, self.gate, self.order = hang, threading.Event(), []

    def upload(self, files):
        self.order.append("upload")
        super().upload(files)

    def sync_up(self, local, prefix, **kw):
        self.order.append(f"sync_up:{prefix}")
        if self.hang in prefix:
            self.gate.wait(10)
        super().sync_up(local, prefix, **kw)
