"""External-write lanes for worlds: bounded concurrency, a circuit breaker and a persistent outbox.

A world that writes to an outside service (a Git host, an API with one token) gives each kind
of write its own lane, so one kind never starves another:

    lanes = Lanes({"push": Lane(slots=1), "merge": Lane(slots=1),
                   "comment": Lane(slots=1, gap_s=1.0)}, dir=state_dir)
    with lanes.hold("push", deadline=time.monotonic() + 20) as slot:   # LaneBusy past deadline
        ...write...; slot.ok()        # or slot.fail("transient") / slot.fail("final")
    lanes.outbox("comment", drain=post).put(key=pr, payload=text)   # persistent, never drops

`Lane.slots`, `Lane.gap_s` and `Lane.paused` are plain attributes (`Lane.configure()` changes
the first two and wakes waiters); the breaker never changes them, it only lowers what is
effective while tripped.
"""
from __future__ import annotations

import collections
import contextlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger("flock.world.lanes")
POLL_S = 0.25  # longest single wait: notices pause flips, configure() and a test's fake clock


class LaneBusy(Exception):
    """No slot within the deadline; `reason` is "slots", "gap" or "paused"."""

    def __init__(self, lane: str, reason: str, retry_after_s: float) -> None:
        super().__init__(f"lane {lane} busy ({reason}); retry in {retry_after_s:.0f} s")
        self.lane, self.reason, self.retry_after_s = lane, reason, retry_after_s


class Lane:
    """One kind of external write.

    Breaker: `trip_after` transient failures within `trip_window_s` trip the lane to
    min(slots, 1) slot and at least `trip_gap_s` between writes; after `quiet_s` without a
    transient failure it returns to `slots` and `gap_s`."""

    def __init__(self, slots: int = 1, gap_s: float = 0.0, *, trip_after: int = 3,
                 trip_window_s: float = 900.0, trip_gap_s: float = 5.0, quiet_s: float = 600.0,
                 retry_after_s: float = 60.0) -> None:
        self.slots, self.gap_s, self.paused = slots, gap_s, False
        self.trip_after, self.trip_window_s = trip_after, trip_window_s
        self.trip_gap_s, self.quiet_s, self.retry_after_s = trip_gap_s, quiet_s, retry_after_s
        self.name, self.clock, self.paused_fn = "", time.monotonic, None  # set by Lanes
        self.cond = threading.Condition()
        self.in_flight, self.last_at = 0, float("-inf")
        self.failures = collections.deque()  # clock times of recent transient failures
        self.tripped = False
        self.stats = {"waits_total": 0, "wait_s_total": 0.0, "ok": 0, "transient": 0, "final": 0}

    def configure(self, slots: int | None = None, gap_s: float | None = None) -> None:
        with self.cond:
            if slots is not None:
                self.slots = slots
            if gap_s is not None:
                self.gap_s = gap_s
            self.cond.notify_all()

    def is_paused(self) -> bool:
        return self.paused or bool(self.paused_fn and self.paused_fn(self.name))

    def _effective(self, now):
        """(slots, gap_s) in force now (cond held); ends a trip after quiet_s."""
        if self.tripped and now - self.failures[-1] >= self.quiet_s:
            self.tripped = False
            self.failures.clear()
            log.warning("lane %s: recovered after %.0f quiet s; slots=%s gap_s=%s",
                        self.name, self.quiet_s, self.slots, self.gap_s)
        if self.tripped:
            return min(self.slots, 1), max(self.gap_s, self.trip_gap_s)
        return self.slots, self.gap_s

    def acquire(self, deadline: float | None, pausable: bool = True) -> None:
        with self.cond:
            start = now = self.clock()
            while True:
                slots, gap = self._effective(now)
                reason = ("paused" if pausable and self.is_paused() else
                          "slots" if self.in_flight >= slots else
                          "gap" if now < self.last_at + gap else None)
                if reason is None:
                    break
                if deadline is not None and now >= deadline:
                    self.stats["waits_total"] += 1
                    self.stats["wait_s_total"] += now - start
                    retry = (self.last_at + gap - now) if reason == "gap" else self.retry_after_s
                    raise LaneBusy(self.name, reason, max(1.0, retry))
                self.cond.wait(POLL_S if deadline is None else min(POLL_S, deadline - now))
                now = self.clock()
            if now > start:
                self.stats["waits_total"] += 1
                self.stats["wait_s_total"] += now - start
            self.in_flight += 1
            self.last_at = now

    def release(self) -> None:
        with self.cond:
            self.in_flight -= 1
            self.cond.notify_all()

    def report(self, kind: str) -> None:
        """Count an outcome: "ok", "transient" (feeds the breaker) or "final"."""
        with self.cond:
            self.stats[kind] += 1
            if kind != "transient":
                return
            now = self.clock()
            self.failures.append(now)
            while self.failures and self.failures[0] < now - self.trip_window_s:
                self.failures.popleft()
            if not self.tripped and len(self.failures) >= self.trip_after:
                self.tripped = True
                log.warning("lane %s: tripped after %d transient failures in %.0f s; 1 slot, "
                            "%.0f s gap until %.0f quiet s", self.name, len(self.failures),
                            self.trip_window_s, max(self.gap_s, self.trip_gap_s), self.quiet_s)

    def metrics(self) -> dict:
        with self.cond:
            slots, _ = self._effective(self.clock())
            return {"slots": self.slots, "effective_slots": slots, "in_flight": self.in_flight,
                    **self.stats, "tripped": int(self.tripped), "paused": int(self.is_paused())}


class Slot:
    """A held slot: report the write's outcome with ok() or fail()."""

    def __init__(self, lane: Lane) -> None:
        self.lane = lane

    def ok(self) -> None:
        self.lane.report("ok")

    def fail(self, kind: str = "transient") -> None:
        if kind not in ("transient", "final"):
            raise ValueError("kind must be transient or final")
        self.lane.report(kind)


class Outbox:
    """Writes for one lane that must happen eventually, drained by a background thread.

    `drain(key, payloads)` makes one write of the queued payloads of `key` (more than one when
    they piled up: the world joins them) and returns "ok", "transient" or "final" (an exception
    counts as transient). Transient failures retry forever with capped doubling backoff; only
    ok and final remove payloads. Stored as JSON lines in `path` (keys: str or int), compacted
    on load and when mostly settled."""

    def __init__(self, lanes: Lanes, name: str, path: Path, drain, backoff=(5.0, 600.0),
                 max_batch: int = 10) -> None:
        self.lanes, self.name, self.path, self.drain = lanes, name, path, drain
        self.backoff, self.max_batch = backoff, max_batch
        self.lock = threading.Lock()
        self.entries = {}  # key -> {"payloads", "queued_at", "tries", "next_at"} in put order
        self.lines = 0  # lines in the file
        self._load()
        self.stop, self.wake = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"lane-{name}-outbox", daemon=True)
        if drain is not None:
            self.thread.start()

    def _entry(self, key, ts):
        return self.entries.setdefault(key, {"payloads": [], "queued_at": ts, "tries": 0, "next_at": 0.0})

    def _load(self):
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    op = json.loads(line)
                    if "put" in op:
                        self._entry(op["put"], op["ts"])["payloads"].append(op["payload"])
                    elif op["done"] in self.entries:
                        entry = self.entries[op["done"]]
                        del entry["payloads"][:op["n"]]
                        if not entry["payloads"]:
                            del self.entries[op["done"]]
                except (ValueError, KeyError, TypeError):  # a line cut by a crash
                    log.warning("lane %s outbox: skipped an unreadable line", self.name)
        self._compact()

    def _compact(self):
        """Rewrite the file with the pending payloads only (lock held, or during __init__)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
        with open(tmp, "w") as out:
            for key, e in self.entries.items():
                out.writelines(json.dumps({"put": key, "payload": payload, "ts": e["queued_at"]}) + "\n" for payload in e["payloads"])
        os.replace(tmp, self.path)
        self.lines = sum(len(e["payloads"]) for e in self.entries.values())

    def _append(self, op):
        with open(self.path, "a") as out:
            out.write(json.dumps(op) + "\n")
        self.lines += 1

    def put(self, key, payload) -> None:
        """Queue payload for key; joins payloads of the same key still waiting."""
        with self.lock:
            ts = self.lanes.wall()
            self._append({"put": key, "payload": payload, "ts": ts})
            self._entry(key, ts)["payloads"].append(payload)
        self.wake.set()

    def depth(self) -> tuple[int, float]:
        """(pending payloads, age in seconds of the oldest)."""
        with self.lock:
            now = self.lanes.wall()
            return (sum(len(e["payloads"]) for e in self.entries.values()),
                    max((now - e["queued_at"] for e in self.entries.values()), default=0.0))

    def _settle(self, key, n, kind):
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                return
            if kind == "transient":
                entry["tries"] += 1
                delay = min(self.backoff[1], self.backoff[0] * 2 ** (entry["tries"] - 1))
                entry["next_at"] = self.lanes.wall() + delay
                log.warning("lane %s outbox: write for %r failed; retry %d in %.0f s",
                            self.name, key, entry["tries"], delay)
                return
            if kind == "final":
                log.warning("lane %s outbox: dropped %d payload(s) for %r: final", self.name, n, key)
            del entry["payloads"][:n]
            entry["tries"], entry["next_at"] = 0, 0.0
            if not entry["payloads"]:
                del self.entries[key]
            self._append({"done": key, "n": n})
            pending = sum(len(e["payloads"]) for e in self.entries.values())
            if self.lines > 2 * pending + 1000:
                self._compact()

    def _run(self):
        while not self.stop.is_set():
            with self.lock:
                now = self.lanes.wall()
                due = [(e["queued_at"], i, k) for i, (k, e) in enumerate(self.entries.items())
                       if e["next_at"] <= now]
                key = min(due)[2] if due else None
            if key is None:
                self.wake.wait(POLL_S)
                self.wake.clear()
                continue
            try:
                with self.lanes.hold(self.name, deadline=self.lanes.clock() + 1.0) as slot:
                    with self.lock:  # taken with the slot held: joins what piled up meanwhile
                        payloads = list(self.entries.get(key, {"payloads": []})["payloads"][:self.max_batch])
                    if not payloads:
                        continue
                    try:
                        kind = self.drain(key, payloads) or "ok"
                    except Exception:
                        log.exception("lane %s outbox: drain failed", self.name)
                        kind = "transient"
                    slot.ok() if kind == "ok" else slot.fail(kind)
            except LaneBusy:
                self.stop.wait(POLL_S)
                continue
            self._settle(key, len(payloads), kind)

    def close(self, timeout: float = 5.0) -> None:
        self.stop.set()
        self.wake.set()
        if self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout)


class Lanes:
    """Named lanes sharing a clock, an optional `paused(name)` callback and a state dir for
    outboxes (`<dir>/outbox-<name>.jsonl`)."""

    def __init__(self, lanes: dict[str, Lane], dir: str | Path | None = None,
                 clock=time.monotonic, wall=time.time, paused=None) -> None:
        self.lanes, self.dir, self.clock, self.wall = lanes, Path(dir) if dir else None, clock, wall
        self.outboxes: dict[str, Outbox] = {}
        for name, lane in lanes.items():
            lane.name, lane.clock, lane.paused_fn = name, clock, paused

    def __getitem__(self, name: str) -> Lane:
        return self.lanes[name]

    @contextlib.contextmanager
    def hold(self, name: str, deadline: float | None = None, pausable: bool = True):
        """Hold a slot of lane `name`, waiting until `deadline` (on `clock`; None: no limit);
        raises LaneBusy. `pausable=False` ignores the pause (e.g. closing what was opened)."""
        lane = self.lanes[name]
        lane.acquire(deadline, pausable)
        try:
            yield Slot(lane)
        finally:
            lane.release()

    def outbox(self, name: str, drain=None, **kwargs) -> Outbox:
        """The lane's outbox; the first call with `drain` starts its drainer thread."""
        if name not in self.outboxes:
            if self.dir is None:
                raise ValueError("Lanes(dir=...) is needed for an outbox")
            self.outboxes[name] = Outbox(self, name, self.dir / f"outbox-{name}.jsonl", drain, **kwargs)
        return self.outboxes[name]

    def metrics(self) -> dict[str, dict]:
        out = {}
        for name, lane in self.lanes.items():
            box = self.outboxes.get(name)
            depth, age = box.depth() if box else (0, 0.0)
            out[name] = {**lane.metrics(), "outbox_depth": depth, "outbox_oldest_age_s": age}
        return out

    def close(self, timeout: float = 5.0) -> None:
        """Stop the drainer threads; what is queued stays on disk for the next start."""
        end = time.monotonic() + timeout
        for box in self.outboxes.values():
            box.close(max(0.0, end - time.monotonic()))
