"""Runtime knobs and lane pauses of a world (docs/world.md, "Admin API").

A world registers knobs (`World.knob`) and consults pauses (`World.paused`); operators change
them through the server's `/admin` routes. Values set there and the paused lanes are written to
`knobs.json` in the world's state dir (`World.state_dir()`) and read back at the next start, so a
requeued world keeps them; the defaults a world passes (often from its environment) apply only
to knobs nobody set.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("flock.world")

KNOBS_FILE = "knobs.json"
LANE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
TYPES = (int, float, bool, str)


@dataclass
class Knob:
    name: str
    type: type
    default: Any
    min: float | None = None
    max: float | None = None
    doc: str = ""
    on_change: Callable[[Any], None] | None = None

    def check(self, value: Any) -> Any:
        """`value` as this knob's type (raises ValueError)."""
        t = self.type
        if t is bool or t is str:
            ok = isinstance(value, t)
        else:  # int or float: JSON numbers, not booleans; 2.0 is an int, 2.5 is not
            ok = isinstance(value, int | float) and not isinstance(value, bool) and (
                t is float or float(value).is_integer())
        if not ok:
            raise ValueError(f"{self.name}: expected {t.__name__}, got {json.dumps(value)}")
        value = t(value)
        if self.min is not None and value < self.min:
            raise ValueError(f"{self.name}: {value} is below the minimum {self.min:g}")
        if self.max is not None and value > self.max:
            raise ValueError(f"{self.name}: {value} is above the maximum {self.max:g}")
        return value

    def describe(self, value: Any) -> dict[str, Any]:
        return {"value": value, "default": self.default, "type": self.type.__name__,
                "min": self.min, "max": self.max, "doc": self.doc}


class Admin:
    """Knobs (name -> value) and paused lanes (lane -> reason), persisted in
    `<state dir>/knobs.json` when a state dir is given. Thread-safe."""

    def __init__(self, state_dir: str | os.PathLike | None = None) -> None:
        self.path: Path | None = None
        self.knobs: dict[str, Knob] = {}
        self.values: dict[str, Any] = {}
        self.stored: dict[str, Any] = {}  # values set through the admin API
        self.paused: dict[str, str] = {}
        self.lock = threading.RLock()
        if state_dir is not None:
            self.attach(state_dir)

    def attach(self, state_dir: str | os.PathLike) -> None:
        """Read `knobs.json` from `state_dir` and save there from now on. Lanes paused before
        (while the world could not be built) stay paused."""
        path = Path(state_dir) / KNOBS_FILE
        knobs: dict[str, Any] = {}
        paused: dict[str, str] = {}
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")  # noqa: TRY004 (handled below)
            knobs, paused = data.get("knobs") or {}, data.get("paused") or {}
            if not isinstance(knobs, dict) or not isinstance(paused, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in paused.items()):
                raise ValueError('expected {"knobs": {name: value}, "paused": {lane: reason}}')
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:  # never park the world over this file
            log.warning("%s: unreadable (%s); starting from the defaults", path, e)
            knobs, paused = {}, {}
        with self.lock:
            self.path = path
            self.stored = dict(knobs)
            self.paused = {**paused, **self.paused}

    def register(self, knob: Knob) -> Any:
        with self.lock:
            value = knob.check(knob.default)
            if knob.name in self.stored:
                try:
                    value = knob.check(self.stored[knob.name])
                except ValueError as e:
                    log.warning("%s: stored value ignored: %s", KNOBS_FILE, e)
            self.knobs[knob.name] = knob
            self.values[knob.name] = value
            return value

    def describe(self) -> dict[str, dict[str, Any]]:
        with self.lock:
            return {n: k.describe(self.values[n]) for n, k in self.knobs.items()}

    def set(self, updates: dict[str, Any]) -> list[dict[str, Any]]:
        """Validate every update, apply them (on_change callbacks included) and save, all or
        nothing. Returns [{name, old, new}]; raises ValueError (nothing changed) on a bad value
        or a failing on_change (those already applied get on_change(old) back)."""
        if not isinstance(updates, dict) or not updates:
            raise ValueError("expected {knob: value, ...}")
        with self.lock:
            unknown = sorted(set(updates) - set(self.knobs))
            if unknown:
                raise ValueError(f"unknown knob(s) {', '.join(unknown)}; knobs: "
                                 f"{', '.join(self.knobs) or 'none'}")
            checked = {n: self.knobs[n].check(v) for n, v in updates.items()}
            changes: list[dict[str, Any]] = []
            for n, v in checked.items():
                try:
                    if self.knobs[n].on_change is not None:
                        self.knobs[n].on_change(v)
                except Exception as e:  # noqa: BLE001 (any refusal rolls the patch back)
                    for c in reversed(changes):
                        try:
                            if self.knobs[c["name"]].on_change is not None:
                                self.knobs[c["name"]].on_change(c["old"])
                        except Exception:
                            log.exception("knob %s: restoring %r failed", c["name"], c["old"])
                    raise ValueError(f"{n}: the world refused {json.dumps(v)}: {e}") from None
                changes.append({"name": n, "old": self.values[n], "new": v})
            for c in changes:
                self.values[c["name"]] = self.stored[c["name"]] = c["new"]
            self._save()
            return changes

    def pause(self, lanes: list[str], reason: str = "") -> None:
        self._lanes(lanes)
        with self.lock:
            self.paused.update(dict.fromkeys(lanes, reason))
            self._save()

    def resume(self, lanes: list[str]) -> None:
        self._lanes(lanes)
        with self.lock:
            for lane in lanes:
                self.paused.pop(lane, None)
            self._save()

    @staticmethod
    def _lanes(lanes: Any) -> None:
        if not isinstance(lanes, list) or not lanes or not all(
                isinstance(x, str) and LANE_RE.match(x) for x in lanes):
            raise ValueError(f"lanes: expected a list of names matching {LANE_RE.pattern}")

    def _save(self) -> None:
        if self.path is None:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"knobs": self.stored, "paused": self.paused}, indent=1))
        tmp.replace(self.path)
