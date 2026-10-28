"""Run 3's wave runner: agents join on a schedule (waves.yaml) by raising the spec's replicas
and applying the folder with `flock update` (the same path as a live update), up to a cap.

    waves.py plan                              # the schedule: when, how many in total
    waves.py status --spec S                   # the current count on the controller, next wave
    waves.py tick   --spec S [--start T]       # apply the wave that is due, if any (bootstrap)
    waves.py run    --spec S [--interval 30]   # tick until the schedule is over (jobs mode)
    waves.py add N  --spec S [--stagger 10]    # N more agents now (operator, after the schedule)
    waves.py add N  --spec S --type agent-pi   # N more of one type only (the others unchanged)
    waves.py set N  --spec S [--type T]        # exactly N (the operator's override, may lower)
    waves.py pin    --spec S                   # write the controller's replicas into S (no apply)

The controller is the one `flock` reaches: FLOCK_URL / FLOCK_TOKEN (jobs mode, from the operator
session), else the local run's .flock/current (the single-job bootstrap runs it in the
experiment folder). S is the rendered spec (`spec.local.yaml`, render.py): its
`types.<type>.replicas` are set, then S is applied as `flock update S` does, so prompts,
tools and everything else in the folder go with it. The count on the controller is the truth: a wave never lowers it, so an operator's `flock scale` up stays. Replica types outside the
schedule (agent-pi) are written into S as the controller runs them, so a wave never resets them.
A wave with `stagger_s` (waves.yaml: 10 s for wave 1 by default, 0 after) raises the count
one agent at a time, that many seconds apart; `tick` and `run` block meanwhile. Every change is
one JSON line in --log (default: <S>.waves.jsonl). The start time is kept in
<S>.waves.json (`--start` epoch seconds overrides it)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def seconds(v: object) -> float:
    """0, 90, "30s", "5m", "1h", "1h20m" -> seconds."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    parts = re.findall(r"(\d+(?:\.\d+)?)([smhd])", s)
    if not parts or "".join(a + b for a, b in parts) != s:
        if re.fullmatch(r"\d+(\.\d+)?", s):
            return float(s)
        raise ValueError(f"bad duration {v!r} (e.g. 0, 30s, 5m, 1h)")
    return sum(float(a) * UNITS[b] for a, b in parts)


@dataclass
class Schedule:
    events: list[tuple[float, int]]  # (seconds from start, total agents after it), in order
    cap: int
    type: str = "agent"
    think_type: str | None = None
    think_share: float = 0.0
    raw: dict = field(default_factory=dict)
    staggers: list[float] = field(default_factory=list)  # per event: seconds between its agents

    def stagger(self, elapsed: float) -> float:
        """The stagger of the latest event due at `elapsed` (0 before the first)."""
        due = [s for (t, _), s in zip(self.events, self.staggers) if t <= elapsed]
        return due[-1] if due else 0.0

    def target(self, elapsed: float) -> int:
        """Agents the schedule wants `elapsed` seconds after the start."""
        return min(self.cap, max([n for t, n in self.events if t <= elapsed], default=0))

    def next(self, elapsed: float) -> tuple[float, int] | None:
        return next(((t, n) for t, n in self.events if t > elapsed), None)

    def split(self, total: int) -> dict[str, int]:
        """The total by type: think_share of it (rounded) on think_type."""
        think = round(total * self.think_share) if self.think_type else 0
        return {self.type: total - think, **({self.think_type: think} if self.think_type else {})}


def parse(doc: dict | str) -> Schedule:
    d = yaml.safe_load(doc) if isinstance(doc, str) else dict(doc)
    cap = int(d.get("cap") or 1000)
    share = float(d.get("think_share") or 0)
    if not 0 <= share <= 1:
        raise ValueError("think_share must be 0 to 1")
    events, staggers, t, total = [], [], 0.0, 0
    first = float(d.get("stagger_s", 10))  # wave 1's default; later waves default to 0
    for i, w in enumerate(d.get("waves") or []):
        gap = float(w.get("stagger_s", first if i == 0 else 0))
        if gap < 0:
            raise ValueError(f"wave {w}: stagger_s must be >= 0")
        add = int(w["add"])
        if add <= 0:
            raise ValueError(f"wave {w}: add must be > 0")
        if "every" in w:
            step, until = seconds(w["every"]), int(w.get("until") or cap)
            t = seconds(w["from"]) if "from" in w else t
            if step <= 0:
                raise ValueError(f"wave {w}: every must be > 0")
            while total < min(until, cap):
                t, total = t + step, min(total + add, until, cap)
                events.append((t, total))
                staggers.append(gap)
        else:
            at = seconds(w["at"])
            if events and at < events[-1][0]:
                raise ValueError(f"wave {w}: waves must be in time order")
            t, total = at, min(total + add, cap)
            events.append((t, total))
            staggers.append(gap)
    return Schedule(events, cap, d.get("type") or "agent", d.get("think_type"), share, d,
                    staggers)


def load(path: Path | str = HERE / "waves.yaml") -> Schedule:
    return parse(Path(path).read_text())


def _clock(s: float) -> str:
    h, m, sec = int(s // 3600), int(s % 3600 // 60), int(s % 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


# ---------------------------------------------------------------------- the spec file
def write_replicas(spec: Path, counts: dict[str, int]) -> None:
    """Set `types.<type>.replicas` in the rendered spec (a generated file: layout is not kept)."""
    from flock.models import load_spec_yaml

    d = load_spec_yaml(spec.read_text())
    for t, n in counts.items():
        if t not in d["types"]:
            raise SystemExit(f"{spec}: no type {t!r} (waves.yaml type / think_type, --type)")
        d["types"][t]["replicas"] = n
    spec.write_text(yaml.safe_dump(d, sort_keys=False, width=100))


def current(c, swarm: str, types: list[str]) -> dict[str, int]:
    """The replicas the controller runs now, by type."""
    from flock.cli import _request

    r = _request(c, "GET", f"/swarms/{swarm}")
    if r.status_code >= 400:
        raise SystemExit(f"GET /swarms/{swarm}: {r.status_code} {r.text[:300]}")
    ts = r.json()["spec"]["types"]
    return {t: int((ts.get(t) or {}).get("replicas") or 0) for t in types}


def spec_types(spec: Path) -> list[str]:
    """The spec's replica types (not ephemeral: those are spawned, never replicas)."""
    from flock.models import load_spec_yaml

    return [t for t, d in load_spec_yaml(spec.read_text())["types"].items()
            if not d.get("ephemeral")]


def apply(c, spec: Path, sched: Schedule, total: int, why: str, log: Path,
          lower: bool = False, stagger: float = 0.0, only: str | None = None) -> dict:
    """Make the total `total` (capped; never lower unless `lower`), apply the spec, log it.
    `stagger` > 0 raises the count one agent at a time, that many seconds apart (each agent is
    created, and gets its first prompt, on its own), so a wave does not hit the board at once.
    `only`: `total` is that one type's count (the schedule's types keep theirs; the cap counts
    them too). Every other replica type is written as the controller runs it, so applying the
    spec never changes it (e.g. agent-pi agents an operator added stay)."""
    from flock.cli import SpecError, load_spec, update_swarm

    swarm = load_spec(str(spec)).swarm
    types = [only] if only else list(sched.split(0))
    keep = [t for t in spec_types(spec) if t not in types]
    others = current(c, swarm, keep)
    before = current(c, swarm, types)
    have = sum(before.values())
    room = sched.cap - (sum(others[t] for t in sched.split(0) if t in others) if only else 0)
    total = max(0, min(room, total if lower else max(total, have)))
    split = (lambda k: {only: k}) if only else sched.split
    counts = split(total)
    rec = {"ts": round(time.time(), 3), "why": why, "before": before, "after": counts,
           "total": total, "cap": sched.cap}
    if counts == before:
        write_replicas(spec, {**others, **counts})
        rec["lines"] = ["no changes"]
        return rec
    steps = range(have + 1, total + 1) if stagger > 0 and total > have else [total]
    rec["stagger_s"], rec["steps"] = (stagger if len(steps) > 1 else 0), len(steps)
    for i, k in enumerate(steps):
        if i:
            time.sleep(stagger)
        write_replicas(spec, {**others, **split(k)})
        try:
            rec["lines"] = update_swarm(c, str(spec))
        except SpecError as e:
            rec["error"] = e.lines
            break
    with open(log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def _has_state(spec: Path) -> bool:
    f = spec.with_name(spec.name + ".waves.json")
    try:
        return "start" in json.loads(f.read_text())
    except (OSError, ValueError):
        return False


def _state(spec: Path, start: float | None) -> float:
    f = spec.with_name(spec.name + ".waves.json")
    d = json.loads(f.read_text()) if f.is_file() else {}
    if start is not None or "start" not in d:
        d["start"] = start if start is not None else time.time()
        f.write_text(json.dumps(d) + "\n")
    return float(d["start"])


def _say(rec: dict) -> None:
    head = (f"wave ({rec['why']}): {sum(rec['before'].values())} -> {rec['total']} agents "
            f"{rec['after']} (cap {rec['cap']}"
            + (f"; {rec['steps']} steps {rec['stagger_s']:g} s apart" if rec.get("stagger_s")
               else "") + ")")
    lines = [x for x in rec.get("lines", []) if not x.startswith("world.")]  # controller-written
    print(head + ("; FAILED: " + " / ".join(rec["error"]) if "error" in rec else
                  "; " + " / ".join(lines)), flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["plan", "status", "tick", "run", "add", "set", "pin"])
    ap.add_argument("n", nargs="?", type=int, help="add / set: how many")
    ap.add_argument("--spec", type=Path, help="the rendered spec (spec.local.yaml)")
    ap.add_argument("--waves", type=Path, default=HERE / "waves.yaml")
    ap.add_argument("--start", type=float, help="epoch seconds of the run's start")
    ap.add_argument("--interval", type=float, default=30.0, help="run: seconds between ticks")
    ap.add_argument("--log", type=Path, help="default: <spec>.waves.jsonl")
    ap.add_argument("--type", dest="only", metavar="TYPE",
                    help="add / set: change only this type's replicas (e.g. agent-pi); the "
                         "schedule's types keep theirs")
    ap.add_argument("--stagger", type=float, default=0.0,
                    help="add / set: seconds between the new agents (schedule: stagger_s)")
    a = ap.parse_args(argv)
    sched = load(a.waves)
    if a.cmd == "plan":
        for t, n in sched.events:
            print(f"{_clock(t):>8}  {min(n, sched.cap):>5}  {sched.split(min(n, sched.cap))}")
        print(f"cap {sched.cap}; after the last wave: `waves.py add N`")
        return 0
    if a.spec is None:
        ap.error("--spec is required")
    if a.cmd in ("add", "set") and (a.n is None or a.n < 0):
        ap.error(f"{a.cmd} needs a number >= 0")
    from flock.cli import client, load_spec

    spec, log = a.spec.resolve(), a.log or a.spec.with_name(a.spec.name + ".waves.jsonl")
    swarm = load_spec(str(spec)).swarm
    with client() as c:
        if a.cmd == "pin":
            write_replicas(spec, current(c, swarm, spec_types(spec)))
            return 0
        if a.cmd in ("add", "set"):
            if a.only and a.only not in spec_types(spec):
                ap.error(f"--type {a.only}: not a replica type of {spec.name}")
            have = sum(current(c, swarm, [a.only] if a.only else list(sched.split(0))).values())
            why = f"{a.cmd} {a.n}" + (f" --type {a.only}" if a.only else "")
            rec = apply(c, spec, sched, have + a.n if a.cmd == "add" else a.n, why, log,
                        lower=a.cmd == "set", stagger=a.stagger, only=a.only)
            _say(rec)
            return 1 if "error" in rec else 0
        if a.cmd == "run" and a.start is None and not _has_state(spec):
            # the clock starts at the first answer from the controller, not at our own start:
            # on a cold node the controller can come up after the second wave is due, and the
            # first tick would apply both waves as one
            waited = 0.0
            while True:
                try:
                    current(c, swarm, list(sched.split(0)))
                    break
                except Exception as e:  # noqa: BLE001 - not up yet (or not applied yet)
                    if waited % 60 == 0:
                        print(f"waiting for the controller ({type(e).__name__}); the waves "
                              "clock starts when it answers", flush=True)
                    time.sleep(5)
                    waited += 5
        fresh = a.start is None and not _has_state(spec)
        start = _state(spec, a.start)
        if fresh:
            print(f"waves clock started at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start))}",
                  flush=True)
        while True:
            elapsed = time.time() - start
            have, nxt = current(c, swarm, list(sched.split(0))), sched.next(elapsed)
            if a.cmd == "status":
                print(json.dumps({"swarm": swarm, "replicas": have, "total": sum(have.values()),
                                  "cap": sched.cap, "elapsed_min": round(elapsed / 60, 1),
                                  "schedule_now": sched.target(elapsed),
                                  "next": {"in_min": round((nxt[0] - elapsed) / 60, 1),
                                           "total": nxt[1]} if nxt else None}))
                return 0
            due = sched.target(elapsed)
            if due > sum(have.values()):
                rec = apply(c, spec, sched, due, f"schedule {_clock(elapsed)}", log,
                            stagger=sched.stagger(elapsed))
                _say(rec)
                if a.cmd == "tick":
                    return 1 if "error" in rec else 0
            elif a.cmd == "tick":
                return 0
            if nxt is None:
                print(f"schedule over at {sum(have.values())} agents: `waves.py add N` from "
                      "here", flush=True)
                return 0
            time.sleep(max(1.0, min(a.interval, nxt[0] - elapsed)))


if __name__ == "__main__":
    sys.exit(main())
