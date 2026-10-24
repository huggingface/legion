"""Finish the rendered *.local.yaml specs in the job (bootstrap.sh `render`), then print the smoke
settings in effect (JSON, one line) or nothing.

- CHATBOT_SMOKE=1: job/smoke.json's loop and budget on every rendered spec.
- Always: the kickoff states the real duration (the loop's `while.until`) and the phase split
  (CHATBOT_SWITCH_AT when set, else half), in place of the real run's "4 hours", "2 h".

usage: render.py <experiment dir>"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

from flock.models import load_spec_yaml, parse_duration


def smoke(exp: Path) -> dict | None:
    if not os.environ.get("CHATBOT_SMOKE"):
        return None
    o = json.loads((exp / "job" / "smoke.json").read_text())
    for f in (exp / "spec.local.yaml", exp / "phase2" / "spec.local.yaml"):
        d = load_spec_yaml(f.read_text())
        for t in d["types"].values():
            if t.get("loop"):
                lp = t["loop"] = {**t["loop"], "every": o["every"]}
                lp["while"] = {**lp["while"], **o["while"]}
                lp["final"] = {**lp["final"], "reserve_usd": o["reserve_usd"]}
        d["limits"] = {**d["limits"], **o["limits"]}
        f.write_text(yaml.safe_dump(d, sort_keys=False))
    return o


def _human(s: float) -> str:
    if s >= 3600:
        h = round(s / 3600, 1)
        return f"{h:g} hours" if h != 1 else "1 hour"
    return f"{round(s / 60):g} min"


def kickoff(exp: Path) -> None:
    d = load_spec_yaml((exp / "spec.local.yaml").read_text())
    until = next((t["loop"]["while"].get("until") for t in d["types"].values()
                  if t.get("loop") and t["loop"].get("while")), None)
    if not until:
        return
    total = parse_duration(until).total_seconds()
    sw = os.environ.get("CHATBOT_SWITCH_AT")
    p1 = min(float(sw), total) if sw else total / 2
    k = exp / "prompts" / "kickoff.md"

    def short(s: float) -> str:
        return _human(s).replace(" hours", " h").replace(" hour", " h")

    k.write_text(k.read_text().replace(" 4 hours,", f" {_human(total)},")
                 .replace("(first 2 h)", f"(first {short(p1)})")
                 .replace("(last 2 h)", f"(last {short(total - p1)})"))


if __name__ == "__main__":
    exp = Path(sys.argv[1])
    o = smoke(exp)
    kickoff(exp)
    if o is not None:
        print(json.dumps(o, separators=(",", ":")))
