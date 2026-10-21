"""`flock-tool`: call a tools-folder tool from a CLI harness's shell (docs/tools.md).

    flock-tool NAME --json '{"a": 1}'     run NAME; prints the result, exit 1 on a tool error
    flock-tool NAME --json -              arguments from stdin
    flock-tool --list                     the tools of this turn, with their JSON schemas

The worker puts FLOCK_TOOLS_PIN (the agent's pin file: tools version, folder, selection) in the
harness env and rewrites the file at each turn start, so a call runs the version the turn
started with. The call itself is the same subprocess the `llm` harness uses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from flock.tools.toolset import PIN_ENV, read_pin


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flock-tool", description="Call a tools-folder tool.")
    ap.add_argument("name", nargs="?", help="the tool to call")
    ap.add_argument("--json", dest="args", default="{}",
                    help="the arguments as a JSON object ('-': read it from stdin)")
    ap.add_argument("--list", action="store_true", help="list the tools and their parameters")
    ns = ap.parse_args(argv)
    pin = os.environ.get(PIN_ENV)
    if not pin or not os.path.isfile(pin):
        print(json.dumps({"error": f"no tools here: {PIN_ENV} is not set or missing (does this "
                                   "agent's type select tools?)"}))
        return 2
    ts, agent, workdir = read_pin(pin)
    if ns.list:
        print(json.dumps({"version": ts.version, "tools": [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in ts.tools]}, indent=1))
        return 0
    if not ns.name:
        ap.print_usage(sys.stderr)
        return 2
    raw = sys.stdin.read() if ns.args == "-" else ns.args
    try:
        args = json.loads(raw or "{}")
    except ValueError as e:
        print(json.dumps({"error": f"--json is not valid JSON: {e}"}))
        return 2
    import asyncio

    r = asyncio.run(ts.call(ns.name, args, agent=agent, workdir=workdir, env=dict(os.environ)))
    print(r.output)
    return 1 if r.error else 0


if __name__ == "__main__":
    sys.exit(main())
