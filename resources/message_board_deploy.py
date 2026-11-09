"""Run msgboard.space.deploy() for the board resource in a process of its own.

deploy()'s Hub calls have no timeout, so the resource kills this process when its
startup limit runs out. The request comes as JSON on stdin, which keeps the tokens
out of the process list. The last line on stdout is {"deployment": {...}} or
{"error": "<type>: <message>"}.

Usage: python -P message_board_deploy.py LAUNCHER_PID. The process exits when the
launcher dies, so a killed launcher leaves no deploy running with its HF token.
"""

import contextlib
import dataclasses
import json
import os
import sys
import threading
import time


def exit_with(launcher):
    def watch():
        while os.getppid() == launcher:
            time.sleep(0.5)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def main():
    exit_with(int(sys.argv[1]))
    # Imported once the watch runs, since importing it takes a while.
    from msgboard.space import deploy

    request = json.loads(sys.stdin.read())
    try:
        # deploy() and huggingface_hub may print; stdout is kept for the result.
        with contextlib.redirect_stdout(sys.stderr):
            deployment = deploy(request.pop("space_id"), **request)
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        result = {"deployment": dataclasses.asdict(deployment)}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
