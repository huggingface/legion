#!/usr/bin/python3
"""Narrow Podman Compose 1.x compatibility for Harbor exec and cp."""

import os
import subprocess
import sys

args = sys.argv[1:]
if "cp" in args:
    i = args.index("cp")
    global_args = args[:i]
    copy_args = args[i + 1 :]
    project = None
    for j, a in enumerate(global_args):
        if a in ["--project-name", "-p"] and j + 1 < len(global_args):
            project = global_args[j + 1]
    if not project or len(copy_args) != 2:
        raise SystemExit("Unsupported compose cp arguments")
    mapped = []
    for value in copy_args:
        if ":" in value and not value.startswith("/"):
            service, path = value.split(":", 1)
            name = project + "_" + service + "_1"
            subprocess.run(["podman", "container", "exists", name], check=True)
            mapped.append(name + ":" + path)
        else:
            mapped.append(value)
    raise SystemExit(subprocess.call(["podman", "cp", *mapped]))
for i in range(len(args) - 1):
    if args[i] == "exec" and args[i + 1] == "--no-TTY":
        args[i + 1] = "-T"
os.execv(
    os.environ["VENV"] + "/bin/podman-compose", [os.environ["VENV"] + "/bin/podman-compose", *args]
)
