"""Shared-world operations for any World API v1 world (docs/world.md): list its tools, call one.

Connection settings come from the agent environment (resources.world): WORLD_SOCKET, a launcher
proxy that adds this agent's credentials, or WORLD_URL, WORLD_AGENT and WORLD_TOKEN. A world tool
with `x-flock-workspace` gets the named local git clone uploaded with the call, captured here.
"""

import base64
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import httpx

DEPENDENCIES = ["httpx"]
REQUIRES = ["world"]

Scalar = str | int | float | bool | None
MAX_BYTES, MAX_FILES = 128 << 20, 10000  # as flock.world.workspace, which the world enforces
# Never run a command the clone configured, or let its diff settings change the patch format.
SAFE_GIT = ("-c", "core.fsmonitor=false", "-c", "diff.noprefix=false",
            "-c", "diff.mnemonicPrefix=false", "-c", "diff.relative=false")


def _client():
    socket_path = os.environ.get("WORLD_SOCKET")
    if socket_path:
        transport = httpx.HTTPTransport(uds=socket_path)
        return httpx.Client(base_url="http://world", transport=transport, timeout=60)
    headers = {"X-World-Token": os.environ["WORLD_TOKEN"],
               "X-Flock-Agent": os.environ["WORLD_AGENT"]}
    return httpx.Client(base_url=os.environ["WORLD_URL"], headers=headers, timeout=60)


def _body(response):
    """The JSON answer, or an error carrying the world's code, message and retry hint."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.is_success and isinstance(body, dict) and body.get("ok") is not False:
        return body
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        retry = error.get("retry_after_s")
        raise RuntimeError(f"{error.get('code')}: {error.get('message')}"
                           + (f" (retry after {retry} s)" if retry else ""))
    raise RuntimeError(f"world answered HTTP {response.status_code}: "
                       f"{str(error or response.text)[:300]}")


def _code(response):
    try:
        return response.json()["error"]["code"]
    except (ValueError, KeyError, TypeError):
        return None


def _git(root, *args, env=None):
    p = subprocess.run(["git", *SAFE_GIT, "-C", root, *args], capture_output=True, env=env,
                       timeout=120, check=False)
    if p.returncode:
        raise ValueError(f"git {args[0]} in {root} failed: "
                         f"{p.stderr.decode(errors='replace').strip()[-300:]}")
    return p.stdout


def _capture(path, patch):
    """The clone at `path` against its base on origin/main: a binary patch (workspace-upload-v2)
    or every file (v1). The world checks the base and the files; nothing here is trusted."""
    root = _git(path, "rev-parse", "--show-toplevel").decode().strip()
    base = _git(root, "merge-base", "HEAD", "refs/remotes/origin/main").decode().strip()
    with tempfile.TemporaryDirectory() as tmp:
        # A private index of what the world gets: the clone's own index stays as it is.
        env = {**os.environ, "GIT_INDEX_FILE": os.path.join(tmp, "index")}
        _git(root, "read-tree", "HEAD", env=env)
        _git(root, "add", "--all", env=env)
        names, total = [], 0
        for entry in _git(root, "ls-files", "--stage", "-z", env=env).decode().split("\0"):
            if not entry:
                continue
            meta, name = entry.split("\t", 1)
            mode = meta.split()[0]
            if mode == "120000":
                raise ValueError(f"{name} is a symlink: the world takes regular files only; "
                                 "remove it or add it to .gitignore")
            if mode == "160000":
                raise ValueError(f"{name} is a nested git repository (e.g. a git worktree): "
                                 "move it out of the clone or add it to .gitignore")
            names.append(name)
            total += os.lstat(os.path.join(root, name)).st_size
        if len(names) > MAX_FILES or total > MAX_BYTES:
            raise ValueError("the clone is over 128 MiB or 10,000 files: remove large or "
                             "generated files, or add them to .gitignore")
        if not patch:  # regular files only (modes checked above): nothing follows a link
            files = {name: base64.b64encode(Path(root, name).read_bytes()).decode("ascii")
                     for name in names}
            return {"base_sha": base, "files": files}
        diff = _git(root, "diff", "--cached", "--binary", "--full-index", "--find-renames",
                    "--no-ext-diff", "--no-textconv", "--no-color", base, env=env)
    return {"base_sha": base, "patch": base64.b64encode(diff).decode("ascii")}


def tools() -> dict:
    """List the world's tools with their descriptions and JSON-schema parameters."""
    with _client() as world:
        manifest = _body(world.get("/manifest"))
    return {
        "world": manifest["name"],
        "description": manifest.get("description", ""),
        "tools": [
            {key: tool[key] for key in ("name", "description", "parameters") if key in tool}
            for tool in manifest.get("tools", [])
        ],
    }


def call(tool: str, args: dict[str, Scalar | list[Scalar] | dict[str, Scalar]] | None = None):
    """Call a world tool (see world.tools) as yourself; a clone path it takes is uploaded."""
    with _client() as world:
        manifest = _body(world.get("/manifest"))
        spec = next((t for t in manifest.get("tools", []) if t.get("name") == tool), None)
        if spec is None:
            raise ValueError(f"the world has no tool {tool!r}; run world.tools")
        args = args or {}
        payload = {"args": args}
        capabilities = manifest.get("capabilities", [])
        upload = spec.get("x-flock-workspace") if "workspace-upload-v1" in capabilities else None
        local = args.get(upload["arg"], upload.get("default")) if upload else None
        if local is not None and local not in upload.get("skip", []) and not (
            upload.get("skip_prefix") and str(local).startswith(upload["skip_prefix"])
        ):
            payload["workspace"] = _capture(str(local), "workspace-upload-v2" in capabilities)
        headers = {"X-Request-Id": uuid.uuid4().hex}  # one logical call: a resend reuses it
        timeout = max(60, spec.get("x-flock-timeout_s") or 0)
        response = world.post(f"/tools/{tool}", json=payload, headers=headers, timeout=timeout)
        if "patch" in payload.get("workspace", {}) and _code(response) == "need_full":
            # The world lacks the patch's base; nothing happened yet, so send the whole tree.
            payload["workspace"] = _capture(str(local), False)
            response = world.post(f"/tools/{tool}", json=payload, headers=headers,
                                  timeout=timeout)
    return _body(response)["result"]


TOOLS = {"tools": tools, "call": call}
