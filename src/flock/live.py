"""Live changes to a running swarm (docs/live.md): the short diff `flock update` prints, the
files `flock run --watch` polls, and the argument syntax of `flock grant` / `flock revoke`.
No I/O to the controller here: the CLI and the local run do that."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flock.models import ALL_COMMS_TOOLS as COMMS_TOOLS
from flock.models import Grant, GrantLimits, SwarmSpec

# Keys whose values are prompt text: reported as "prompts changed", never printed.
PROMPT_KEYS = ("system_prompt", "prompt", "first", "kickoff", "protocols_text")
# Written by apply or the controller, or reported on their own line (tools).
SKIP_KEYS = ("resolved_model", "channel_id", "kickoff_id", "tools_version")
# `world.*` keys the controller blanks and refills on every apply (host.resolve): always
# different between the folder's spec and the stored one. Manifest tools are reported apart.
WORLD_DERIVED = ("endpoint", "manifest", "credential_origin", "credential_id")


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in SKIP_KEYS:
            continue
        if isinstance(v, dict) and v:
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _show(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, str):
        return v if len(v) <= 60 and "\n" not in v else f"<{len(v)} chars>"
    text = json.dumps(v)
    return text if len(text) <= 60 else text[:59] + "…"


def _items(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v)


def _delta(a: Any, b: Any) -> str:
    """`old -> new`; two lists as `+added -removed` element by element (never truncated)."""
    if isinstance(a, list) and isinstance(b, list):
        add = [f"+{_items(x)}" for x in b if x not in a]
        gone = [f"-{_items(x)}" for x in a if x not in b]
        return " ".join(add + gone) or "reordered"
    return f"{_show(a)} -> {_show(b)}"


def _changes(old: dict[str, Any], new: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    a, b = _flat(old), _flat(new)
    return [(k, a.get(k), b.get(k)) for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)]


def keep_replicas(old: SwarmSpec, new: SwarmSpec) -> list[str]:
    """Copy the live replica count of every type `old` already has into `new`; returns one
    line per type whose spec value differed (`agent: replicas kept (spec 20, live 300)`)."""
    lines = []
    for name, ts in new.types.items():
        live = old.types[name].replicas if name in old.types else None
        if live is not None and live != ts.replicas:
            lines.append(f"{name}: replicas kept (spec {ts.replicas}, live {live})")
            ts.replicas = live
    return lines


def world_tool_changes(old: SwarmSpec, new: SwarmSpec) -> str | None:
    """`world tools: +a -b` between the two manifests, or None when the tools are the same."""
    a = old.world.manifest.names() if old.world and old.world.manifest else []
    b = new.world.manifest.names() if new.world and new.world.manifest else []
    return "world tools: " + _delta(a, b) if a != b else None


def diff_specs(old: SwarmSpec, new: SwarmSpec, tools: tuple[list[str], list[str]] | None = None
               ) -> list[str]:
    """What applying `new` over the running `old` changes, one short line each: the tools
    version (old -> new, tools added and removed; `tools` is (old names, new names)), the
    prompts changed per type, then every other key as `path: old -> new`. Empty: no change."""
    lines: list[str] = []
    if old.tools_version != new.tools_version:
        line = f"tools: {old.tools_version or '-'} -> {new.tools_version or '-'}"
        if tools is not None:
            added = [f"+{n}" for n in tools[1] if n not in tools[0]]
            gone = [f"-{n}" for n in tools[0] if n not in tools[1]]
            if added or gone:
                line += f" ({' '.join(added + gone)})"
        lines.append(line)
    prompts: dict[str, list[str]] = {}
    keys: list[str] = []
    o, n = old.model_dump(mode="json"), new.model_dump(mode="json")
    if old.comms and new.comms and old.comms.managed and new.comms.managed:
        for doc in (o, n):
            for key in ("url", "invite_env"):
                doc["comms"].pop(key, None)  # resolved per-run connection is not a YAML edit
    for name in sorted(o["types"].keys() | n["types"].keys()):
        if name not in n["types"]:
            keys.append(f"- type {name} (its agents are removed)")
            continue
        if name not in o["types"]:
            keys.append(f"+ type {name} ({n['types'][name]['harness']}, "
                        f"replicas {n['types'][name]['replicas']})")
            continue
        for path, a, b in _changes(o["types"][name], n["types"][name]):
            if path.rsplit(".", 1)[-1] in PROMPT_KEYS:
                prompts.setdefault(name, []).append(path)
            else:
                keys.append(f"types.{name}.{path}: {_delta(a, b)}")
    for top in sorted((o.keys() | n.keys()) - {"types"}):
        a, b = o.get(top), n.get(top)
        if a == b:
            continue
        if not isinstance(a, dict) or not isinstance(b, dict):
            if top not in SKIP_KEYS:
                keys.append(f"{top}: {_show(a)} -> {_show(b)}")
            continue
        if top == "world":
            a, b = ({k: v for k, v in d.items() if k not in WORLD_DERIVED} for d in (a, b))
        for path, x, y in _changes(a, b):
            if path.rsplit(".", 1)[-1] in PROMPT_KEYS:
                note = " (posted once per channel: not posted again)" if path == "kickoff" else ""
                prompts.setdefault(top, []).append(path + note)
            else:
                keys.append(f"{top}.{path}: {_delta(x, y)}")
    if prompts:
        lines.append("prompts changed: " + "; ".join(f"{t} ({', '.join(p)})"
                                                     for t, p in prompts.items()))
    return lines + keys


# --------------------------------------------------------------------------- watching


def watched_files(spec_file: Path, spec: SwarmSpec | None) -> list[Path]:
    """The spec file, every file it references (system prompts, loop prompts, the comms
    kickoff and protocols) and the tools folder's files."""
    base = spec_file.parent
    names: list[str | None] = []
    tools_dir = base / "tools"
    if spec is not None:
        for ts in spec.types.values():
            names.append(ts.system_prompt_file)
            names.append(ts.system_prompt_append_file)
            if ts.loop is not None:
                names.append(ts.loop.prompt_file)
                names.append(ts.loop.final.prompt_file if ts.loop.final else None)
        if spec.comms is not None:
            names += [spec.comms.kickoff_file, *spec.comms.protocols]
        if spec.tools_dir:
            tools_dir = base / spec.tools_dir
    files = [spec_file] + [p if (p := Path(x)).is_absolute() else base / p for x in names if x]
    if tools_dir.is_dir():
        files += sorted(f for f in tools_dir.rglob("*") if f.is_file()
                        and "__pycache__" not in f.parts and not f.name.startswith(".")
                        and f.suffix != ".pyc")
    return list(dict.fromkeys(files))


def snapshot(files: list[Path]) -> dict[str, tuple[int, int] | None]:
    out: dict[str, tuple[int, int] | None] = {}
    for f in files:
        try:
            st = f.stat()
            out[str(f)] = (st.st_mtime_ns, st.st_size)
        except OSError:
            out[str(f)] = None
    return out


class FolderWatch:
    """Polls the experiment folder's files (no inotify dependency); the file list is re-read
    each poll, so a new tool module or a newly referenced prompt counts. `poll(now)` is True
    once a change has settled: something changed and then nothing for `quiet` seconds (an
    editor's save in several writes gives one update). `spec` is the last spec read, which
    says what files to watch (`retarget` after each update)."""

    def __init__(self, spec_file: Path, spec: SwarmSpec | None, quiet: float = 1.0) -> None:
        self.spec_file, self.spec, self.quiet = Path(spec_file), spec, quiet
        self.last = self._snap()
        self._dirty_at: float | None = None

    def _snap(self) -> dict[str, tuple[int, int] | None]:
        return snapshot(watched_files(self.spec_file, self.spec))

    def retarget(self, spec: SwarmSpec) -> None:
        """After an update: watch what `spec` references. Files newly watched start from
        their state now; the others keep theirs, so a save during the update is not lost."""
        self.spec = spec
        now = snapshot(watched_files(self.spec_file, spec))
        self.last = {k: self.last.get(k, v) for k, v in now.items()}

    def poll(self, now: float) -> bool:
        snap = self._snap()
        if snap != self.last:
            self.last, self._dirty_at = snap, now
            return False
        if self._dirty_at is not None and now - self._dirty_at >= self.quiet:
            self._dirty_at = None
            return True
        return False


# --------------------------------------------------------------------------- grant arguments

ON, OFF = ("on", "true", "yes", "1"), ("off", "false", "no", "0")


def _switch(cap: str, value: str | None) -> bool:
    v = (value or "on").lower()
    if v not in ON + OFF:
        raise ValueError(f"{cap}={value}: use on or off")
    return v in ON


def _tool_name(name: str) -> str:
    """`post` / `comms:post` -> `comms:post`; anything else is a folder tool name."""
    return f"comms:{name}" if f"comms:{name}" in COMMS_TOOLS else name


def grant_patch(g: Grant, args: list[str]) -> Grant:
    """`g` with `flock grant` arguments applied: `tools=+a,-b` (or `+a` / `-b` alone),
    `comms.post[=on|off]` (the same as tools +post / -post), `bash[=on|off]`,
    `limits.max_turns=N`, `limits.budget_usd=X`."""
    tools, limits, bash = dict(g.tools), g.limits.model_dump(), g.bash
    for arg in args:
        cap, _, value = arg.partition("=")
        cap, value = cap.strip(), value.strip() if "=" in arg else None
        if cap[:1] in "+-" and value is None:
            cap, value = "tools", cap
        if cap == "tools":
            if not value:
                raise ValueError("tools=+name,-name: name the tools to add (+) or take away (-)")
            for item in value.split(","):
                item = item.strip()
                on = not item.startswith("-")
                name = item.lstrip("+-")
                if not name:
                    raise ValueError(f"tools: empty name in {value!r}")
                tools[_tool_name(name)] = on
        elif cap.startswith("comms.") and f"comms:{cap[6:]}" in COMMS_TOOLS:
            tools[f"comms:{cap[6:]}"] = _switch(cap, value)
        elif cap == "bash":
            bash = _switch(cap, value)
        elif cap in ("limits.max_turns", "limits.budget_usd"):
            if value is None:
                raise ValueError(f"{cap}=<value>: give a number")
            key = cap.split(".", 1)[1]
            try:
                limits[key] = int(value) if key == "max_turns" else float(value.lstrip("$"))
            except ValueError:
                raise ValueError(f"{cap}={value}: not a number") from None
        else:
            raise ValueError(f"unknown capability {cap!r}; grantable: tools=+a,-b, comms.post, "
                             "comms.read, comms.search, bash, limits.max_turns, limits.budget_usd")
    return Grant(tools=tools, limits=GrantLimits(**limits), bash=bash)


def revoke_patch(g: Grant, args: list[str]) -> Grant:
    """`g` without the named capabilities: `tools` (all), `tools=a,b`, `comms.post`, `bash`,
    `limits` (both), `limits.max_turns`, `limits.budget_usd`."""
    tools, limits, bash = dict(g.tools), g.limits.model_dump(), g.bash
    for arg in args:
        cap, _, value = arg.partition("=")
        if cap[:1] in "+-" and not value:
            cap, value = "tools", cap
        if cap == "tools" and not value:
            tools = {}
        elif cap == "tools":
            for name in value.split(","):
                tools.pop(_tool_name(name.strip().lstrip("+-")), None)
        elif cap.startswith("comms.") and f"comms:{cap[6:]}" in COMMS_TOOLS:
            tools.pop(f"comms:{cap[6:]}", None)
        elif cap == "bash":
            bash = None
        elif cap == "limits":
            limits = {}
        elif cap in ("limits.max_turns", "limits.budget_usd"):
            limits[cap.split(".", 1)[1]] = None
        else:
            raise ValueError(f"unknown capability {cap!r}")
    return Grant(tools=tools, limits=GrantLimits(**limits), bash=bash)
