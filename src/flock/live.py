"""Live changes to a running swarm (docs/live.md): the short diff `flock update` prints.
No I/O to the controller here: the CLI does that."""

from __future__ import annotations

import json
from typing import Any

from flock.models import SwarmSpec

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


def keep_replicas(old: SwarmSpec, new: SwarmSpec, *, ramps_only: bool = False) -> list[str]:
    """Copy the live replica count of every type `old` already has into `new`; returns one
    line per type whose spec value differed (`agent: replicas kept (spec 20, live 300)`).
    `ramps_only` (`update --replicas spec`): only where the file would lower a ramp type
    (the ramp raised it; lower `replicas_ramp.max` instead)."""
    lines = []
    for name, ts in new.types.items():
        live = old.types[name].replicas if name in old.types else None
        if ramps_only and (live is None or ts.replicas_ramp is None or live < ts.replicas):
            continue
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
