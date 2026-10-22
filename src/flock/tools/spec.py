"""Apply-time glue between a swarm spec and its tools folder (docs/tools.md, "Selection")."""

from __future__ import annotations

from pathlib import Path

from flock.models import SwarmSpec
from flock.tools.bundle import Bundle, pack
from flock.tools.loader import Manifest, ToolsError, select

DEFAULT_DIR = "tools"


def tools_folder(spec: SwarmSpec, spec_file: str | Path) -> Path:
    p = Path(spec.tools_dir or DEFAULT_DIR)
    return p if p.is_absolute() else Path(spec_file).resolve().parent / p


def check_selection(spec: SwarmSpec, manifest: Manifest | None) -> None:
    """Every `types.<t>.tools` names tools the bundle has (raises ToolsError)."""
    for name, ts in spec.types.items():
        if ts.tools is None:
            continue
        if manifest is None:
            raise ToolsError(f"types.{name}.tools: the swarm has no tools bundle")
        select(manifest, ts.tools, where=f"types.{name}.tools")


def prepare(spec: SwarmSpec, spec_file: str | Path) -> Bundle | None:
    """The bundle to upload for this spec, or None when it uses no tools folder (no type selects
    tools and `tools_dir` is unset: an existing spec next to an unrelated `tools/` is unchanged).
    Raises ToolsError: a missing folder, a bad module, an unknown tool name."""
    selecting = [n for n, ts in spec.types.items() if ts.tools is not None]
    if not selecting and spec.tools_dir is None:
        return None
    folder = tools_folder(spec, spec_file)
    if not folder.is_dir():
        who = f"types {', '.join(selecting)} select tools, but " if selecting else ""
        raise ToolsError(f"{who}the tools folder {folder} does not exist")
    bundle = pack(folder)
    check_selection(spec, bundle.manifest)
    return bundle
