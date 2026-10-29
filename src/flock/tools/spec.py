"""Apply-time glue between a swarm spec and its tools folder (docs/tools.md, "Selection")."""

from pathlib import Path

from flock.models import CommsSpec, SwarmSpec, TypeSpec, comms_name, comms_tools
from flock.tools.bundle import Bundle, pack
from flock.tools.loader import Manifest, ToolsError, select

DEFAULT_DIR = "tools"


def tools_folder(spec: SwarmSpec, spec_file: str | Path) -> Path:
    p = Path(spec.tools_dir or DEFAULT_DIR)
    return p if p.is_absolute() else Path(spec_file).resolve().parent / p


def comms_clashes(ts: TypeSpec, comms: CommsSpec | None, folder: list[str]) -> str | None:
    """A folder tool that has the model-visible name of one of the type's comms tools."""
    clash = sorted({comms_name(e) for e in comms_tools(ts, comms)} & set(folder))
    if not clash:
        return None
    return (f"folder tool(s) {', '.join(clash)} have the name of a built-in comms tool the type "
            "gets (docs/comms.md); rename them")


def check_selection(spec: SwarmSpec, manifest: Manifest | None) -> None:
    """Every `types.<t>.tools` names tools the bundle has, and none clashes with the type's
    built-in comms tools (raises ToolsError). `comms:*` entries are not folder tools."""
    for name, ts in spec.types.items():
        sel = ts.folder_tools
        if sel is None:
            continue
        if manifest is None:
            raise ToolsError(f"types.{name}.tools: the swarm has no tools bundle")
        chosen = select(manifest, sel, where=f"types.{name}.tools")
        if clash := comms_clashes(ts, spec.comms, [t.name for t in chosen]):
            raise ToolsError(f"types.{name}.tools: {clash}")


def prepare(spec: SwarmSpec, spec_file: str | Path) -> Bundle | None:
    """The bundle to upload for this spec, or None when it uses no tools folder (no type selects
    tools and `tools_dir` is unset: an existing spec next to an unrelated `tools/` is unchanged).
    Raises ToolsError: a missing folder, a bad module, an unknown tool name."""
    selecting = [n for n, ts in spec.types.items() if ts.folder_tools is not None]
    if not selecting and spec.tools_dir is None:
        return None
    folder = tools_folder(spec, spec_file)
    if not folder.is_dir():
        who = f"types {', '.join(selecting)} select tools, but " if selecting else ""
        raise ToolsError(f"{who}the tools folder {folder} does not exist")
    bundle = pack(folder)
    check_selection(spec, bundle.manifest)
    return bundle
