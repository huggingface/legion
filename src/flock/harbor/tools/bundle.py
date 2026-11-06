# from shepherd f3970bc:src/flock/tools/bundle.py
"""Build a portable bundle from selected tool modules and installed dependencies."""

import gzip
import hashlib
import importlib.metadata
import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .runner import load_selected


@dataclass
class ToolBundle:
    archive: Path
    manifest: dict
    sha256: str


def build_tool_bundle(directory: Path, names: list[str], destination: Path):
    _, definitions, modules = load_selected(directory, names)
    files, versions, visited, required = {}, {}, set(), set()
    target = {
        **default_environment(),
        "python_version": "3.10",
        "python_full_version": "3.10.0",
        "extra": "",
    }

    def add(name, content):
        if name in files and files[name] != content:
            raise ValueError(f"Conflicting bundle file: {name}")
        files[name] = content

    def dependency(name, subset=None):
        dist = importlib.metadata.distribution(name)
        canonical = canonicalize_name(dist.metadata["Name"])
        versions[canonical] = dist.version
        if subset is None:
            if canonical in visited:
                return
            visited.add(canonical)
            for requirement in dist.requires or []:
                req = Requirement(requirement)
                if req.marker is None or req.marker.evaluate(target):
                    if req.extras:
                        raise ValueError(f"Dependency extras are not supported: {req}")
                    installed = importlib.metadata.version(req.name)
                    if installed not in req.specifier:
                        raise ValueError(f"Installed dependency does not satisfy {req}")
                    dependency(req.name)
        entries = {str(entry): entry for entry in dist.files or []}
        selected = subset if subset is not None else entries
        for name in selected:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "__pycache__" in path.parts:
                if subset is not None:
                    raise ValueError(f"Invalid dependency file: {name}")
                continue
            # RECORD hashes the venv's bin/ scripts (shebang = install path) and INSTALLER names
            # pip or uv: neither is portable between the operator's venv and a worker's
            if path.suffix == ".pyc" or path.name == "direct_url.json" or (
                path.name in ("RECORD", "INSTALLER") and path.parent.name.endswith(".dist-info")
            ):
                continue
            if path.suffix in (".so", ".pyd", ".dll", ".pth"):
                raise ValueError(
                    f"Tool bundles require pure Python dependencies: {name}"
                )
            if name not in entries:
                raise ValueError(f"Missing dependency file: {name}")
            source = dist.locate_file(entries[name])
            if source.is_file():
                add(name, source.read_bytes())

    for module_name, module in modules.items():
        for name in getattr(module, "REQUIRES", []):
            if not isinstance(name, str) or not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_]*", name
            ):
                raise ValueError(f"Invalid resource requirement in {module_name}")
            required.add(name)
        add(f"tools/{module_name}.py", (directory / f"{module_name}.py").read_bytes())
        for name in getattr(module, "DEPENDENCIES", []):
            dependency(name)
        for name, paths in getattr(module, "BUNDLE_FILES", {}).items():
            dependency(name, paths)
    package_root = Path(__file__).resolve().parent.parent
    for name in (
        "__init__.py",
        "results.py",
        "tools/__init__.py",
        "tools/runner.py",
    ):
        add(f"flock/{name}", (package_root / name).read_bytes())
    manifest = {
        "tools": definitions,
        "dependencies": versions,
        "requires": sorted(required),
    }
    add("manifest.json", json.dumps(manifest, sort_keys=True, indent=2).encode())
    add(
        "bin/flock-tool",
        b"#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nfrom flock.tools.runner import main\nsys.exit(main())\n",
    )
    with destination.open("wb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for name, content in sorted(files.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    info.mode = 0o755 if name.startswith("bin/") else 0o644
                    archive.addfile(info, io.BytesIO(content))
    return ToolBundle(
        destination, manifest, hashlib.sha256(destination.read_bytes()).hexdigest()
    )
