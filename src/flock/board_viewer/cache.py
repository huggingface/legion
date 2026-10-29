"""Validated immutable exports cached outside the live board storage."""
from __future__ import annotations

import json
import re
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

from .export import SCHEMA, digest

MANIFEST = re.compile(r"flock/runs/([\w-]+)/([0-9a-f]{32})/viewer/manifest.json", re.ASCII)
HEX = re.compile(r"[0-9a-f]{64}")


class BoardCache:
    def __init__(self, api, bucket: str, root: Path):
        self.api, self.bucket, self.root = api, bucket, root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.boards: dict[str, dict] = {}
        self.error = False
        self.lock = threading.Lock()

    def refresh(self):
        with self.lock:
            self.error = False
            try:
                for entry in self.api.list_bucket_tree(self.bucket, prefix="flock/runs", recursive=True):
                    match = MANIFEST.fullmatch(entry.path)
                    if match:
                        try:
                            self._load(entry.path, match[1], match[2])
                        except Exception:  # noqa: BLE001 — redact upstream errors and retain last good data
                            self.error = True
                            if match[2] in self.boards:
                                self.boards[match[2]]["refresh_error"] = True
                # Keep last good boards during partial listing/publication errors.
            except Exception:  # noqa: BLE001 — redact upstream errors and retain last good data
                self.error = True
                return

    def _load(self, path: str, swarm: str, run_id: str):
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            staging = Path(tmp)
            self.api.download_bucket_files(self.bucket, [(path, staging / "manifest.json")],
                                           raise_on_missing_files=True)
            manifest = json.loads((staging / "manifest.json").read_text())
            generation = manifest["generation"]
            if (manifest["schema_version"] != SCHEMA or manifest["run_id"] != run_id
                    or manifest["swarm"] != swarm or not re.fullmatch("[0-9a-f]{32}", generation)
                    or type(manifest["completed"]) is not bool):
                raise ValueError("Unsupported manifest")
            datetime.fromisoformat(manifest["published_at"]).astimezone(UTC)
            files = manifest["files"]
            if "board.json" not in files or any(
                (name != "board.json" and not HEX.fullmatch(name)) or not HEX.fullmatch(sha)
                for name, sha in files.items()
            ):
                raise ValueError("Invalid manifest files")
            previous = self.boards.get(run_id)
            if previous and previous["manifest"] == manifest:
                previous["refresh_error"] = False
                return
            remote = path.removesuffix("manifest.json") + generation
            self.api.download_bucket_files(self.bucket,
                [(f"{remote}/{name}", staging / name) for name in files], raise_on_missing_files=True)
            if any(digest(staging / name) != sha for name, sha in files.items()):
                raise ValueError("Export checksum mismatch")
            doc = json.loads((staging / "board.json").read_text())
            if doc["schema_version"] != SCHEMA:
                raise ValueError("Unsupported board export")
            for name in ("principals", "channels", "messages", "channel_members", "message_reactions",
                         "attachments"):
                if not isinstance(doc[name], list):
                    raise TypeError("Invalid board export")
            if any(a["file"] not in files or a["file"] != a["sha256"] for a in doc["attachments"]):
                raise ValueError("Invalid attachment manifest")
            target = self.root / run_id / generation
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                # A process restart can reuse an immutable, verified disk generation.
                if any(not (target / name).is_file() or digest(target / name) != sha
                       for name, sha in files.items()):
                    raise ValueError("Generation changed in place")
            else:
                staging.rename(target)
            self.boards[run_id] = {"manifest": manifest, "data": doc, "path": target,
                                   "refresh_error": False}

    def summaries(self):
        return [{**b["manifest"], "refresh_error": b["refresh_error"],
                 "messages": len(b["data"]["messages"])} for b in self.boards.copy().values()]
