# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen AgentEscape scenario membership and pinned Harbor registry selection."""

import hashlib
import json
import re
import tomllib
from collections import Counter
from pathlib import Path

COUNTS = {level: {"sft": 25, "rl": 25, "test": 10} for level in (5, 10, 15, 20)}
COUNTS[25] = {"sft": 12, "rl": 13, "test": 5}
REPO = "HuggingFaceSwarms/AgentEscapeBench-harbor"
TASK_REVISION = "144b647af561e28d681552cb04bc7b4a8dbeb1ce"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build_manifest(tasks):
    rows = sorted((dict(task) for task in tasks), key=lambda row: row["scenario_id"])
    source_digest = digest(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())
    for difficulty, counts in COUNTS.items():
        group = sorted(
            (row for row in rows if row["difficulty"] == difficulty),
            key=lambda row: digest(("agentescape-splits-v1:42:" + row["scenario_id"]).encode()),
        )
        if len(group) != sum(counts.values()):
            raise ValueError(f"Unexpected scenario count for difficulty {difficulty}")
        offset = 0
        for split in ("test", "sft", "rl"):
            for row in group[offset : offset + counts[split]]:
                row["split"] = split
            offset += counts[split]
    manifest = {
        "version": "1.0.0",
        "seed": 42,
        "assignment": "sha256(agentescape-splits-v1:42:<scenario_id>); test, sft, rl",
        "source_digest": source_digest,
        "task_repo": REPO,
        "task_revision": TASK_REVISION,
        "scenarios": rows,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    if manifest["version"] != "1.0.0" or manifest["task_repo"] != REPO:
        raise ValueError("Unsupported split manifest")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["task_revision"]):
        raise ValueError("Task revision must be an immutable commit")
    rows = manifest["scenarios"]
    for key in ("scenario_id", "task_path", "scenario_sha256"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate {key}; scenarios and variants must not cross splits")
    expected = Counter(
        {
            (level, split): count
            for level, counts in COUNTS.items()
            for split, count in counts.items()
        }
    )
    if Counter((row["difficulty"], row["split"]) for row in rows) != expected:
        raise ValueError("Split counts do not match the frozen v1 protocol")
    for row in rows:
        if row["task_path"] != "tasks/" + row["scenario_id"]:
            raise ValueError("Task path does not match scenario identity")
        if not re.fullmatch(r"[0-9a-f]{64}", row["scenario_sha256"]):
            raise ValueError("Invalid scenario content hash")


def build_registry(manifest):
    validate_manifest(manifest)
    return [
        {
            "name": f"agentescape-{split}",
            "version": "1.0.0",
            "description": f"Frozen AgentEscape v1 {split} scenarios; see splits/v1.json",
            "tasks": [
                {
                    "name": row["scenario_id"],
                    "git_url": "https://huggingface.co/datasets/" + manifest["task_repo"],
                    "git_commit_id": manifest["task_revision"],
                    "path": row["task_path"],
                }
                for row in manifest["scenarios"]
                if row["split"] == split
            ],
        }
        for split in ("sft", "rl", "test")
    ]


def select_tasks(manifest, registry, split, difficulties, command, names=None):
    validate_manifest(manifest)
    if split not in ("sft", "rl", "test") or (command == "train" and split != "rl"):
        raise ValueError("An explicit split is required; RL training requires split=rl")
    if (
        not difficulties
        or len(set(difficulties)) != len(difficulties)
        or set(difficulties) - COUNTS.keys()
    ):
        raise ValueError("Select unique supported difficulties")
    expected = build_registry(manifest)
    for dataset in expected:
        found = [
            entry
            for entry in registry
            if entry["name"] == dataset["name"] and entry["version"] == "1.0.0"
        ]
        if (
            len(found) != 1
            or sorted(found[0]["tasks"], key=lambda x: x["name"]) != dataset["tasks"]
        ):
            raise ValueError(
                "Harbor registry membership or task revision differs from split manifest"
            )
    pools = [
        [
            row
            for row in manifest["scenarios"]
            if row["split"] == split and row["difficulty"] == level
        ]
        for level in difficulties
    ]
    # Round-robin, including unequal 30-scenario difficulty pools.
    selected = [pool[i] for i in range(max(map(len, pools))) for pool in pools if i < len(pool)]
    if names is not None:
        if len(set(names)) != len(names) or set(names) - {row["scenario_id"] for row in selected}:
            raise ValueError("Requested tasks lie outside the selected split/difficulties")
        selected = [row for row in selected if row["scenario_id"] in names]
    return selected


def verify_task(path, row):
    path = Path(path)
    metadata = tomllib.loads((path / "task.toml").read_text())["metadata"]
    if (
        metadata["scenario_sha256"] != row["scenario_sha256"]
        or metadata["difficulty"] != row["difficulty"]
    ):
        raise ValueError(f"Task metadata changed for {row['scenario_id']}")
    if digest((path / "environment/puzzle/scenario.json").read_bytes()) != row["scenario_sha256"]:
        raise ValueError(f"Scenario content changed for {row['scenario_id']}")
