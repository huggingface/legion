"""Deploy the read-only board viewer, without modifying the controller Space.

HF_TOKEN deploys the Space. FLOCK_VIEWER_READ_TOKEN can supply a separate read-only
bucket credential; otherwise the explicitly authorized HF_TOKEN is used.
The viewer credential must also list the permitted organization's members.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default="HuggingFaceSwarms/flock-msgboards")
    parser.add_argument("--bucket", default="HuggingFaceSwarms/Experiments")
    parser.add_argument("--organization", default="HuggingFaceSwarms")
    args = parser.parse_args()
    token = os.environ.get("FLOCK_VIEWER_READ_TOKEN") or os.environ["HF_TOKEN"]
    reader = HfApi(token=token)
    if not reader.bucket_info(args.bucket).private:
        raise ValueError("Board bucket must be private")
    list(reader.list_organization_members(args.organization))
    api = HfApi()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copytree(root / "src", stage / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(root / "pyproject.toml", stage / "pyproject.toml")
        shutil.copy2(root / "spaces/board-viewer/Dockerfile", stage / "Dockerfile")
        readme = (root / "spaces/board-viewer/README.md").read_text().replace(
            "HuggingFaceSwarms", args.organization)
        (stage / "README.md").write_text(readme)
        api.create_repo(args.space, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
        api.add_space_secret(args.space, "HF_TOKEN", token)
        api.add_space_variable(args.space, "FLOCK_VIEWER_BUCKET", args.bucket)
        api.add_space_variable(args.space, "FLOCK_VIEWER_ORG", args.organization)
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=stage,
                          commit_message="Deploy authenticated Flock board viewer",
                          delete_patterns=["src/**"])
    print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
