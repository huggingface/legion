"""Create and retain one private HF dataset for a run's code collaboration."""

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from pydantic import BaseModel, ConfigDict, Field

from flock.results import write_json


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    namespace: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    token_env: str = Field(default="HF_TOKEN", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.options = Options.model_validate(options)
        self.state_dir = Path(state_dir)
        self.agents = agents
        self.token = os.environ.get(self.options.token_env)
        if not self.token:
            raise ValueError(f"Set {self.options.token_env} for the hf_repo resource")
        secrets.append(self.token)
        self.api = HfApi(token=self.token)
        self.metadata = {}

    async def start(self):
        await asyncio.to_thread(self._start)

    def _start(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        record = self.state_dir / "repository.json"
        if record.exists():
            self.metadata = json.loads(record.read_text())
            repo_id = self.metadata["repo_id"]
            if repo_id.split("/")[0] != self.options.namespace:
                raise ValueError("Recorded repository belongs to a different namespace")
        else:
            run_id = self.state_dir.parents[2].name
            name = "flock-" + re.sub(r"[^A-Za-z0-9_-]", "-", run_id)
            if len(name) > 96:
                name = (
                    name[:83] + "-" + hashlib.sha256(run_id.encode()).hexdigest()[:12]
                )
            repo_id = f"{self.options.namespace}/{name}"
            self.api.create_repo(
                repo_id, repo_type="dataset", private=True, exist_ok=False
            )
            self.metadata = {
                "repo_id": repo_id,
                "url": f"https://huggingface.co/datasets/{repo_id}",
                "private": True,
                "initialized": False,
            }
            write_json(record, self.metadata)
        info = self.api.repo_info(repo_id, repo_type="dataset")
        if not info.private:
            raise ValueError(
                "The recorded collaboration repository must remain private"
            )
        if not self.metadata.get("initialized"):
            # Retrying startup must never overwrite agent work after initialization.
            if "README.md" not in self.api.list_repo_files(
                repo_id, repo_type="dataset"
            ):
                files = {
                    "README.md": "# Flock collaboration\n\nOriginal source shared by this run's agents.\n"
                    "Reviews are advisory. Each agent must build and verify its own submission.\n",
                    ".gitignore": "executable\n.env\n.env.*\n__pycache__/\n*.pyc\n.venv/\n",
                }
                self.api.create_commit(
                    repo_id,
                    repo_type="dataset",
                    commit_message="Initialize Flock collaboration",
                    operations=[
                        CommitOperationAdd(path_in_repo=k, path_or_fileobj=v.encode())
                        for k, v in files.items()
                    ],
                )
            self.metadata["initialized"] = True
            write_json(record, self.metadata)

    def bind(self, name):
        return {
            "env": {
                "FLOCK_HF_REPO": self.metadata["repo_id"],
                "FLOCK_HF_REPO_TOKEN": self.token,
                "FLOCK_HF_REPO_AGENT": name,
            },
            "allowed_hosts": ["huggingface.co"],
            "metadata": {**self.metadata, "agent": name},
        }

    async def wait(self):
        await asyncio.Event().wait()

    async def export(self, directory):
        def snapshot():
            if not self.metadata:
                return {"created": False}
            repo_id = self.metadata["repo_id"]
            info = self.api.repo_info(repo_id, repo_type="dataset")
            prs = [
                {
                    "number": p.num,
                    "title": p.title,
                    "author": p.author,
                    "status": p.status,
                    "url": f"{self.metadata['url']}/discussions/{p.num}",
                }
                for p in self.api.get_repo_discussions(repo_id, repo_type="dataset")
                if p.is_pull_request
            ]
            return {**self.metadata, "main_sha": info.sha, "pull_requests": prs}

        write_json(directory / "repository.json", await asyncio.to_thread(snapshot))
        return {"repository": "repository.json"}

    async def stop(self):
        # Collaboration history deliberately outlives the local experiment.
        pass
