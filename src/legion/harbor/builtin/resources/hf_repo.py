"""Bind every worker to one existing private HF collaboration dataset."""

import asyncio
import os

from huggingface_hub import HfApi
from pydantic import BaseModel, ConfigDict, Field, field_validator

from legion.harbor.results import write_json


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    repo_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
    token_env: str = Field(default="HF_REPO_TOKEN", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")

    @field_validator("token_env")
    @classmethod
    def repository_credential(cls, value):
        if value == "HF_TOKEN":
            raise ValueError("Use a separate repository-scoped token, not HF_TOKEN")
        return value


class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.options = Options.model_validate(options)
        self.token = os.environ.get(self.options.token_env)
        if not self.token:
            raise ValueError(f"Set {self.options.token_env} for the hf_repo resource")
        if self.token == os.environ.get("HF_TOKEN"):
            raise ValueError("Repository token must differ from the infrastructure HF_TOKEN")
        secrets.append(self.token)
        self.api = HfApi(endpoint="https://huggingface.co", token=self.token)
        self.metadata = {}

    async def start(self):
        # Reading an existing main avoids creation/initialization races across workers.
        info = await asyncio.to_thread(
            self.api.repo_info, self.options.repo_id, repo_type="dataset", revision="main"
        )
        if not info.private:
            raise ValueError("The hf_repo collaboration dataset must be private")
        self.metadata = {
            "repo_id": self.options.repo_id,
            "url": f"https://huggingface.co/datasets/{self.options.repo_id}",
            "private": True,
        }

    def bind(self, name):
        return {
            "env": {
                "LEGION_HF_REPO": self.options.repo_id,
                "LEGION_HF_REPO_TOKEN": self.token,
                "LEGION_HF_REPO_AGENT": name,
            },
            "allowed_hosts": ["huggingface.co"],
            "metadata": {**self.metadata, "agent": name},
        }

    async def wait(self):
        await asyncio.Event().wait()

    async def export(self, directory):
        def snapshot():
            if not self.metadata:
                return {"connected": False}
            repo_id = self.options.repo_id
            info = self.api.repo_info(repo_id, repo_type="dataset", revision="main")
            prs = [
                {"number": p.num, "title": p.title, "author": p.author,
                 "status": p.status,
                 "url": f"{self.metadata['url']}/discussions/{p.num}"}
                for p in self.api.get_repo_discussions(repo_id, repo_type="dataset")
                if p.is_pull_request
            ]
            return {**self.metadata, "main_sha": info.sha, "pull_requests": prs}

        write_json(directory / "repository.json", await asyncio.to_thread(snapshot))
        return {"repository": "repository.json"}

    async def stop(self):
        # The operator owns the repository; stopping a worker never deletes it.
        pass
