"""Experiment-only shared HF_TOKEN binding, explicitly approved by the operator."""
import os

from huggingface_hub import HfApi
from pydantic import field_validator

from flock.harbor.builtin.resources.hf_repo import Options as BaseOptions
from flock.harbor.builtin.resources.hf_repo import Resource as BaseResource


class Options(BaseOptions):
    token_env: str = "HF_TOKEN"

    @field_validator("token_env")
    @classmethod
    def repository_credential(cls, value):
        return value


class Resource(BaseResource):
    def __init__(self, options, *, state_dir, agents, secrets):
        self.options = Options.model_validate(options)
        self.token = os.environ.get(self.options.token_env)
        if not self.token:
            raise ValueError(f"Set {self.options.token_env}")
        secrets.append(self.token)
        self.api = HfApi(endpoint="https://huggingface.co", token=self.token)
        self.metadata = {}
