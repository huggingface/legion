"""Offline checks for the math experiment server configurations."""

from pathlib import Path

import pytest

from legion.controller.hf_jobs_servers import vllm_command
from legion.harbor.spec import Spec, compile_spec

ROOT = Path(__file__).resolve().parents[2] / "experiments"
BENCHMARKS = {
    "imo-answerbench": "40d9d407c8b5e4b62c6b2047e21c807214f2a068",
    "imo-proofbench": "10d6c29485d5fb117daa9db2b24f1497ea5ae6fc",
    "smallmath": "57dbf3f7e83aa8c5c980387a2216d7bb87e4e7cc",
}


@pytest.mark.parametrize("name,revision", BENCHMARKS.items())
@pytest.mark.parametrize("filename", ["spec.yaml", "hf-jobs-nothink.yaml", "hf-jobs-thinking.yaml"])
def test_math_spec(name, revision, filename):
    path = ROOT / name / filename
    spec = Spec.load(path, env={"TASK_ID": "example", "BOARD_SPACE": "test-board",
                                "SERVER_URL": "https://example.com/v1"})
    assert spec.task.revision == revision
    assert spec.team_file == path.parent / "team.md"
    compile_spec(spec, {"repo": "example", "selector": "example", "revision": revision})
    if filename == "spec.yaml":
        assert spec.inference.urls == ["https://example.com/v1"]
        return
    command = vllm_command(spec.model, spec.inference.hf_jobs, spec.inference.max_working, 1)
    assert command[:3] == ["vllm", "serve", "Qwen/Qwen3.8-27B"]
    if filename == "hf-jobs-thinking.yaml":
        assert command[command.index("--reasoning-parser") + 1] == "qwen3"
        assert spec.inference.reasoning_effort == "high"
        assert "--default-chat-template-kwargs" not in command
    else:
        assert "--reasoning-parser" not in command
        assert command[command.index("--default-chat-template-kwargs") + 1] == (
            '{"enable_thinking": false}'
        )
