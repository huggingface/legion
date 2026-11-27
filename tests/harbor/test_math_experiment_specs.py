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
    compiled = compile_spec(spec, {"repo": "example", "selector": "example", "revision": revision})
    assert compiled.types["agent"].env["LEGION_RESULTS_DIR"] == "./results"
    if filename == "spec.yaml":
        assert spec.inference.urls == ["https://example.com/v1"]
        return
    command = vllm_command(spec.model, spec.inference.hf_jobs, spec.inference.max_working, 1)
    expected = ["vllm", "serve", "Qwen/Qwen3.8-27B", "--host", "0.0.0.0", "--port", "8000",
                "--served-model-name", "Qwen/Qwen3.8-27B", "--enable-auto-tool-choice",
                "--tool-call-parser", "qwen3_coder"]
    if filename == "hf-jobs-thinking.yaml":
        expected += ["--reasoning-parser", "qwen3"]
        assert spec.inference.reasoning_effort == "xhigh"
    expected += ["--tensor-parallel-size", "1", "--max-num-seqs", "4",
                 "--max-model-len", "131072"]
    if filename == "hf-jobs-nothink.yaml":
        expected += ["--default-chat-template-kwargs", '{"enable_thinking": false}']
    assert command == expected


@pytest.mark.parametrize("name", BENCHMARKS)
def test_shared_server_for_different_tasks(name):
    path = ROOT / name / "spec.yaml"
    base = {"BOARD_SPACE": "test-board", "SERVER_URL": "https://example.com/v1",
            "REASONING_EFFORT": "xhigh"}
    first = Spec.load(path, env={**base, "TASK_ID": "task-one"})
    second = Spec.load(path, env={**base, "TASK_ID": "task-two"})
    assert first.task.selector != second.task.selector
    for spec in (first, second):
        swarm = compile_spec(spec, {"repo": "example", "selector": spec.task.selector,
                                    "revision": BENCHMARKS[name]})
        assert swarm.inference.urls == ["https://example.com/v1"]
        assert swarm.inference.reasoning_effort == "xhigh"


@pytest.mark.parametrize("name,revision", BENCHMARKS.items())
@pytest.mark.parametrize("filename", ["solo.yaml", "hf-jobs-solo-nothink.yaml"])
def test_solo_math_spec(name, revision, filename):
    path = ROOT / name / filename
    spec = Spec.load(path, env={"TASK_ID": "example", "SERVER_URL": "https://example.com/v1"})
    assert spec.task.revision == revision
    assert spec.agents.count == 1 and spec.workers.agents_per_worker == 1
    assert spec.team_file is None and "message_board" not in spec.resources
    compiled = compile_spec(spec, {"repo": "example", "selector": "example", "revision": revision})
    assert compiled.types["agent"].replicas == 1
    if filename == "solo.yaml":
        assert spec.inference.urls == ["https://example.com/v1"]
        assert spec.inference.reasoning_effort == "off"
    else:
        assert spec.inference.max_working == 1
        assert vllm_command(spec.model, spec.inference.hf_jobs, 1, 1) == [
            "vllm", "serve", "Qwen/Qwen3.8-27B", "--host", "0.0.0.0", "--port", "8000",
            "--served-model-name", "Qwen/Qwen3.8-27B", "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder", "--tensor-parallel-size", "1",
            "--max-num-seqs", "1", "--max-model-len", "131072",
            "--default-chat-template-kwargs", '{"enable_thinking": false}',
        ]
