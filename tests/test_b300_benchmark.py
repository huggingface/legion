"""No GPUs/network: verify EOS accounting, failures, tool parsing and resource layouts."""

import asyncio
import importlib.util
import json
import subprocess
from pathlib import Path

import httpx
import pytest

BASE = Path(__file__).resolve().parents[1] / "experiments/agent-escape/b300"
spec = importlib.util.spec_from_file_location("b300_benchmark", BASE / "benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
spec = importlib.util.spec_from_file_location("b300_prepare", BASE / "prepare.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def row(tools=False):
    r = {
        "id": "real-row",
        "sha256": "abc",
        "prompt_tokens": 300,
        "payload": {"messages": [{"role": "user", "content": "Explain photosynthesis."}]},
    }
    if tools:
        r["payload"].update(
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "parameters": {
                            "type": "object",
                            "required": ["n"],
                            "properties": {"n": {"type": "integer"}},
                        },
                    },
                }
            ],
        )
    return r


def stream(events, done=True):
    return "".join("data: " + json.dumps(e) + "\n\n" for e in events) + (
        "data: [DONE]\n\n" if done else ""
    )


@pytest.mark.parametrize("tools", [False, True])
def test_generation_waits_for_eos(tools):
    p = bench.payload_for(row(tools))
    assert p["max_tokens"] == bench.CONTEXT - 300
    assert "ignore_eos" not in p and "min_tokens" not in p and "stop" not in p
    assert p["reasoning_effort"] == p["chat_template_kwargs"]["reasoning_effort"] == "low"


@pytest.mark.parametrize("finish,status", [("stop", "completed"), ("length", "context_limit")])
async def test_streaming_usage_and_finish(finish, status):
    events = [
        {"choices": [{"delta": {"reasoning_content": "think"}}]},
        {"choices": [{"delta": {"content": "answer"}, "finish_reason": finish}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 300,
                "completion_tokens": 10,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        },
    ]

    def handle(request):
        assert json.loads(request.content)["stream"]
        return httpx.Response(200, text=stream(events))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        r = await bench.measure(client, "http://test", row())
    assert r["status"] == status
    assert r["output_tokens"] == 10  # reasoning is a subset, not another seven tokens
    assert r["ttft_s"] is not None
    assert r["response"]["reasoning_content"] == "think"
    summary = bench.summarize([r], 2, 2)
    assert summary["output_tokens_s"] == (5 if status == "completed" else 0)
    assert summary["total_tokens_s"] == (155 if status == "completed" else 0)


async def test_tool_fragments_and_argument_validation():
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call1",
                                "function": {"name": "read", "arguments": '{"n":'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "12}"}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 6}},
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=stream(events)))
    ) as client:
        r = await bench.measure(client, "http://test", row(True))
    assert r["status"] == "completed"
    assert r["response"]["tool_calls"][0]["function"]["arguments"] == '{"n":12}'


@pytest.mark.parametrize(
    "arguments,status",
    [("bad", "malformed_tool_arguments"), ('{"n":"bad"}', "invalid_tool_arguments")],
)
def test_invalid_tools(arguments, status):
    assert (
        bench.tool_error(
            {0: {"function": {"name": "read", "arguments": arguments}}},
            row(True)["payload"]["tools"],
        )
        == status
    )
    assert bench.tool_error({}, row(True)["payload"]["tools"]) == "missing_tool_call"


@pytest.mark.parametrize(
    "code,body,status",
    [
        (429, "rate limited", "http_error"),
        (200, stream([], done=False), "stream_interrupted"),
        (200, stream([]), "missing_usage"),
    ],
)
async def test_failures_are_not_completions(code, body, status):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(code, text=body))
    ) as client:
        r = await bench.measure(client, "http://test", row())
    assert r["status"] == status


async def test_timeout():
    async def slow(request):
        await asyncio.sleep(1)

    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
        r = await bench.measure(client, "http://test", row(), timeout=0.01)
    assert r["status"] == "timeout"


@pytest.mark.parametrize("layout,gpus", bench.MATRIX)
def test_resource_layout(layout, gpus):
    args = bench.server_command("/model", layout, gpus, True, 8000)
    tp = int(args[args.index("--tensor-parallel-size") + 1])
    dp = int(args[args.index("--data-parallel-size") + 1])
    assert tp * dp == gpus
    assert (dp == 1) if layout == "tp" else (tp == 1)
    assert "--enable-prefix-caching" in args
    assert "--no-enable-prefix-caching" in bench.server_command("/m", layout, gpus, False, 8000)


def test_script_syntax():
    subprocess.run(["bash", "-n", str(BASE / "serve.sbatch")], check=True)


def test_replay_preserves_tool_ids_and_thinking():
    m = prep.wire_message(
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "check"},
                {"type": "toolCall", "id": "a", "name": "read", "arguments": {"n": 12}},
            ],
        }
    )
    assert m["reasoning_content"] == "check"
    assert m["tool_calls"][0]["id"] == "a"
    assert json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"n": 12}
    r = prep.wire_message(
        {
            "role": "toolResult",
            "toolCallId": "a",
            "content": [{"type": "text", "text": "actual result"}],
        }
    )
    assert r == {"role": "tool", "tool_call_id": "a", "content": "actual result"}


def test_stop_process_terminates_group():
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    bench.stop_process(proc)
    assert proc.poll() is not None


@pytest.mark.parametrize(
    "encoded", [[1, 2, 3], {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]
)
def test_token_count_handles_transformers_mapping(encoded):
    assert prep.token_count(encoded) == 3


def test_submit_freezes_code_and_data(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("b300_submit", BASE / "submit.py")
    submitter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submitter)
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{"files": {}}')
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        return "abc123\n" if args[0] == "git" else "12345;cluster\n"

    monkeypatch.setattr(submitter.subprocess, "check_output", command)
    r = submitter.submit(data, tmp_path / "runs")
    assert r["job_id"] == "12345"
    run = Path(r["run"])
    assert (run / "source/benchmark.py").read_bytes() == (BASE / "benchmark.py").read_bytes()
    assert (run / "data/manifest.json").exists()
    assert commands[-1][0] == "sbatch"
    assert "benchmark" in commands[-1]


def test_report_does_not_rank_incomplete_or_failed_groups(tmp_path):
    spec = importlib.util.spec_from_file_location("b300_analyze", BASE / "analyze.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    cases = []
    for repeat in range(3):
        cases.append(
            {
                "dataset": "ultrachat",
                "layout": "tp",
                "gpus": 1,
                "concurrency": 4,
                "repeat": repeat,
                "seconds": 10,
                "completed": 4,
                "requests": 4,
                "output_tokens_s": 100,
                "latency_s_p50": 2,
            }
        )
    cases.append({**cases[0], "gpus": 8, "output_tokens_s": 10000})
    cases.extend({**c, "gpus": 4, "completed": 3, "output_tokens_s": 20000} for c in cases[:3])
    (tmp_path / "summary.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    result = report.analyze(tmp_path)
    assert not result["coverage_complete"]
    assert result["best_measured_throughput"]["gpus"] == 1
    assert result["best_measured_four_agent_latency"] is None
