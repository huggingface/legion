"""SmolTalkBench evals (experiments/run4/eval): the AlpacaEval port, the SwarmChat server, the
Inspect task with mock models, and the Job helpers. Nothing here calls the Hub or a judge.
The expected AlpacaEval values were computed with the alpaca_eval package (0.6.x)."""

import importlib.util
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from .test_world import ROOT

EVAL = ROOT / "experiments" / "run4" / "eval"


def load(name):
    spec = importlib.util.spec_from_file_location(f"smoltalk_{name}", EVAL / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


judge = load("judge")


def test_order_switches_as_alpaca_eval():
    expected = {"What's a good way to keep my houseplants alive while I'm away for a week?": False,
                "Can you recommend a good board game for a family night?": False,
                "Hi": True, "What is photosynthesis?": True, "How do I boil an egg?": False,
                "Tell me a joke about cats.": False}
    assert {i: judge.is_switched(i) for i in expected} == expected


def test_judge_messages_fill_the_template_in_order():
    system, user = judge.judge_messages("Name a fruit.", "Apple.", "A {braced} pear.")
    assert system == {"role": "system", "content": judge.SYSTEM}
    assert '"instruction": """Name a fruit.""",' in user["content"]
    assert '"model_identifier": "m",\n        "output": """Apple."""' in user["content"]
    assert '"model_identifier": "M",\n        "output": """A {braced} pear."""' in user["content"]
    assert user["content"].endswith("## Best Model Identifier")


@pytest.mark.parametrize(("top", "expected"), [
    ({"M": -0.0176, "m": -4.05, "Both": -14.9}, 1.9825772133114985),
    ({"m": -0.2, "M": -1.7}, 1.1824255238063563),
    ({"M": -0.01}, 2.0),
    ({"m": -0.3}, 1.0),
])
def test_preference_from_top_logprobs(top, expected):
    assert judge.preference(top) == pytest.approx(expected, abs=1e-12)


def test_preference_is_nan_without_either_identifier():
    assert math.isnan(judge.preference({"The": -0.1, "Based": -2.0}))


def synthetic():
    n = 60
    ref = [40 + (i * 37) % 200 for i in range(n)]
    out = [10 + (i * 53) % 400 for i in range(n)]
    prefs = [1 + min(1.0, max(0.0, 0.3 + (o - r) / 500 + 0.25 * math.sin(i)))
             for i, (o, r) in enumerate(zip(out, ref, strict=True))]
    prefs[7], prefs[11] = 1.5, math.nan
    return prefs, out, ref


def test_win_rates_match_alpaca_eval():
    prefs, _, _ = synthetic()
    assert judge.win_rates(prefs) == pytest.approx({
        "win_rate": 43.7161961129, "standard_error": 3.7640881026,
        "discrete_win_rate": 41.5254237288, "n_wins": 24, "n_wins_base": 34, "n_draws": 1,
        "n_total": 59}, abs=1e-9)


def test_length_controlled_win_rate_matches_alpaca_eval():
    pytest.importorskip("sklearn")
    prefs, out, ref = synthetic()
    lc = judge.length_controlled(prefs, out, ref, bootstrap=20)
    # 5e-8 off: rows the fit used to drop for a zero weight now carry PROB_EPS, which moves
    # liblinear's solution within its tolerance
    assert lc["length_controlled_win_rate"] == pytest.approx(29.5837587498, abs=1e-6)
    assert 0 < lc["lc_standard_error"] < 20


def test_length_controlled_win_rate_without_wins_or_losses():
    pytest.importorskip("sklearn")
    lengths, reference = list(range(10, 1010)), [500] * 1000
    # Nothing to fit: the GLM alone would put both at 50%
    assert judge.length_controlled([1.0] * 1000, lengths, reference, bootstrap=5) == {
        "length_controlled_win_rate": 0, "lc_standard_error": 0}
    assert judge.length_controlled([2.0] * 1000, lengths, reference, bootstrap=5) == {
        "length_controlled_win_rate": 100, "lc_standard_error": 0}
    # One win, or one loss: a cross-validation fold can miss it (this was 17% and 99.86%)
    one_win = judge.length_controlled([2.0] + [1.0] * 999, [10] * 1000, [10] * 1000, bootstrap=2)
    one_loss = judge.length_controlled([1.0] + [2.0] * 999, [10] * 1000, [10] * 1000, bootstrap=2)
    assert one_win["length_controlled_win_rate"] == pytest.approx(
        100 - one_loss["length_controlled_win_rate"], abs=1e-3)
    assert one_win["length_controlled_win_rate"] < 0.2
    # Three wins: the resamples that miss them all are 0%, not 50% (the SE was 14 points)
    rng = random.Random(0)
    lc = judge.length_controlled([2.0] * 3 + [1.0] * 997, [rng.randint(5, 50) for _ in range(1000)],
                                 [20] * 1000, bootstrap=50)
    assert lc["length_controlled_win_rate"] < 1 and lc["lc_standard_error"] < 1


# A bot whose replies expose everything that could differ between conversations: module
# state, the global and per-instance random generators, the conversation's memory and the hash
# seed. id(memory) comes last: object ids move with address randomisation, so they are left out.
PROBE = '''import random

CALLS = [0]
OWN = random.Random()


def respond(text, memory):
    if text == "hang":
        while True:
            pass
    CALLS[0] += 1
    memory["turns"] = memory.get("turns", 0) + 1
    return " ".join(map(str, [text, CALLS[0], memory["turns"], random.random(), OWN.random(),
                              random.Random().random(), hash("swarm"), "id", id(memory)]))
'''


def without_ids(replies):
    return tuple(r.split(" id ")[0] for r in replies)


@pytest.fixture
def probe_tree(tmp_path):
    tree = tmp_path / "bot-tree"
    shutil.copytree(ROOT / "worlds" / "chatbot4" / "seed" / "bot", tree / "bot")
    (tree / "bot" / "modules" / "probe.py").write_text(PROBE)
    return tree


@pytest.fixture
def server():
    return load("swarmchat_server")


def test_every_conversation_starts_from_the_loaded_state(server, probe_tree, tmp_path):
    with open(tmp_path / "bot.log", "w") as log:
        bot = server.Bot(probe_tree, seed=0, timeout=2, workers=3, log=log)
        try:
            assert bot.ready["handlers"] == 2
            with ThreadPoolExecutor(8) as pool:
                results = list(pool.map(bot.ask, [["hi", "again"]] * 30 + [["hi"]] * 6))
            again = bot.ask(["hi", "again"])
            hang = bot.ask(["hang"])
            after = bot.ask(["hi"])
        finally:
            bot.close()
    assert {r["status"] for r in results} == {"ok"}
    two = {without_ids(r["replies"]) for r in results[:30]} | {without_ids(again["replies"])}
    assert len(two) == 1, "the same conversation got different replies"
    first, second = next(iter(two))
    # State lives within a conversation and never leaks into the next one.
    assert first.split()[1:3] == ["1", "1"] and second.split()[1:3] == ["2", "2"]
    assert {without_ids(r["replies"]) for r in results[30:]} == {(first,)}
    assert hang == {"replies": [], "status": "worker exited"}
    assert without_ids(after["replies"]) == (first,)
    # A second loader agrees too.
    with open(tmp_path / "bot.log", "a") as log:
        rerun = server.Bot(probe_tree, seed=0, timeout=2, workers=1, log=log)
        try:
            replies = rerun.ask(["hi", "again"])["replies"]
        finally:
            rerun.close()
    assert without_ids(replies) == (first, second)


def test_endpoint_answers_the_user_turns_as_one_conversation(server, probe_tree, tmp_path):
    with open(tmp_path / "bot.log", "w") as log:
        bot = server.Bot(probe_tree, seed=0, timeout=2, workers=2, log=log)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.handler(bot, "SwarmChat"))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{httpd.server_port}"
        try:
            def post(body):
                req = urllib.request.Request(f"{url}/v1/chat/completions",
                                             data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req) as response:
                    return json.load(response)

            messages = [{"role": "system", "content": "ignored"},
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "ignored"},
                        {"role": "user", "content": [{"type": "text", "text": "again\nnow"}]}]
            body = post({"model": "SwarmChat", "messages": messages})
            direct = bot.ask(["hi", "again now"])
            with urllib.request.urlopen(f"{url}/health") as response:
                health = json.load(response)
            with pytest.raises(urllib.error.HTTPError, match="400") as refused:
                post({"messages": messages, "stream": True})
            refused.value.close()
        finally:
            httpd.shutdown()
            bot.close()
    assert body["model"] == "SwarmChat"
    assert without_ids([body["choices"][0]["message"]["content"]]) == without_ids(direct["replies"][-1:])
    assert health["ok"] and health["model"] == "SwarmChat"


def test_sigterm_stops_the_bot_with_the_server(probe_tree, tmp_path):
    proc = subprocess.Popen([sys.executable, str(EVAL / "swarmchat_server.py"), str(probe_tree),
                             "--port", "0", "--workers", "2", "--log", str(tmp_path / "bot.log")],
                            stdout=subprocess.PIPE, text=True)
    try:
        assert "loaded in" in proc.stdout.readline()
        loader = int(subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True,
                                    text=True, check=True).stdout.split()[0])
        os.killpg(loader, 0)  # the loader's group: the loader and its waiting forks
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == 0
    finally:
        proc.kill()
        proc.stdout.close()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.killpg(loader, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail("the bot outlived the server")


def test_task_scores_the_model_against_the_reference(tmp_path):
    pytest.importorskip("inspect_ai")
    pytest.importorskip("sklearn")
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.log import read_eval_log
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        Logprob,
        Logprobs,
        ModelOutput,
        TopLogprob,
        get_model,
    )

    bench = load("smoltalk_bench")
    hf_job = load("hf_job")
    # The judge always prefers "m" with P = 0.9. "Hi" is switched (the model's reply is "m"),
    # "How do I boil an egg?" is not (the reference is "m").
    top = [TopLogprob(token="m", logprob=math.log(0.9)), TopLogprob(token="M", logprob=math.log(0.1))]
    verdict = ModelOutput(model="judge", choices=[ChatCompletionChoice(
        message=ChatMessageAssistant(content="m"), stop_reason="max_tokens",
        logprobs=Logprobs(content=[Logprob(token="m", logprob=math.log(0.9), top_logprobs=top)]))])
    grader = get_model("mockllm/judge", custom_outputs=[verdict] * 2)
    dataset = MemoryDataset([Sample(id="hi", input="Hi", target="Hello! How can I help?"),
                             Sample(id="egg", input="How do I boil an egg?", target="Boil it.")])
    log = inspect_eval(bench.smoltalk_bench(dataset), model="mockllm/model",
                       model_roles={"grader": grader}, log_dir=str(tmp_path), display="none",
                       metadata={"model_id": "mock", "revision": "r1"})[0]
    assert log.status == "success"
    log = read_eval_log(log.location)
    scores = {s.id: s.scores["alpaca_eval_judge"] for s in log.samples}
    assert scores["hi"].value == pytest.approx(0.9) and scores["hi"].metadata["switched"]
    assert scores["egg"].value == pytest.approx(0.1) and not scores["egg"].metadata["switched"]
    summary = hf_job.build_summary(log)
    assert summary["metrics"]["win_rate"] == pytest.approx(50)
    assert summary["metrics"]["n_total"] == 2
    assert summary["primary_metric"]["name"] == "length_controlled_win_rate"
    # Too few samples for the GLM: NaN, written as null.
    assert summary["primary_metric"]["value"] is None
    assert summary["metadata"] | {"judged_samples": 2, "model_id": "mock",
                                  "model_roles": {"grader": "mockllm/judge"}} == summary["metadata"]
    json.dumps(summary)


def test_a_sample_that_errors_still_counts(tmp_path):
    pytest.importorskip("inspect_ai")
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.log import read_eval_log
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        Logprob,
        Logprobs,
        ModelOutput,
        TopLogprob,
        get_model,
    )
    from inspect_ai.solver import solver

    bench, evaluate, hf_job = load("smoltalk_bench"), load("evaluate"), load("hf_job")

    @solver
    def fails_on(sample_id):
        async def solve(state, generate):
            if state.sample_id == sample_id:
                raise RuntimeError("the model server went away")
            return await generate(state)
        return solve

    # The judge prefers the reference ("M" on "egg", which is not switched) with P = 0.9
    top = [TopLogprob(token="M", logprob=math.log(0.9)), TopLogprob(token="m", logprob=math.log(0.1))]
    verdict = ModelOutput(model="judge", choices=[ChatCompletionChoice(
        message=ChatMessageAssistant(content="M"), stop_reason="max_tokens",
        logprobs=Logprobs(content=[Logprob(token="M", logprob=math.log(0.9), top_logprobs=top)]))])
    grader = get_model("mockllm/judge", custom_outputs=[verdict] * 4)
    dataset = MemoryDataset([Sample(id="hi", input="Hi", target="Hello! How can I help?"),
                             Sample(id="egg", input="How do I boil an egg?", target="Boil it.")])
    task = Task(dataset=dataset, solver=fails_on("egg"), scorer=bench.alpaca_eval_judge())
    log = inspect_eval(task, model="mockllm/model", model_roles={"grader": grader},
                       log_dir=str(tmp_path), display="none", **evaluate.ON_ERROR)[0]
    assert log.status == "success"
    log = read_eval_log(log.location)
    egg = next(s for s in log.samples if s.id == "egg")
    assert egg.error is not None and egg.scores["alpaca_eval_judge"].answer == ""
    summary = hf_job.build_summary(log)
    assert summary["metrics"]["n_total"] == 2 and summary["metadata"]["error_samples"] == 1


def test_swarmchat_revisions_sort_by_commit_time():
    evaluate = load("evaluate")
    paris = timezone(timedelta(hours=1))
    earlier = evaluate.swarmchat_revision("fdef14e9f8e3d798", datetime(2026, 10, 28, 17, 27, 18,
                                                                         tzinfo=paris))
    later = evaluate.swarmchat_revision("0aaa", datetime(2026, 10, 29, 0, 44, tzinfo=UTC))
    assert earlier == "20261028T162718Z-fdef14e9f8e3"
    assert earlier < later
    assert evaluate.log_model_id("hf-inference-providers/org/model:fireworks-ai") == "org/model"
    assert (evaluate._safe_name("smoltalk_bench_gold-HuggingFaceTB/SmolLM-1.7B-Instruct-main")
            == "smoltalk_bench_gold-HuggingFaceTB-SmolLM-1-7B-Instruct-main")


def test_job_files_go_to_the_output_bucket_namespace(monkeypatch, capsys):
    evaluate = load("evaluate")
    calls = {}

    class Api:
        def __init__(self, token):
            pass

        def create_bucket(self, bucket, **kwargs):
            calls["bucket"] = bucket

        def sync_job_volume(self, source, mount_path, **kwargs):
            calls["files"] = {p.relative_to(source).as_posix()
                              for p in Path(source).rglob("*") if p.is_file()}
            calls["volume_namespace"] = kwargs["namespace"]
            return "volume"

        def run_job(self, **kwargs):
            calls["job_namespace"], calls["volumes"] = kwargs["namespace"], kwargs["volumes"]
            return types.SimpleNamespace(url="https://huggingface.co/jobs/lewtun/1")

    monkeypatch.setattr(evaluate, "HfApi", Api)
    monkeypatch.setattr(evaluate, "get_token", lambda: "token")
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--backend", "endpoint", "--model",
                                      "hf-inference-providers/org/model", "--namespace", "lewtun"])
    evaluate.main()
    assert calls["bucket"] == "HuggingFaceSwarms/swarmchat-evals"
    assert calls["volume_namespace"] == "HuggingFaceSwarms"
    assert calls["job_namespace"] == "lewtun" and calls["volumes"] == ["volume"]
    assert "experiments/run4/eval/hf_job.py" in calls["files"]


def test_swarmchat_runs_one_conversation_per_vcpu(monkeypatch, capsys):
    evaluate = load("evaluate")
    committed = datetime(2026, 10, 29, 0, 44, 31, tzinfo=UTC)

    class Api:
        def __init__(self, token):
            pass

        def dataset_info(self, repo_id, revision):
            return types.SimpleNamespace(sha="d8645ac9e103" + "0" * 28, last_modified=committed)

        def list_jobs_hardware(self):
            return [types.SimpleNamespace(name="cpu-basic", cpu="2 vCPU", accelerator=None),
                    types.SimpleNamespace(name="cpu-upgrade", cpu="8 vCPU", accelerator=None),
                    types.SimpleNamespace(name="cpu-xl", cpu="16 vCPU", accelerator=None)]

    monkeypatch.setattr(evaluate, "HfApi", Api)
    monkeypatch.setattr(evaluate, "get_token", lambda: "token")
    for extra, flavor, workers in (([], "cpu-upgrade", "8"), (["--flavor", "cpu-xl"], "cpu-xl", "16")):
        monkeypatch.setattr(sys, "argv", ["evaluate.py", "--model", "SwarmChat", "--namespace",
                                          "lewtun", "--dry_run", *extra])
        evaluate.main()
        [job] = json.loads(capsys.readouterr().out)
        command = job["command"]
        assert job["flavor"] == flavor
        assert "/HuggingFaceSwarms/SwarmChat-chatbot-v4/20261029T004431Z-d8645ac9e103/" in job["output_url"]
        assert command[command.index("--workers") + 1] == workers
        assert json.loads(command[command.index("--eval_kwargs") + 1])["max_connections"] == int(workers)
