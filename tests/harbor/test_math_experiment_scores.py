"""Exercise scoring against synthetic Legion bundles and mocked judge APIs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx

ROOT = Path(__file__).resolve().parents[2] / "experiments"


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / name / "score.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def trial(root, task, agent, answer=None, reward=0, number=1, step=False, proof=None,
          finished_at=None, trial_name=None):
    path = root / "swarm" / agent / "trials" / (trial_name or f"{agent}-t{number}")
    verifier = path / ("steps/check/verifier" if step else "verifier")
    verifier.mkdir(parents=True)
    if "/" not in task:
        dataset = ("imo-answerbench" if task.startswith("imo-bench-") else
                   "imo-proofbench" if task.startswith("pb-") else "smallmath")
        task = f"{dataset}/{task}"
    result = {"task_name": task, "finished_at": finished_at or f"2026-11-27T00:00:{number:02d}Z",
              "verifier_result": {"rewards": {"reward": reward}}}
    if step:
        result["verifier_result"] = None
        result["step_results"] = [{"step_name": "check", "verifier_result": {
            "rewards": {"reward": reward}}}]
    (path / "result.json").write_text(json.dumps(result))
    if answer is not None:
        (verifier / "answer.txt").write_text(answer)
    if proof is not None:
        (verifier / "proof.md").write_text(proof)
        (verifier / "grade.json").write_text(json.dumps({"status": "submitted_for_judge"}))
        (verifier / "reference.json").write_text(json.dumps({"problem": "P", "solution": "S",
            "grading_guidelines": "G", "max_score": 7}))
    return verifier


def answer_args(**overrides):
    values = {"judge": False, "judge_model": "judge", "judge_base_url": "https://judge/v1",
              "judge_api_key_env": "TEST_JUDGE_KEY", "reference_dir": None}
    return SimpleNamespace(**(values | overrides))


def proof_args(**overrides):
    values = {"judge_model": "judge", "judge_base_url": "https://judge/v1",
              "judge_api_key_env": "TEST_JUDGE_KEY", "provider": "openai",
              "selected_agent": None}
    return SimpleNamespace(**(values | overrides))


def openai_reply(value):
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}}]})


def test_answer_aggregation_latest_ties_and_missing(tmp_path):
    mod = module("imo-answerbench")
    task = "imo-bench-algebra-001"
    trial(tmp_path, task, "agent-1", "0", 0)
    trial(tmp_path, task, "agent-1", "3", 1, number=2, step=True)
    trial(tmp_path, task, "agent-2", "3", 1)
    trial(tmp_path, task, "agent-3", "7", 0)
    result = mod.score([tmp_path], answer_args())
    assert result["solo"] == {"count": 3, "mean": 2 / 3}
    assert result["teams"][0]["maj_at_n"] == result["teams"][0]["best_of_n"] == 1
    trial(tmp_path, task, "agent-4", "7", 0)
    result = mod.score([tmp_path], answer_args())
    assert result["teams"][0]["majority_answer"] is None
    assert result["teams"][0]["maj_at_n"] == 0


@respx.mock
def test_answer_dual_order_and_conservative_fallback(tmp_path, monkeypatch):
    mod = module("imo-answerbench")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    task = "imo-bench-algebra-002"
    trial(tmp_path, task, "agent-1", "x + x", 0)
    refs = tmp_path / "references"
    refs.mkdir()
    (refs / f"{task}.json").write_text(json.dumps({"kind": "expression", "answer": "2*x"}))
    (refs / f"{task}.md").write_text("Find all n >= 3 with x = 1.")
    route = respx.post("https://judge/v1/chat/completions").mock(side_effect=[
        openai_reply({"equivalent": True}), openai_reply({"equivalent": False})])
    args = answer_args(judge=True, reference_dir=str(refs))
    result = mod.score([tmp_path], args)
    assert result["teams"][0]["agents"][0]["correct"] == 0
    assert route.call_count == 2
    contents = [json.loads(call.request.content)["messages"][0]["content"] for call in route.calls]
    assert all("Find all n >= 3 with x = 1." in prompt for prompt in contents)
    assert result["teams"][0]["task"] == "imo-answerbench/" + task
    assert result["teams"][0]["task_id"] == task
    assert contents[0].index("x + x") < contents[0].index("2*x")
    assert contents[1].index("2*x") < contents[1].index("x + x")
    route.mock(return_value=openai_reply({"equivalent": True}))
    assert mod.score([tmp_path], args)["teams"][0]["best_of_n"] == 1


def test_smallmath_levels_and_integer_votes(tmp_path):
    mod = module("smallmath")
    task = "smallmath-0015"
    trial(tmp_path, task, "agent-1", "+001", 1)
    trial(tmp_path, task, "agent-2", "1", 1)
    trial(tmp_path, task, "agent-3", "2", 0)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{"task": task, "level": 6}]}))
    result = mod.score([tmp_path], manifest)
    assert result["teams"][0]["task"] == "smallmath/" + task
    assert result["teams"][0]["task_id"] == task
    assert result["levels"]["6"]["solo"]["mean"] == 2 / 3
    assert result["levels"]["6"]["maj_at_n"]["mean"] == 1
    assert result["levels"]["1"]["solo"]["mean"] is None
    manifest.write_text('{"tasks": []}')
    with pytest.raises(ValueError, match="difficulty"):
        mod.score([tmp_path], manifest)


@respx.mock
def test_proof_openai_grades_and_aggregates(tmp_path, monkeypatch):
    mod = module("imo-proofbench")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    for name, proof in [("a", "A proof"), ("b", "A proof"), ("c", "Another proof")]:
        trial(tmp_path, "pb-basic-001", name, reward=1, proof=proof)
    route = respx.post("https://judge/v1/chat/completions").mock(side_effect=[
        openai_reply({"score": n, "reason": "reason"}) for n in (4, 6, 7)])
    records = mod.score([tmp_path], proof_args(selected_agent="a"))
    assert route.call_count == 3
    assert records[0]["percent_of_7"] == 400 / 7
    team = records[-1]
    assert (team["mean"], team["max"], team["consensus"], team["selected"]) == (17 / 3, 7, 5, 4)
    prompt = json.loads(route.calls[0].request.content)["messages"][0]["content"]
    assert '"grading_guidelines": "G"' in prompt
    assert '"reference_solution": "S"' in prompt


@respx.mock
def test_proof_gemini_and_invalid_scores(tmp_path, monkeypatch):
    mod = module("imo-proofbench")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    trial(tmp_path, "pb-basic-001", "a", reward=1, proof="A proof")
    route = respx.post("https://judge/v1/models/judge:generateContent").mock(return_value=httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": '{"score": 7, "reason": "ok"}'}]}}]}))
    result = mod.score([tmp_path], proof_args(provider="gemini"))
    assert result[0]["score"] == 7
    assert route.calls[0].request.headers["x-goog-api-key"] == "test"
    route.mock(return_value=httpx.Response(200, json={"candidates": [{"content": {"parts": [
        {"text": '{"score": 8, "reason": "bad"}'}]}}]}))
    with pytest.raises(ValueError, match="0–7"):
        mod.score([tmp_path], proof_args(provider="gemini"))


def test_pinned_reference_download(tmp_path, monkeypatch):
    mod = module("imo-answerbench")
    reference = tmp_path / "ref.json"
    reference.write_text('{"kind": "literal", "answer": "yes"}')
    calls = []
    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(reference)
    monkeypatch.setattr(mod, "hf_hub_download", download)
    assert mod.reference("imo-bench-algebra-002", None)["answer"] == "yes"
    assert calls == [(mod.REPO, "tasks/imo-bench-algebra-002/tests/reference.json",
                      {"repo_type": "dataset", "revision": mod.REVISION})]


def test_missing_verification_and_invalid_proof(tmp_path):
    answers = module("imo-answerbench")
    proof = module("imo-proofbench")
    verifier = trial(tmp_path / "answers", "imo-bench-algebra-001", "a", "2", 0)
    result = verifier.parent / "result.json"
    result.write_text('{"task_name": "imo-answerbench/imo-bench-algebra-001"}')
    with pytest.raises(ValueError, match="Missing binary verifier reward"):
        answers.score([tmp_path / "answers"], answer_args())
    verifier = trial(tmp_path / "proofs", "pb-basic-001", "a", reward=0, proof="partial")
    (verifier / "grade.json").write_text('{"status": "empty"}')
    records = proof.score([tmp_path / "proofs"], proof_args())
    assert records[0]["score"] == 0
    assert records[-1]["mean_percent_of_7"] == 0


@respx.mock
def test_judge_invalid_answer_and_http_failure(tmp_path, monkeypatch):
    mod = module("imo-answerbench")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    task = "imo-bench-algebra-002"
    trial(tmp_path, task, "a", "x", 0)
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / f"{task}.json").write_text('{"kind":"literal","answer":"y"}')
    (refs / f"{task}.md").write_text("Problem statement")
    route = respx.post("https://judge/v1/chat/completions").mock(
        return_value=openai_reply({"equivalent": "yes"}))
    with pytest.raises(ValueError, match="boolean"):
        mod.score([tmp_path], answer_args(judge=True, reference_dir=str(refs)))
    route.mock(return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        mod.score([tmp_path], answer_args(judge=True, reference_dir=str(refs)))


def test_multiple_runs_not_pooled_and_missing_answers(tmp_path):
    mod = module("smallmath")
    task = "smallmath-0015"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{"task": task, "level": 6}]}))
    trial(tmp_path / "one", task, "a", "1", 1)
    trial(tmp_path / "two", task, "a", None, 0)
    result = mod.score([tmp_path / "one", tmp_path / "two"], manifest)
    assert len(result["teams"]) == 2
    assert result["overall"]["maj_at_n"] == {"count": 2, "mean": 0.5}


def test_smallmath_pinned_manifest_download(tmp_path, monkeypatch):
    mod = module("smallmath")
    task = "smallmath-0015"
    trial(tmp_path, task, "a", "1", 1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{"task": task, "level": 6}]}))
    calls = []
    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(manifest)
    monkeypatch.setattr(mod, "hf_hub_download", download)
    assert mod.score([tmp_path], None)["teams"][0]["maj_at_n"] == 1
    assert calls == [("HuggingFaceSwarms/SmallMath", "tasks/manifest.json",
                      {"repo_type": "dataset", "revision": mod.REVISION})]


def test_score_cli_json(tmp_path):
    import subprocess
    import sys

    task = "smallmath-0015"
    trial(tmp_path / "result", task, "a", "1", 1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{"task": task, "level": 6}]}))
    output = subprocess.run([sys.executable, str(ROOT / "smallmath/score.py"),
                             str(tmp_path / "result"), "--manifest", str(manifest)],
                            check=True, capture_output=True, text=True)
    assert json.loads(output.stdout)["overall"]["solo"]["mean"] == 1


def test_latest_completion_wins_over_attempt_after_worker_restart(tmp_path):
    mod = module("smallmath")
    task = "smallmath-0015"
    trial(tmp_path, task, "a", "0", 0, number=2, trial_name="old-worker-a-t2",
          finished_at="2026-11-27T10:00:00+00:00")
    trial(tmp_path, task, "a", "83", 1, number=1, trial_name="new-worker-a-t1",
          finished_at="2026-11-27T06:01:00-04:00")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{"task": task, "level": 6}]}))
    result = mod.score([tmp_path], manifest)
    assert result["teams"][0]["agents"][0]["answer"] == "83"
    assert result["teams"][0]["maj_at_n"] == 1


def test_missing_timestamp_with_multiple_trials_is_ambiguous(tmp_path):
    mod = module("smallmath")
    task = "smallmath-0015"
    old = trial(tmp_path, task, "a", "0", 0, number=2)
    result_path = old.parent / "result.json"
    data = json.loads(result_path.read_text())
    data["finished_at"] = None
    result_path.write_text(json.dumps(data))
    trial(tmp_path, task, "a", "83", 1, number=1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{"task": task, "level": 6}]}))
    with pytest.raises(ValueError, match="without finished_at"):
        mod.score([tmp_path], manifest)


def test_rejected_oversized_integer_spellings_stay_distinct(tmp_path):
    mod = module("smallmath")
    task = "smallmath-0015"
    trial(tmp_path, task, "accepted", "83", 1)
    trial(tmp_path, task, "zeros", "0" * 1000 + "83", 0)
    trial(tmp_path, task, "huge", "9" * 5000, 0)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{"task": task, "level": 6}]}))
    result = mod.score([tmp_path], manifest)
    answers = {row["agent"]: row["answer"] for row in result["teams"][0]["agents"]}
    assert answers == {"accepted": "83", "zeros": "0" * 1000 + "83", "huge": "9" * 5000}
    assert result["teams"][0]["majority_answer"] is None


def test_pinned_instruction_download(tmp_path, monkeypatch):
    mod = module("imo-answerbench")
    instruction = tmp_path / "instruction.md"
    instruction.write_text("Prove the statement for n >= 3.")
    calls = []
    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(instruction)
    monkeypatch.setattr(mod, "hf_hub_download", download)
    assert "n >= 3" in mod.dataset_file("imo-bench-algebra-002", None, "instruction.md")
    assert calls == [(mod.REPO, "tasks/imo-bench-algebra-002/instruction.md",
                      {"repo_type": "dataset", "revision": mod.REVISION})]


def test_completion_tie_uses_attempt_number(tmp_path):
    mod = module("smallmath")
    task = "smallmath-0015"
    for number, answer, reward in [(1, "0", 0), (2, "83", 1)]:
        trial(tmp_path, task, "a", answer, reward, number=number,
              finished_at="2026-11-27T10:00:00Z")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{"task": task, "level": 6}]}))
    assert mod.score([tmp_path], manifest)["teams"][0]["agents"][0]["answer"] == "83"


def test_problem_and_reference_from_dataset_checkout(tmp_path):
    mod = module("imo-answerbench")
    task = "imo-bench-algebra-002"
    task_dir = tmp_path / "tasks" / task
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "reference.json").write_text('{"kind":"literal","answer":"yes"}')
    (task_dir / "instruction.md").write_text("Only consider n >= 3.")
    assert mod.reference(task, tmp_path)["answer"] == "yes"
    assert mod.dataset_file(task, tmp_path, "instruction.md") == "Only consider n >= 3."


def test_smallmath_votes_do_not_depend_on_correctness(tmp_path):
    mod = module("smallmath")
    task = "smallmath-0015"
    for agent, answer, reward in [("a", "83", 1), ("b", "+083", 1),
                                   ("c", "1", 0), ("d", "+001", 0)]:
        trial(tmp_path, task, agent, answer, reward)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{"task": task, "level": 6}]}))
    team = mod.score([tmp_path], manifest)["teams"][0]
    assert [row["answer"] for row in team["agents"]] == ["83", "83", "1", "1"]
    assert team["majority_answer"] is None
    assert team["maj_at_n"] == 0


def test_smallmath_integer_canonicalization_respects_verifier_bounds():
    mod = module("smallmath")
    assert mod.canonical("0" * 999 + "1") == "1"
    assert mod.canonical("+" + "0" * 998 + "1") == "1"
    assert mod.canonical("+" + "0" * 999 + "1") == "+" + "0" * 999 + "1"
    assert mod.canonical("1" * 5000) == "1" * 5000
    assert mod.canonical("1.0") == "1.0"
