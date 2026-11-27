"""Exercise scoring against synthetic Legion bundles and mocked judge APIs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

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
    return httpx.Response(200, json={"choices": [{"message": {"content": value if isinstance(value, str) else json.dumps(value)}}]})


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
def test_answer_qed_single_order_prompt_and_verdict(tmp_path, monkeypatch):
    mod = module("imo-answerbench")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    task = "imo-bench-algebra-002"
    trial(tmp_path, task, "agent-1", "x + x", 0)
    refs = tmp_path / "references"
    refs.mkdir()
    (refs / f"{task}.json").write_text(json.dumps({"kind": "expression", "answer": "2*x"}))
    (refs / f"{task}.md").write_text("Find all n >= 3 with x = 1.")
    route = respx.post("https://judge/v1/chat/completions").mock(
        return_value=openai_reply(r"<thinking>Equivalent.</thinking> \boxed{Correct}"))
    args = answer_args(judge=True, reference_dir=str(refs))
    result = mod.score([tmp_path], args)
    assert result["teams"][0]["agents"][0]["correct"] == 1
    assert route.call_count == 1
    payload = json.loads(route.calls[0].request.content)
    template = (ROOT / "qed-nano-prompts/answerbench.txt").read_text()
    assert payload["messages"][0]["content"] == template.format(
        problem_statement="Find all n >= 3 with x = 1.", student_answer="x + x", gold_answer="2*x")
    assert "response_format" not in payload
    assert result["teams"][0]["task"] == "imo-answerbench/" + task
    assert result["teams"][0]["task_id"] == task
    route.mock(return_value=openai_reply(r"<thinking>Not equivalent.</thinking> \boxed{Incorrect}"))
    assert mod.score([tmp_path], args)["teams"][0]["best_of_n"] == 0


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
        openai_reply(f"analysis <points>{n} out of 7</points>") for n in (1, 6, 7)])
    records = mod.score([tmp_path], proof_args(selected_agent="a"))
    assert route.call_count == 3
    assert records[0]["percent_of_7"] == 100 / 7
    team = records[-1]
    assert (team["mean"], team["max"], team["consensus"], team["selected"]) == (14 / 3, 7, 3.5, 1)
    prompt = json.loads(route.calls[0].request.content)["messages"][0]["content"]
    template = (ROOT / "qed-nano-prompts/proofbench.txt").read_text()
    assert prompt == template.format(problem_statement="P", solution="S", guidelines="G",
                                     student_answer="A proof")
    assert "response_format" not in json.loads(route.calls[0].request.content)


@respx.mock
def test_proof_gemini_and_invalid_scores(tmp_path, monkeypatch):
    mod = module("imo-proofbench")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    trial(tmp_path, "pb-basic-001", "a", reward=1, proof="A proof")
    route = respx.post("https://judge/v1/models/judge:generateContent").mock(return_value=httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": "<points>7 out of 7</points>"}]}}]}))
    result = mod.score([tmp_path], proof_args(provider="gemini"))
    assert result[0]["score"] == 7
    assert route.calls[0].request.headers["x-goog-api-key"] == "test"
    route.mock(return_value=httpx.Response(200, json={"candidates": [{"content": {"parts": [
        {"text": "<points>8 out of 7</points>"}]}}]}))
    with pytest.raises(ValueError, match="N in 0, 1, 6, 7"):
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
    import math_eval_common

    monkeypatch.setattr(math_eval_common, "JUDGE_RETRY_DELAYS", (0, 0, 0))
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    task = "imo-bench-algebra-002"
    trial(tmp_path, task, "a", "x", 0)
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / f"{task}.json").write_text('{"kind":"literal","answer":"y"}')
    (refs / f"{task}.md").write_text("Problem statement")
    route = respx.post("https://judge/v1/chat/completions").mock(
        return_value=openai_reply("Maybe"))
    with pytest.raises(ValueError, match="boxed Correct or Incorrect"):
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


def test_binary_comparison_solos_team_and_tokens(tmp_path):
    mod = module("smallmath")
    task = "smallmath-0015"
    roots = []
    for idx, (answer, reward) in enumerate([("83", 1), ("+083", 1), ("1", 0), ("+001", 0)]):
        root = tmp_path / f"solo-{idx}"
        verifier = trial(root, task, "a", answer, reward)
        result = verifier.parent / "result.json"
        data = json.loads(result.read_text())
        data["agent_result"] = {"n_output_tokens": 10 * (idx + 1)}
        result.write_text(json.dumps(data))
        roots.append(root)
    team = tmp_path / "team"
    for idx, (answer, reward) in enumerate([("83", 1), ("83", 1), ("1", 0), ("2", 0)]):
        trial(team, task, f"a{idx}", answer, reward)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{"task": task, "level": 6}]}))
    comparison = mod.score([*roots, team], manifest)["comparison"]
    result = comparison["tasks"][0]
    assert result["solo"]["pass_at_1"] == 0.5
    assert result["solo"]["pass_at_k"] == 1
    assert result["solo"]["indep_maj_at_k"] == 0
    assert result["solo"]["mean_output_tokens"] == {"count": 4, "mean": 25}
    assert result["team"]["maj_at_k"]["mean"] == result["team"]["pass_at_k"]["mean"] == 1
    assert result["team"]["mean_output_tokens"]["mean"] is None
    assert comparison["levels"]["6"]["solo_pass_at_1"]["mean"] == 0.5
    assert comparison["levels"]["1"]["solo_pass_at_1"]["mean"] is None
    partial = mod.score(roots[:3], manifest)["comparison"]["tasks"][0]["solo"]
    assert partial["pass_at_k"] is None and partial["indep_maj_at_k"] is None


def test_answer_comparison_unbiased_pass_at_four(tmp_path):
    mod = module("imo-answerbench")
    task = "imo-bench-algebra-001"
    roots = []
    for idx in range(5):
        root = tmp_path / str(idx)
        trial(root, task, "a", "3" if idx == 0 else "2", int(idx == 0))
        roots.append(root)
    comparison = mod.score(roots, answer_args())["comparison"]["tasks"][0]["solo"]
    assert comparison["n"] == 5
    assert comparison["pass_at_1"] == 0.2
    assert comparison["pass_at_k"] == pytest.approx(0.8)
    assert comparison["indep_maj_at_k"] == 0


@respx.mock
def test_proof_comparison_solo_and_team(tmp_path, monkeypatch):
    mod = module("imo-proofbench")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    roots = []
    for idx in range(4):
        root = tmp_path / f"solo-{idx}"
        trial(root, "pb-basic-001", "a", proof="proof", reward=1)
        roots.append(root)
    team = tmp_path / "team"
    for idx in range(4):
        trial(team, "pb-basic-001", f"a{idx}", proof="proof", reward=1)
    respx.post("https://judge/v1/chat/completions").mock(side_effect=[
        openai_reply(f"<points>{n} out of 7</points>") for n in (1, 1, 6, 7, 1, 1, 6, 7)])
    records = mod.score([*roots, team], proof_args())
    result = mod.compare_proof(records)[0]
    assert result["solo"]["pass_at_1_percent_of_7"]["mean"] == pytest.approx(100 * 15 / 28)
    assert result["solo"]["pass_at_k_ge_6"] == 1
    assert result["solo"]["max_observed_percent_of_7"] == 100
    assert result["team"]["pass_at_k_ge_6"]["mean"] == 1
    assert result["team"]["max_percent_of_7"]["mean"] == 100


def test_team_token_sum_and_missing_solo_token(tmp_path):
    mod = module("smallmath")
    task = "smallmath-0015"
    team = tmp_path / "team"
    for idx in range(4):
        verifier = trial(team, task, f"a{idx}", "83", 1)
        result = verifier.parent / "result.json"
        data = json.loads(result.read_text())
        data["agent_result"] = {"n_output_tokens": 12}
        result.write_text(json.dumps(data))
    solo = tmp_path / "solo"
    trial(solo, task, "a", "83", 1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [{"task": task, "level": 6}]}))
    result = mod.score([solo, team], manifest)["comparison"]["tasks"][0]
    assert result["team"]["mean_output_tokens"] == {"count": 1, "mean": 48}
    assert result["solo"]["mean_output_tokens"] == {"count": 0, "mean": None}
    assert result["solo"]["pass_at_k"] is None


def test_proof_comparison_insufficient_samples():
    mod = module("imo-proofbench")
    rows = [{"type": "team", "task_id": "pb-basic-001", "n": 1, "mean": 5,
             "output_tokens": None}]
    result = mod.compare_proof(rows)[0]
    assert result["solo"]["pass_at_k_ge_6"] is None
    assert result["solo"]["max_observed_percent_of_7"] == 500 / 7
    assert result["team"]["n_runs"] == 0


@respx.mock
def test_selected_team_agent_with_mixed_solo_and_team(tmp_path, monkeypatch):
    mod = module("imo-proofbench")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    solo_roots = []
    for idx in range(2):
        root = tmp_path / f"solo-{idx}"
        trial(root, "pb-basic-001", "agent-001", proof="solo proof", reward=1)
        solo_roots.append(root)
    team_root = tmp_path / "team"
    for idx in range(1, 5):
        trial(team_root, "pb-basic-001", f"agent-{idx:03d}", proof="team proof", reward=1)
    respx.post("https://judge/v1/chat/completions").mock(side_effect=[
        openai_reply(f"<points>{n} out of 7</points>") for n in (1, 0, 1, 6, 7, 6)])
    records = mod.score([*solo_roots, team_root], proof_args(selected_agent="agent-002"))
    teams = [r for r in records if r["type"] == "team"]
    assert all(r["selected"] is None and r["selected_agent"] is None for r in teams[:2])
    assert teams[2]["selected_agent"] == "agent-002"
    assert teams[2]["selected"] == 6
    comparison = mod.compare_proof(records)[0]
    assert comparison["solo"]["n"] == 2 and comparison["team"]["n_runs"] == 1


def test_proof_expected_max_unbiased_and_insufficient_samples():
    mod = module("imo-proofbench")
    scores = [0, 0, 0, 0, 7]
    records = [{"type": "team", "task_id": "pb-basic-001", "n": 1, "mean": score,
                "output_tokens": None} for score in scores]
    comparison = mod.compare_proof(records, 4)
    assert comparison[0]["solo"]["expected_max_at_k_percent_of_7"] == pytest.approx(80.0)
    assert comparison[0]["solo"]["max_observed_percent_of_7"] == 100
    assert comparison[-1]["solo_expected_max_at_k_percent_of_7"]["mean"] == pytest.approx(80.0)
    assert mod.compare_proof(records[:3], 4)[0]["solo"]["expected_max_at_k_percent_of_7"] is None


@pytest.mark.parametrize("name", ["imo-answerbench", "imo-proofbench", "smallmath"])
@pytest.mark.parametrize("k", ["1", "0", "-1"])
def test_comparison_cli_rejects_small_k(name, k):
    import subprocess
    import sys

    command = [sys.executable, str(ROOT / name / "score.py"), "unused-results", "--k", k]
    if name == "imo-proofbench":
        command.extend(["--judge-model", "unused"])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "--k must be >= 2" in result.stderr


def test_comparison_helpers_reject_small_k():
    mod = module("imo-proofbench")
    import math_eval_common

    for k in (1, 0, -1):
        with pytest.raises(ValueError, match="k must be an integer >= 2"):
            math_eval_common.compare_binary([], k)
        with pytest.raises(ValueError, match="k must be an integer >= 2"):
            mod.compare_proof([], k)
        with pytest.raises(ValueError, match="k must be an integer >= 2"):
            math_eval_common.pass_estimate(1, 0, k)


@respx.mock
def test_selected_agent_still_required_for_team(tmp_path, monkeypatch):
    mod = module("imo-proofbench")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    for name in ("agent-001", "agent-003"):
        trial(tmp_path, "pb-basic-001", name, proof="proof", reward=1)
    respx.post("https://judge/v1/chat/completions").mock(
        return_value=openai_reply("<points>1 out of 7</points>"))
    with pytest.raises(ValueError, match="Selected agent agent-002 absent"):
        mod.score([tmp_path], proof_args(selected_agent="agent-002"))


def test_qed_nano_prompt_files_match_pinned_revision():
    import hashlib

    hashes = {"answerbench": "77852b856741cd08de5e0de49f97aacfacd3e2453a561e71d610bfae0c8703a2",
              "proofbench": "1f05fe22b17b20bf670daf6faad21e7904827e86002b54fe420513bf2b065cbd"}
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT / "qed-nano-prompts" / f"{name}.txt").read_bytes()).hexdigest() == digest


def test_qed_boxed_verdicts_and_nested_braces():
    module("imo-answerbench")
    from math_eval_common import find_last_boxed_content, parse_answer_verdict

    assert find_last_boxed_content(r"first \boxed{wrong}; then \boxed{\frac{1}{2}}") == r"wrong,\frac{1}{2}"
    assert find_last_boxed_content(r"\boxed{\boxed{Correct}}") == "Correct"
    assert find_last_boxed_content("\\boxed{first}\n\\boxed{last}") == "last"
    assert parse_answer_verdict(r"\boxed{Incorrect} and finally \boxed{Correct}") is True
    assert parse_answer_verdict(r"\boxed{Correct} and finally \boxed{Incorrect}") is False
    for invalid in ("Correct", r"\boxed{Maybe}", r"\boxed{Correct"):
        with pytest.raises(ValueError, match="boxed"):
            parse_answer_verdict(invalid)


@pytest.mark.parametrize("value", [0, 1, 6, 7])
def test_qed_points_first_tag_and_discrete_buckets(value):
    module("imo-proofbench")
    from math_eval_common import parse_proof_points

    assert parse_proof_points(f"<points>{value} out of 7</points> then <points>7 out of 7</points>") == value
    if value != 7:
        assert parse_proof_points(f"<points>7 out of 7</points><points>{value} out of 7</points>") == 7


@pytest.mark.parametrize("text", ["no points", "<points>2 out of 7</points>",
                                   "<points>7/7</points>",
                                   "<points>invalid</points><points>7 out of 7</points>"])
def test_qed_points_rejects_invalid_first_tag(text):
    module("imo-proofbench")
    from math_eval_common import parse_proof_points

    with pytest.raises(ValueError, match="N in 0, 1, 6, 7"):
        parse_proof_points(text)


def test_qed_self_evaluation_cleanup():
    module("imo-proofbench")
    from math_eval_common import remove_self_evaluation

    for heading in ("# Evaluation", "## Self Evaluation", "### **Final Evaluation**"):
        assert remove_self_evaluation(f"A valid proof.\n\n{heading}\nI deserve 7 points.\nMore") == "A valid proof."
    assert remove_self_evaluation("A valid proof.\n## Details\nProof continues.") == "A valid proof.\n## Details\nProof continues."


@respx.mock
def test_proof_judge_receives_qed_cleaned_proof(tmp_path, monkeypatch):
    mod = module("imo-proofbench")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    trial(tmp_path, "pb-basic-001", "a", reward=1,
          proof="Proof of claim.\n## Self Evaluation\nI deserve 7 points.")
    route = respx.post("https://judge/v1/chat/completions").mock(
        return_value=openai_reply("Justification. <points>6 out of 7</points>"))
    result = mod.score([tmp_path], proof_args())
    template = (ROOT / "qed-nano-prompts/proofbench.txt").read_text()
    request = json.loads(route.calls[0].request.content)
    assert request["messages"][0]["content"] == template.format(
        problem_statement="P", solution="S", guidelines="G", student_answer="Proof of claim.")
    assert result[0]["score"] == 6
    assert "I deserve 7 points." in result[0]["proof"]
    assert "response_format" not in request


@respx.mock
def test_answer_gemini_text_response_and_api_key(tmp_path, monkeypatch):
    mod = module("imo-answerbench")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    task = "imo-bench-algebra-002"
    trial(tmp_path, task, "a", "x", 0)
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / f"{task}.json").write_text('{"kind":"literal","answer":"x"}')
    (refs / f"{task}.md").write_text("Problem statement")
    route = respx.post("https://judge/v1/models/judge:generateContent").mock(return_value=httpx.Response(
        200, json={"candidates": [{"content": {"parts": [
            {"text": "internal thought", "thought": True}, {"text": r"\boxed{Correct}"}]}}]}))
    result = mod.score([tmp_path], answer_args(judge=True, reference_dir=str(refs), provider="gemini"))
    assert result["teams"][0]["best_of_n"] == 1
    assert route.calls[0].request.headers["x-goog-api-key"] == "test"
    assert "generationConfig" not in json.loads(route.calls[0].request.content)


@pytest.mark.parametrize("response", [
    "<thinking>...\\boxed{Correct}...</thinking>\n\\boxed{Incorrect",
    "<thinking>...\\boxed{Correct}...</thinking>",
    "<thinking>...\\boxed{Correct}",
    "\\boxed{Correct} then \\boxed{Incorrect",
    "\\boxed{Correct} then \\boxed",
])
def test_answer_verdict_rejects_incomplete_or_analysis_only(response):
    module("imo-answerbench")
    from math_eval_common import parse_answer_verdict

    with pytest.raises(ValueError):
        parse_answer_verdict(response)


def test_answer_verdict_uses_complete_box_after_thinking():
    module("imo-answerbench")
    from math_eval_common import parse_answer_verdict

    assert parse_answer_verdict("<thinking>\\boxed{Correct}</thinking>\n\\boxed{Incorrect}") is False
    assert parse_answer_verdict("<thinking>\\boxed{Incorrect}</thinking>\n\\boxed{Correct}") is True


@respx.mock
def test_answer_voting_preserves_qed_box_line_boundaries(tmp_path, monkeypatch):
    mod = module("imo-answerbench")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    task = "imo-bench-algebra-002"
    trial(tmp_path, task, "a", "\\boxed{wrong}\n\\boxed{gold}", 0)
    trial(tmp_path, task, "b", "\\boxed{wrong} \\boxed{gold}", 0)
    refs = tmp_path / "references"
    refs.mkdir()
    (refs / f"{task}.json").write_text(json.dumps({"kind": "literal", "answer": "gold"}))
    (refs / f"{task}.md").write_text("Problem statement")
    route = respx.post("https://judge/v1/chat/completions").mock(side_effect=[
        openai_reply(r"<thinking>Matches.</thinking> \boxed{Correct}"),
        openai_reply(r"<thinking>Not equivalent.</thinking> \boxed{Incorrect}"),
    ])
    result = mod.score([tmp_path], answer_args(judge=True, reference_dir=str(refs)))
    rows = result["teams"][0]["agents"]
    assert [row["correct"] for row in rows] == [1, 0]
    assert rows[0]["answer"] != rows[1]["answer"]
    assert result["teams"][0]["majority_answer"] is None
    prompts = [json.loads(call.request.content)["messages"][0]["content"] for call in route.calls]
    assert "Model Solution: gold\n" in prompts[0]
    assert "Model Solution: wrong,gold\n" in prompts[1]


@respx.mock
def test_answer_self_evaluation_indented_heading_keeps_distinct_raw_votes(tmp_path, monkeypatch):
    mod = module("imo-answerbench")
    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    task = "imo-bench-algebra-002"
    trial(tmp_path, task, "a", "## Evaluation\n\\boxed{42}", 0)
    trial(tmp_path, task, "b", " ## Evaluation\n\\boxed{42}", 0)
    refs = tmp_path / "references"
    refs.mkdir()
    (refs / f"{task}.json").write_text(json.dumps({"kind": "literal", "answer": "42"}))
    (refs / f"{task}.md").write_text("Problem statement")
    route = respx.post("https://judge/v1/chat/completions").mock(side_effect=[
        openai_reply(r"<thinking>No answer.</thinking> \boxed{Incorrect}"),
        openai_reply(r"<thinking>Matches.</thinking> \boxed{Correct}"),
    ])
    result = mod.score([tmp_path], answer_args(judge=True, reference_dir=str(refs)))
    rows = result["teams"][0]["agents"]
    assert [row["answer"] for row in rows] == ["## Evaluation\n\\boxed{42}",
                                               " ## Evaluation\n\\boxed{42}"]
    assert [row["correct"] for row in rows] == [0, 1]
    assert route.call_count == 2
    prompts = [json.loads(call.request.content)["messages"][0]["content"] for call in route.calls]
    assert "Model Solution: \n" in prompts[0]
    assert "Model Solution: 42\n" in prompts[1]


def test_answer_offline_votes_keep_evaluation_suffix_distinct(tmp_path):
    mod = module("imo-answerbench")
    task = "imo-bench-algebra-002"
    gold = r"$\left\lfloor \log_{2}a\right\rfloor +1$."
    trial(tmp_path, task, "a", gold, 1)
    trial(tmp_path, task, "b", gold + "\n## Evaluation\nI deserve 7", 0)
    team = mod.score([tmp_path], answer_args())["teams"][0]
    assert [row["correct"] for row in team["agents"]] == [1, 0]
    assert [row["answer"] for row in team["agents"]] == [gold, gold + "\n## Evaluation\nI deserve 7"]
    assert team["majority_answer"] is None and team["maj_at_n"] == 0


@respx.mock
def test_numeric_judge_preserves_offline_votes_and_accepted_box(tmp_path):
    mod = module("imo-answerbench")
    task = "imo-bench-algebra-001"
    pair = tmp_path / "pair"
    trial(pair, task, "a", "3", 1)
    trial(pair, task, "b", "3\n## Evaluation\nI deserve 7", 0)
    single = tmp_path / "single"
    accepted = "## Evaluation\n\\boxed{3}"
    trial(single, task, "a", accepted, 1)
    refs = tmp_path / "references"
    refs.mkdir()
    (refs / f"{task}.json").write_text(json.dumps({"kind": "number", "answer": "3"}))
    route = respx.post("https://judge/v1/chat/completions").mock(
        return_value=openai_reply(r"\boxed{Correct}"))
    results = mod.score([pair, single], answer_args(judge=True, reference_dir=str(refs)))
    first, second = results["teams"]
    assert [row["answer"] for row in first["agents"]] == ["3", "3\n## Evaluation\nI deserve 7"]
    assert [row["correct"] for row in first["agents"]] == [1, 0]
    assert first["majority_answer"] is None and first["maj_at_n"] == 0
    assert second["agents"][0]["answer"] == accepted
    assert second["agents"][0]["correct"] == second["maj_at_n"] == 1
    assert route.call_count == 0


def test_answer_vote_terminal_newlines_and_cleanup_consistency():
    mod = module("imo-answerbench")
    from math_eval_common import remove_self_evaluation

    for text in ("3", " ## Evaluation\n\\boxed{42}", "## Evaluation\n\\boxed{42}"):
        assert mod.normalize(text) == mod.normalize(text + "\r\n\n")
        assert remove_self_evaluation(text) == remove_self_evaluation(text + "\r\n\n")
    assert mod.normalize(" ## Evaluation\n\\boxed{42}") != mod.normalize("## Evaluation\n\\boxed{42}")
    assert mod.normalize(" \t\n") == ""


@pytest.mark.parametrize("status", [408, 429, 500, 503])
@respx.mock
def test_judge_retries_transient_http_errors(status, monkeypatch):
    module("imo-answerbench")
    import math_eval_common

    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    route = respx.post("https://judge/v1/chat/completions").mock(side_effect=[
        httpx.Response(status), openai_reply(r"\boxed{Correct}")])
    delays = []
    answer = math_eval_common.judge_text("prompt", "model", "https://judge/v1", "TEST_JUDGE_KEY",
                                       sleep=delays.append)
    assert answer == r"\boxed{Correct}"
    assert route.call_count == 2 and delays == [1]
    assert route.calls[0].request.extensions["timeout"] == {
        "connect": 180, "read": 180, "write": 180, "pool": 180}


@respx.mock
def test_judge_does_not_retry_client_error(monkeypatch):
    module("imo-answerbench")
    import math_eval_common

    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    route = respx.post("https://judge/v1/chat/completions").mock(return_value=httpx.Response(400))
    delays = []
    with pytest.raises(httpx.HTTPStatusError):
        math_eval_common.judge_text("prompt", "model", "https://judge/v1", "TEST_JUDGE_KEY",
                                   sleep=delays.append)
    assert route.call_count == 1 and delays == []


@respx.mock
def test_judge_retries_empty_response_and_stops_after_four_attempts(monkeypatch):
    module("imo-answerbench")
    import math_eval_common

    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    route = respx.post("https://judge/v1/chat/completions").mock(side_effect=[
        httpx.Response(200, content=b""), httpx.Response(200, content=b" \n"),
        openai_reply(" "), openai_reply(r"\boxed{Correct}")])
    delays = []
    assert math_eval_common.judge_text("p", "m", "https://judge/v1", "TEST_JUDGE_KEY",
                                      sleep=delays.append) == r"\boxed{Correct}"
    assert route.call_count == 4 and delays == [1, 2, 4]
    route.mock(return_value=openai_reply(""))
    delays.clear()
    with pytest.raises(math_eval_common.EmptyJudgeResponse):
        math_eval_common.judge_text("p", "m", "https://judge/v1", "TEST_JUDGE_KEY",
                                   sleep=delays.append)
    assert route.call_count == 8 and delays == [1, 2, 4]


@pytest.mark.parametrize("error", [TimeoutError("slow"), httpx.ConnectTimeout("slow"), URLError("offline")])
@respx.mock
def test_judge_retries_timeouts(error, monkeypatch):
    module("imo-answerbench")
    import math_eval_common

    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    route = respx.post("https://judge/v1/chat/completions").mock(side_effect=[
        error, openai_reply("finished")])
    delays = []
    assert math_eval_common.judge_text("p", "m", "https://judge/v1", "TEST_JUDGE_KEY",
                                      sleep=delays.append) == "finished"
    assert route.call_count == 2 and delays == [1]


@pytest.mark.parametrize("gemini_key", [None, ""])
@respx.mock
def test_gemini_google_api_key_fallback(gemini_key, monkeypatch):
    module("imo-proofbench")
    import math_eval_common

    if gemini_key is None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GEMINI_API_KEY", gemini_key)
    monkeypatch.setenv("GOOGLE_API_KEY", "fallback-key")
    route = respx.post("https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent").mock(
        return_value=httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}))
    assert math_eval_common.judge_text("p", "gemini", None, "GEMINI_API_KEY", gemini=True) == "ok"
    assert route.calls[0].request.headers["x-goog-api-key"] == "fallback-key"
    monkeypatch.delenv("CUSTOM_JUDGE_KEY", raising=False)
    with pytest.raises(ValueError, match="CUSTOM_JUDGE_KEY"):
        math_eval_common.judge_text("p", "gemini", None, "CUSTOM_JUDGE_KEY", gemini=True)


@pytest.mark.parametrize("name", ["imo-answerbench", "imo-proofbench"])
@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_judge_cli_default_configuration(name, provider, monkeypatch, capsys):
    import sys

    mod = module(name)
    captured = []
    def fake_score(roots, args):
        captured.append(args)
        return {} if name == "imo-answerbench" else []
    monkeypatch.setattr(mod, "score", fake_score)
    argv = ["score.py", "run"]
    if name == "imo-answerbench":
        argv.append("--judge")
    if provider == "gemini":
        argv += ["--provider", "gemini"]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    capsys.readouterr()
    args = captured[0]
    assert args.provider == provider
    assert args.judge_model == ("moonshotai/Kimi-K3.1-Pro" if provider == "openai"
                                else "gemini-3.1-pro-preview")
    assert args.judge_base_url == ("https://router.huggingface.co/v1" if provider == "openai" else None)
    assert args.judge_api_key_env == ("HF_INFERENCE_TOKEN" if provider == "openai" else "GEMINI_API_KEY")


@respx.mock
def test_judge_urllib_client_errors_do_not_retry(monkeypatch):
    module("imo-answerbench")
    import math_eval_common

    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    route = respx.post("https://judge/v1/chat/completions").mock(
        side_effect=HTTPError("https://judge/v1", 403, "forbidden", {}, None))
    delays = []
    with pytest.raises(HTTPError):
        math_eval_common.judge_text("p", "m", "https://judge/v1", "TEST_JUDGE_KEY",
                                   sleep=delays.append)
    assert route.call_count == 1 and delays == []


@respx.mock
def test_gemini_key_takes_precedence_over_fallback(monkeypatch):
    module("imo-proofbench")
    import math_eval_common

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    route = respx.post("https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent").mock(
        return_value=httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}))
    assert math_eval_common.judge_text("p", "gemini", None, "GEMINI_API_KEY", gemini=True) == "ok"
    assert route.calls[0].request.headers["x-goog-api-key"] == "gemini-key"


@respx.mock
def test_judge_retries_malformed_json_then_succeeds(monkeypatch):
    module("imo-answerbench")
    import math_eval_common

    monkeypatch.setenv("TEST_JUDGE_KEY", "test")
    route = respx.post("https://judge/v1/chat/completions").mock(side_effect=[
        httpx.Response(200, text="<html>proxy error</html>"),
        httpx.Response(200, text='{"choices": ['),
        openai_reply(r"\boxed{Correct}"),
    ])
    delays = []
    assert math_eval_common.judge_text("p", "m", "https://judge/v1", "TEST_JUDGE_KEY",
                                      sleep=delays.append) == r"\boxed{Correct}"
    assert route.call_count == 3 and delays == [1, 2]


@pytest.mark.parametrize("name", ["imo-answerbench", "imo-proofbench"])
@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_judge_cli_preserves_explicit_model(name, provider, monkeypatch, capsys):
    import sys

    mod = module(name)
    captured = []
    def fake_score(roots, args):
        captured.append(args)
        return {} if name == "imo-answerbench" else []
    monkeypatch.setattr(mod, "score", fake_score)
    monkeypatch.setattr(sys, "argv", ["score.py", "run", "--provider", provider,
                                      "--judge-model", "custom-model"])
    mod.main()
    capsys.readouterr()
    assert captured[0].judge_model == "custom-model"
