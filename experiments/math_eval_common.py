"""Utilities shared by the math experiment scoring CLIs."""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from math import comb
from pathlib import Path
from urllib.error import HTTPError, URLError

import httpx


def load_runs(roots):
    """Read independent result trees, keeping the latest completed trial per agent."""
    runs = []
    for root in roots:
        root = Path(root)
        grouped = {}
        for result in sorted(root.rglob("result.json")):
            trial = result.parent
            if trial.parent.name != "trials":
                continue
            data = json.loads(result.read_text())
            task = data.get("task_name")
            if not isinstance(task, str) or not task:
                raise ValueError(f"Missing task_name in {result}")
            agent = trial.parent.parent.name
            swarm = trial.parent.parent.parent.relative_to(root).as_posix()
            task_id = task.rsplit("/", 1)[-1]
            if not task_id or task_id in {".", ".."}:
                raise ValueError(f"Invalid task_name in {result}: {task!r}")
            key = (swarm, task, agent)
            finished = data.get("finished_at")
            if finished is not None:
                try:
                    timestamp = datetime.fromisoformat(finished)
                    if timestamp.utcoffset() is None:
                        raise ValueError("timestamp has no timezone")
                    finished = timestamp.timestamp()
                except (ValueError, AttributeError, TypeError, OverflowError) as exc:
                    raise ValueError(f"Invalid finished_at in {result}") from exc
            match = re.search(r"-t(\d+)$", trial.name)
            order = (finished, int(match.group(1)) if match else 0, trial.name)
            previous = grouped.get(key)
            if previous is not None and (finished is None or previous[0][0] is None):
                raise ValueError(f"Cannot order multiple trials without finished_at for {agent}")
            if previous is None or order > previous[0]:
                grouped[key] = (order, trial, data, task_id)
        if not grouped:
            raise ValueError(f"No Legion trial results found under {root}")
        teams = {}
        for (swarm, task, agent), (_, trial, data, task_id) in grouped.items():
            verifier = trial / "verifier"
            if not verifier.is_dir():
                candidates = sorted(trial.glob("steps/*/verifier"))
                if len(candidates) != 1:
                    raise ValueError(f"Expected one verifier directory in {trial}")
                verifier = candidates[0]
            teams.setdefault((swarm, task, task_id), []).append({
                "agent": agent, "trial": str(trial), "verifier": verifier, "result": data,
            })
        runs.extend({"run": str(root), "swarm": swarm, "task": task, "task_id": task_id, "agents": agents}
                    for (swarm, task, task_id), agents in sorted(teams.items()))
    return runs


def read_text(path):
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def offline_reward(agent):
    result = agent["result"].get("verifier_result") or {}
    if not result and agent["verifier"].parent.parent.name == "steps":
        step_name = agent["verifier"].parent.name
        matches = [s for s in agent["result"].get("step_results", []) or []
                   if s.get("step_name") == step_name]
        if len(matches) == 1:
            result = matches[0].get("verifier_result") or {}
    reward = (result.get("rewards") or {}).get("reward")
    if reward not in (0, 1):
        raise ValueError(f"Missing binary verifier reward in {agent['trial']}")
    return int(reward)


def majority(values):
    """Unique plurality; absent submissions abstain, ties return None."""
    counts = Counter(v for v in values if v)
    if not counts:
        return None
    ranked = counts.most_common()
    return ranked[0][0] if len(ranked) == 1 or ranked[0][1] > ranked[1][1] else None


def summary(values):
    values = list(values)
    return {"count": len(values), "mean": sum(values) / len(values) if values else None}


DEFAULT_JUDGE_MODEL = "moonshotai/Kimi-K3.1-Pro"
DEFAULT_GEMINI_JUDGE_MODEL = "gemini-3.1-pro-preview"
DEFAULT_JUDGE_BASE_URL = "https://router.huggingface.co/v1"
JUDGE_RETRY_DELAYS = (1, 2, 4)


class EmptyJudgeResponse(ValueError):
    """The judge returned no final text."""


def judge_text(prompt, model, base_url, api_key_env, gemini=False, *, sleep=None):
    """Request text with bounded retries for transient failures."""
    key = os.environ.get(api_key_env)
    if not key and api_key_env == "GEMINI_API_KEY":
        key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(f"Set {api_key_env} for the judge")
    if sleep is None:
        sleep = time.sleep
    if gemini:
        base = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url = f"{base}/models/{model}:generateContent"
        headers = {"x-goog-api-key": key}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
    else:
        base = (base_url or DEFAULT_JUDGE_BASE_URL).rstrip("/")
        url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {key}"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(len(JUDGE_RETRY_DELAYS) + 1):
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            if not response.content.strip():
                raise EmptyJudgeResponse("Judge returned an empty response")
            result = response.json()
            if gemini:
                candidates = result.get("candidates") or []
                parts = (candidates[0].get("content") or {}).get("parts", []) if candidates else []
                content = "".join(part.get("text", "") for part in parts if not part.get("thought"))
            else:
                choices = result.get("choices") or []
                content = (choices[0].get("message") or {}).get("content") if choices else None
            if content is None or content == "" or (isinstance(content, str) and not content.strip()):
                raise EmptyJudgeResponse("Judge returned empty text")
            if not isinstance(content, str):
                raise TypeError("Judge must return text")
            return content
        except (httpx.HTTPStatusError, HTTPError) as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else exc.code
            if status not in (408, 429) and not 500 <= status <= 599:
                raise
            if attempt == len(JUDGE_RETRY_DELAYS):
                raise
        except (httpx.RequestError, URLError, TimeoutError, EmptyJudgeResponse, json.JSONDecodeError):
            if attempt == len(JUDGE_RETRY_DELAYS):
                raise
        delay = JUDGE_RETRY_DELAYS[attempt]
        if delay:
            sleep(delay)
    raise AssertionError("unreachable retry loop")


def remove_self_evaluation(text):
    """Apply QED-Nano's evaluation-heading cleanup to a candidate solution."""
    pattern = r"(?ms)^#+\s*[*\s]*(?:(?:Self|Final)\s*[*\s]*){0,2}Evaluation\b.*$"
    return re.sub(pattern, "", text).strip()


def _boxed_matches(text):
    """Find non-overlapping boxed/fbox values using balanced braces."""
    matches = []
    pos = 0
    while match := re.search(r"(?:boxed|fbox)\{", text[pos:]):
        start = pos + match.start()
        inner = pos + match.end()
        depth = 1
        for i in range(inner, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                matches.append((start, i + 1, text[inner:i]))
                pos = i + 1
                break
        else:
            pos = inner
    return matches


def _remove_inner_boxed(text):
    matches = _boxed_matches(text)
    for start, end, value in reversed(matches):
        if start > 0 and text[start - 1] == "\\":
            text = text[:start - 1] + _remove_inner_boxed(value) + text[end:]
    return text


def find_last_boxed_content(text):
    """QED-Nano extraction: combine boxes on the last line, unwrapping inner boxes."""
    matches = _boxed_matches(text)
    if not matches:
        return None
    if len(matches) > 1:
        for line in reversed(text.split("\n")):
            same_line = _boxed_matches(line)
            if same_line:
                return ",".join(_remove_inner_boxed(value) for _, _, value in same_line)
    return _remove_inner_boxed(matches[-1][2])


def parse_answer_verdict(text):
    """Accept a complete final verdict, never a box left inside the judge's analysis."""
    in_thinking = False
    verdict_start = 0
    for tag in re.finditer(r"</?thinking>", text, re.IGNORECASE):
        closing = tag.group().startswith("</")
        if closing != in_thinking:
            raise ValueError("Judge returned malformed thinking tags")
        in_thinking = not closing
        if closing:
            verdict_start = tag.end()
    if in_thinking:
        raise ValueError("Judge returned unclosed thinking tags")
    verdict = text[verdict_start:]
    # Check every explicit box command, including nested boxes. An unfinished
    # trailing box must not cause the previous valid box to become the verdict.
    for command in re.finditer(r"\\(?:boxed|fbox)\b", verdict):
        opening = command.end()
        if opening >= len(verdict) or verdict[opening] != "{":
            raise ValueError("Judge returned an unfinished boxed verdict")
        depth = 1
        for char in verdict[opening + 1:]:
            depth += (char == "{") - (char == "}")
            if depth == 0:
                break
        else:
            raise ValueError("Judge returned an unfinished boxed verdict")
    matches = _boxed_matches(verdict)
    boxed = _remove_inner_boxed(matches[-1][2]).strip().lower() if matches else None
    if boxed not in {"correct", "incorrect"}:
        raise ValueError("Judge must return a final boxed Correct or Incorrect verdict")
    return boxed == "correct"


def parse_proof_points(text):
    # The user-requested prompt contract specifies the first tag. QED-Nano's current
    # extract_html implementation instead chooses the last tag when several are present.
    match = re.search(r"<points(?:\s[^>]*)?>(.*?)</points>", text, re.DOTALL)
    points = re.fullmatch(r"\s*([0167])\s+out\s+of\s+7\s*", match.group(1), re.IGNORECASE) if match else None
    if points is None:
        raise ValueError("Judge must return a first <points>N out of 7</points> with N in 0, 1, 6, 7")
    return int(points.group(1))


def output_tokens(agent):
    """Harbor TrialResult.agent_result is an AgentContext, with n_output_tokens."""
    usage = agent["result"].get("agent_result") or {}
    count = usage.get("n_output_tokens")
    if count is None:
        return None
    if type(count) is not int or count < 0:
        raise ValueError(f"Invalid n_output_tokens in {agent['trial']}")
    return count


def validate_k(k):
    if type(k) is not int or k < 2:
        raise ValueError("k must be an integer >= 2")


def pass_estimate(n, correct, k):
    """Unbiased pass@k over all k-sized subsets of n independent samples."""
    validate_k(k)
    if n < k:
        return None
    return 1 - (comb(n - correct, k) / comb(n, k) if n - correct >= k else 0)


def compare_binary(teams, k=4):
    """Compare N=1 independent runs and N=k teams, grouped by dataset task ID."""
    validate_k(k)
    tasks = {}
    for team in teams:
        task = tasks.setdefault(team["task_id"], {"solo": [], "teams": [],
                                                   "level": team.get("level")})
        if task["level"] != team.get("level"):
            raise ValueError(f"Inconsistent difficulty for {team['task_id']}")
        if team["n"] == 1:
            task["solo"].append({**team["agents"][0], "run": team["run"], "swarm": team["swarm"]})
        elif team["n"] == k:
            task["teams"].append(team)
    results = []
    for task_id, data in sorted(tasks.items()):
        solo = sorted(data["solo"], key=lambda a: (a["run"], a["swarm"], a["agent"]))
        chosen = solo[:k] if len(solo) >= k else []
        winner = majority([a["answer"] for a in chosen])
        votes = [a["correct"] for a in chosen if a["answer"] == winner] if winner else []
        if len(set(votes)) > 1:
            raise ValueError("Identical answers received inconsistent grades")
        tokens = [a["output_tokens"] for a in solo if a["output_tokens"] is not None]
        teams = data["teams"]
        totals = [sum(a["output_tokens"] for a in t["agents"])
                  for t in teams if all(a["output_tokens"] is not None for a in t["agents"])]
        results.append({"task_id": task_id, "level": data["level"],
            "solo": {"n": len(solo), "pass_at_1": summary(a["correct"] for a in solo)["mean"],
                     "pass_at_k": pass_estimate(len(solo), sum(a["correct"] for a in solo), k),
                     "indep_maj_at_k": (votes[0] if votes else 0) if chosen else None,
                     "majority_runs": [a["run"] for a in chosen],
                     "mean_output_tokens": summary(tokens)},
            "team": {"n_runs": len(teams), "maj_at_k": summary(t["maj_at_n"] for t in teams),
                     "pass_at_k": summary(max(a["correct"] for a in t["agents"]) for t in teams),
                     "mean_output_tokens": summary(totals)}})
    def aggregate(group):
        def average(values):
            return summary(v for v in values if v is not None)
        return {"solo_pass_at_1": average(t["solo"]["pass_at_1"] for t in group),
                "indep_pass_at_k": average(t["solo"]["pass_at_k"] for t in group),
                "indep_maj_at_k": average(t["solo"]["indep_maj_at_k"] for t in group),
                "team_maj_at_k": average(t["team"]["maj_at_k"]["mean"] for t in group),
                "team_pass_at_k": average(t["team"]["pass_at_k"]["mean"] for t in group),
                "solo_mean_output_tokens": average(t["solo"]["mean_output_tokens"]["mean"] for t in group),
                "team_mean_output_tokens": average(t["team"]["mean_output_tokens"]["mean"] for t in group)}
    return {"k": k, "tasks": results, "overall": aggregate(results),
            "levels": {str(level): aggregate([t for t in results if t["level"] == level])
                       for level in range(1, 7)} if any(t["level"] is not None for t in results) else {}}


def compare_proof(records, k=4):
    """Aggregate proof grades, using >=6/7 as the binary pass threshold."""
    validate_k(k)
    tasks = {}
    for row in records:
        data = tasks.setdefault(row["task_id"], {"solo": [], "teams": []})
        if row["type"] == "team" and row["n"] == k:
            data["teams"].append(row)
        elif row["type"] == "team" and row["n"] == 1:
            data["solo"].append(row)
    out = []
    for task, data in sorted(tasks.items()):
        solo, teams = data["solo"], data["teams"]
        values = [r["mean"] for r in solo]
        tokens = [r["output_tokens"] for r in solo if r["output_tokens"] is not None]
        team_tokens = [r["output_tokens"] for r in teams if r["output_tokens"] is not None]
        expected_max = None
        if len(values) >= k:
            denominator = comb(len(values), k)
            expected_max = 100 / 7 * sum(value * (comb(i, k - 1) / denominator)
                                         for i, value in enumerate(sorted(values)) if i >= k - 1)
        out.append({"type": "comparison", "task_id": task, "k": k,
                    "solo": {"n": len(values),
                             "pass_at_1_percent_of_7": summary(100 * v / 7 for v in values),
                             "pass_at_k_ge_6": pass_estimate(len(values), sum(v >= 6 for v in values), k),
                             "expected_max_at_k_percent_of_7": expected_max,
                             "max_observed_percent_of_7": max(values) * 100 / 7 if values else None,
                             "mean_output_tokens": summary(tokens)},
                    "team": {"n_runs": len(teams),
                             "mean_percent_of_7": summary(r["mean_percent_of_7"] for r in teams),
                             "max_percent_of_7": summary(r["max_percent_of_7"] for r in teams),
                             "consensus_percent_of_7": summary(r["consensus_percent_of_7"] for r in teams
                                                               if r["consensus_percent_of_7"] is not None),
                             "pass_at_k_ge_6": summary(int(r["max"] >= 6) for r in teams),
                             "mean_output_tokens": summary(team_tokens)}})
    def average(values):
        return summary(v for v in values if v is not None)
    out.append({"type": "comparison_summary", "k": k,
        "solo_pass_at_1_percent_of_7": average(r["solo"]["pass_at_1_percent_of_7"]["mean"] for r in out),
        "solo_pass_at_k_ge_6": average(r["solo"]["pass_at_k_ge_6"] for r in out),
        "solo_expected_max_at_k_percent_of_7": average(r["solo"]["expected_max_at_k_percent_of_7"] for r in out),
        "solo_max_observed_percent_of_7": average(r["solo"]["max_observed_percent_of_7"] for r in out),
        "team_mean_percent_of_7": average(r["team"]["mean_percent_of_7"]["mean"] for r in out),
        "team_max_percent_of_7": average(r["team"]["max_percent_of_7"]["mean"] for r in out),
        "team_consensus_percent_of_7": average(r["team"]["consensus_percent_of_7"]["mean"] for r in out),
        "team_pass_at_k_ge_6": average(r["team"]["pass_at_k_ge_6"]["mean"] for r in out),
        "solo_mean_output_tokens": average(r["solo"]["mean_output_tokens"]["mean"] for r in out),
        "team_mean_output_tokens": average(r["team"]["mean_output_tokens"]["mean"] for r in out)})
    return out
