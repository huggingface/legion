"""Utilities shared by the math experiment scoring CLIs."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import httpx


def load_runs(roots):
    """Read independent result trees, keeping the latest numbered trial per agent."""
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
            key = (swarm, task, agent)
            match = re.search(r"-t(\d+)$", trial.name)
            order = (int(match.group(1)) if match else 0, data.get("finished_at") or "", trial.name)
            previous = grouped.get(key)
            if previous is None or order > previous[0]:
                grouped[key] = (order, trial, data)
        if not grouped:
            raise ValueError(f"No Legion trial results found under {root}")
        teams = {}
        for (swarm, task, agent), (_, trial, data) in grouped.items():
            verifier = trial / "verifier"
            if not verifier.is_dir():
                candidates = sorted(trial.glob("steps/*/verifier"))
                if len(candidates) != 1:
                    raise ValueError(f"Expected one verifier directory in {trial}")
                verifier = candidates[0]
            teams.setdefault((swarm, task), []).append({
                "agent": agent, "trial": str(trial), "verifier": verifier, "result": data,
            })
        runs.extend({"run": str(root), "swarm": swarm, "task": task, "agents": agents}
                    for (swarm, task), agents in sorted(teams.items()))
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


def judge_json(prompt, model, base_url, api_key_env, gemini=False):
    key = os.environ.get(api_key_env)
    if not key:
        raise ValueError(f"Set {api_key_env} for the judge")
    if gemini:
        base = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url = f"{base}/models/{model}:generateContent"
        response = httpx.post(url, headers={"x-goog-api-key": key}, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }, timeout=120)
        response.raise_for_status()
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        base = (base_url or "https://router.huggingface.co/v1").rstrip("/")
        response = httpx.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                              json={"model": model, "messages": [{"role": "user", "content": prompt}],
                                    "response_format": {"type": "json_object"}}, timeout=120)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    value = json.loads(content)
    if not isinstance(value, dict):
        raise TypeError("Judge must return a JSON object")
    return value
