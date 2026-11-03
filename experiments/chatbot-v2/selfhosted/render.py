"""Render the self-hosted / max-utilization variant of chatbot-v2 into an experiment folder.

    python experiments/chatbot-v2/selfhosted/render.py --into <dir>

<dir> must hold a copy of experiments/chatbot-v2 (its tools/, protocols/chatbot.md,
prompts/interviewer.md, prompts/final.md, world.yaml and job/ are kept); the job's bootstrap
renders into its own copy with CHATBOT_VARIANT=selfhosted. Written: spec.yaml and
phase2/spec.yaml (N teams, providers on the operator's OpenAI-compatible endpoint at price 0,
no $ caps, `every: turn_end`), the per-team prompts, the kickoff, the team protocol, the
phase-2 prompts, and world.yaml with scale.yaml's `world` knobs merged in. Knobs: scale.yaml
(one place); endpoint and model ids: the environment (scale.yaml lists the variables)."""

from __future__ import annotations

import argparse
import math
import os
import string
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
WORLD = HERE.parents[2] / "worlds" / "chatbot2"
PLACEHOLDER_URL = "https://your-endpoint.example.invalid/v1"
TOOLS = ["spawn_interviewer", *(f"world:{t}" for t in (
    "read_file", "grep_file", "write_file", "copy_file", "edit_file", "insert_after",
    "list_files", "run_bot", "promote_best", "best_status", "qualify", "add_probe", "probes",
    "book_interview", "verdicts"))]


def env(name: str, default: str | None = None) -> str:
    return os.environ.get(name) or default or name


def plan(cfg: dict) -> dict:
    """Teams, agents and models from the knobs and the environment."""
    two = bool(os.environ.get("MODEL_GLM"))
    per = {k: v for k, v in cfg["per_team"].items() if v} if two else {
        "ds": sum(cfg["per_team"].values())}
    letters = string.ascii_lowercase[:cfg["teams"]]
    if len(letters) != cfg["teams"] or cfg["teams"] < 2:
        sys.exit("teams must be 2 to 26")
    agents = {t: [f"{m}-{t}-{i:04d}" for m, n in per.items() for i in range(n)]
              for t in letters}
    return {"two": two, "per": per, "letters": letters, "agents": agents,
            "n": sum(len(a) for a in agents.values()), "per_team": sum(per.values())}


def fill(text: str, values: dict) -> str:
    for k, v in values.items():
        text = text.replace(f"<<{k}>>", str(v))
    assert "<<" not in text, text[text.index("<<"):][:40]
    return text


def spec(cfg: dict, p: dict, into: Path, phase: int) -> dict:
    hours, base = cfg["hours"], env("SELFHOSTED_BASE_URL", PLACEHOLDER_URL)
    up = "../" if phase == 2 else ""
    ids = {"ds": env("MODEL_DS"), "glm": env("MODEL_GLM", env("MODEL_DS"))}
    providers, models = {}, {}
    for m in ("ds", "glm") if p["two"] else ("ds",):
        providers[m] = {"type": "openai_compat", "base_url": env(f"BASE_URL_{m.upper()}", base),
                        "api_key_env": cfg["api_key_env"], "timeout_s": cfg["timeout_s"],
                        "max_retries": 5, "self_hosted": True, "pricing": {}}
        if cfg.get("max_working"):
            providers[m]["max_working"] = cfg["max_working"]
        models[m] = {"model": f"{m}:{ids[m]}", "extra": cfg["models"][m]}
    iv_mid = "glm" if p["two"] else "ds"
    for level, prov, mid in (("strong", "ds", env("MODEL_INTERVIEWER_STRONG", ids["ds"])),
                             ("mid", iv_mid, env("MODEL_INTERVIEWER_MID", ids[iv_mid]))):
        models[f"interviewer-{level}"] = {"model": f"{prov}:{mid}",
                                          "extra": cfg["models"]["interviewer"]}
    for name, prov in providers.items():  # price 0 for every id served there: no $ cap fires
        prov["pricing"] = {mid: [0, 0, 0] for pk, mid in (
            m["model"].split(":", 1) for m in models.values()) if pk == name}
    types = {}
    for t in p["letters"]:
        for m, n in p["per"].items():
            loop = {
                "prompt_file": f"round-{t}.md" if phase == 2 else f"prompts/round-{t}.md",
                "every": "turn_end", "on": ["comms.mention", "comms.thread"],
                "wake": dict(cfg["wake"]),
                "while": {"until": f"{round(hours * 60)}m", "max_rounds": cfg["max_rounds"]},
                "final": {"prompt_file": "final.md" if phase == 2 else "prompts/final.md"}}
            types[f"{m}-{t}"] = {
                "harness": "llm", "model": m, "replicas": n,
                "agents_per_worker": cfg["agents_per_worker"],
                "system_prompt_file": "system.md" if phase == 2 else f"prompts/system-{t}.md",
                "comms": True, "tools": list(TOOLS), "bash": False, "tool_choice": "required",
                "max_turns": cfg["max_turns"], "tool_output_chars": 32000,
                "context_tokens": 64000, "loop": loop}
    live = 2 * p["n"]  # spawn.<type>.max live interviewers per agent
    for level in ("mid", "strong"):
        types[f"interviewer-{level}"] = {
            "harness": "llm", "model": f"interviewer-{level}", "replicas": 0,
            "agents_per_worker": cfg["agents_per_worker"],
            "workers": {"initial": 0, "max_extra": math.ceil(live / cfg["agents_per_worker"])},
            "ephemeral": True, "system_prompt_file": f"{up}prompts/interviewer.md",
            "tools": ["world:chat", "world:submit_verdict", "comms:post"], "bash": False,
            "tool_choice": "required", "max_turns": cfg["max_turns"], "tool_output_chars": 4000}
    world_dir = os.path.relpath(WORLD, into / ("phase2" if phase == 2 else "."))
    return {
        "swarm": "chatbot-v2", "compute": "local", "tools_dir": f"{up}tools",
        "providers": providers, "models": models,
        "world": {"dir": world_dir, "command": ["{python}", "chatbot2.py", "serve"],
                  "token_env": "CHATBOT_WORLD_TOKEN"},
        "comms": {"kind": "msgboard", "url": "https://msgboard.example.invalid",
                  "invite_env": "CHATBOT_BOARD_INVITE", "channel": "your-chatbot-channel",
                  "kickoff_file": f"{up}prompts/kickoff.md",
                  "rules": {"min_post_gap": cfg["min_post_gap"], "max_len": 800,
                            "threads": "per_topic", "no_group_mentions": True},
                  "protocols": ["../protocols/chatbot.md", "protocol.md"] if phase == 2 else [
                      "protocols/team.md", "protocols/chatbot.md"]},
        "types": types,
        "spawn": {name: {"types": ["interviewer-mid", "interviewer-strong"], "max": 2, "depth": 1}
                  for name, t in types.items() if t.get("loop")},
        "limits": {"budget_usd": None, "spawned_total": 4 * p["n"] * hours}}


HEAD = """# chatbot-v2, self-hosted / max utilization, phase {phase}: GENERATED by
# selfhosted/render.py from selfhosted/scale.yaml ({teams} teams, {n} agents, {per} per team).
# Do not edit: change scale.yaml or the environment and render again. Price 0 on every model
# (self-hosted), no $ caps: the limits are max_turns per prompt, while.max_rounds per agent
# and while.until for the loop.
"""


def render(into: Path, cfg: dict) -> dict:
    p = plan(cfg)
    letters, n, hours = p["letters"], p["n"], cfg["hours"]
    mix = (", ".join(f"{len(letters) * k} on {m.upper()}" for m, k in p["per"].items())
           if p["two"] else "all on one model")
    base = {"n_agents": n, "n_teams": len(letters), "per_team": p["per_team"], "mix": mix,
            "hours": f"{hours:g}", "phase1": f"{hours / 2:g}", "phase2": f"{hours / 2:g}",
            "threads": ", ".join(f"[team-{t}]" for t in letters),
            "all_agents": ", ".join(a for t in letters for a in p["agents"][t]),
            "first_a": p["agents"][letters[0]][0],
            "finalists": ", ".join(p["agents"][t][0] for t in letters),
            "team_lines": "\n".join(f"- Team {t.upper()} ({', '.join(p['agents'][t])}). Thread: "
                                    f"[team-{t}]." for t in letters)}
    tpl = {f.stem: f.read_text() for f in (HERE / "templates").glob("*.md")}
    out = {"prompts/kickoff.md": fill(tpl["kickoff"], base),
           "protocols/team.md": fill(tpl["team"], base),
           "phase2/system.md": fill(tpl["p2-system"], base),
           "phase2/protocol.md": fill(tpl["p2-protocol"], base),
           "phase2/final.md": fill(tpl["p2-final"], base)}
    for i, t in enumerate(letters):
        others = [x.upper() for x in letters if x != t]
        nxt = letters[(i + 1) % len(letters)]
        v = {**base, "team": t, "TEAM": t.upper(), "members": ", ".join(p["agents"][t]),
             "first": p["agents"][t][0], "next_team": f"team {nxt}",
             "others": f"Team {others[0]}" if len(others) == 1 else
             "Teams " + ", ".join(others[:-1]) + f" and {others[-1]}"}
        out[f"prompts/system-{t}.md"] = fill(tpl["system"], v)
        out[f"prompts/round-{t}.md"] = fill(tpl["round"], v)
        out[f"phase2/round-{t}.md"] = fill(tpl["p2-round"], v)
    for rel, text in out.items():
        (into / rel).parent.mkdir(parents=True, exist_ok=True)
        (into / rel).write_text(text)
    for phase, rel in ((1, "spec.yaml"), (2, "phase2/spec.yaml")):
        head = HEAD.format(phase=phase, teams=len(letters), n=n, per=p["per_team"])
        (into / rel).write_text(head + yaml.safe_dump(spec(cfg, p, into, phase), sort_keys=False))
    w = yaml.safe_load((into / "world.yaml").read_text()) or {}
    for k, v in (cfg.get("world") or {}).items():
        w[k] = {**(w.get(k) or {}), **v}
    (into / "world.yaml").write_text(yaml.safe_dump(w, sort_keys=False))
    workers = sum(math.ceil(k / cfg["agents_per_worker"]) for k in p["per"].values()) * len(
        letters)
    return {"teams": len(letters), "agents": n, "per_team": p["per"], "workers": workers,
            "models": sorted({m["model"] for m in spec(cfg, p, into, 1)["models"].values()})}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--into", required=True, type=Path, help="a copy of experiments/chatbot-v2")
    ap.add_argument("--config", type=Path, default=HERE / "scale.yaml")
    ap.add_argument("--teams", type=int, help="override scale.yaml teams")
    ap.add_argument("--per-team", help="override scale.yaml per_team, e.g. ds=2,glm=2")
    ap.add_argument("--hours", type=float, help="override scale.yaml hours")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    if a.teams:
        cfg["teams"] = a.teams
    if a.per_team:
        cfg["per_team"] = {k: int(v) for k, v in (x.split("=") for x in a.per_team.split(","))}
    if a.hours:
        cfg["hours"] = a.hours
    print(yaml.safe_dump(render(a.into.resolve(), cfg), sort_keys=False).strip())


if __name__ == "__main__":
    main()
