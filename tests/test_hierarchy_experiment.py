"""The hierarchy experiment: coordinators spawn up to four workers; workers cannot spawn."""
from __future__ import annotations

from pathlib import Path

from flock.cli import load_spec
from flock.models import ALL_COMMS_TOOLS, comms_tools

from .test_run4_spawner import as_agent

ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXP = ROOT / "hierarchy"
REPOS = {"hierarchy": "HuggingFaceSwarms/chatbot-v5", "voting": "HuggingFaceSwarms/chatbot-v6"}  # one dataset each
SHARED = "You are one of several agents improving " + REPOS["hierarchy"]


def check_one_spec(exp: Path, monkeypatch, ramped: str, schedule: list[tuple[int, int]]):
    """One spec.yaml per run: Slurm by default (world.url), local with COMPUTE_KIND, the smoke
    with LOOP_UNTIL, the ramp (was waves.yaml) and its load guard inline; no overlay files."""
    for var in ("CHATBOT_RUN", "COMPUTE_KIND", "LOOP_UNTIL", "LOOP_MAX_ROUNDS"):
        monkeypatch.delenv(var, raising=False)
    assert sorted(p.name for p in exp.iterdir() if p.suffix in (".yaml", ".py")) == ["spec.yaml"]
    spec = load_spec(str(exp / "spec.yaml"))
    assert spec.swarm == exp.name and spec.compute.kind == "slurm" and spec.bucket
    assert spec.world.url == "http://${WORLD_HOST}:7860" and spec.providers["vllm"].metrics
    assert [n for n, t in spec.types.items() if t.replicas_ramp] == [ramped]
    r = spec.types[ramped].replicas_ramp
    assert [(t, n) for t, n, _ in r.schedule()] == schedule and r.schedule()[0][2] == 10
    assert r.hold_if.model_dump() == {"waiting": 1, "request_s": 15, "above": 100}
    monkeypatch.setenv("COMPUTE_KIND", "local")
    monkeypatch.setenv("LOOP_UNTIL", "30m")
    monkeypatch.setenv("LOOP_MAX_ROUNDS", "100")
    smoke = load_spec(str(exp / "spec.yaml"))
    assert smoke.compute.kind == "local"
    loops = [t.loop.while_ for t in smoke.types.values() if t.loop]
    assert loops and all(w.until == "30m" and w.max_rounds == 100 for w in loops)
    return spec


def check_types(exp: Path):
    spec = load_spec(str(exp / "spec.yaml"))
    run4 = load_spec(str(ROOT / "run4/spec.yaml")).types["agent"]
    assert set(spec.types) == {"coordinator", "worker"}
    skip = {"system_prompt": True, "system_prompt_append_file": True, "workers": True,
            "loop": {"prompt"}, "replicas_ramp": True}  # voting's round prompt adds the vote
    for t in spec.types.values():  # run 4's agent, the same world tools, board and loop
        assert t.replicas == 0 and t.model_dump(exclude=skip) == run4.model_dump(exclude=skip)
    assert set(spec.spawn) == {"coordinator"}  # workers: depth 0, no spawning
    assert spec.spawn["coordinator"].types == {"worker": 4} and spec.spawn["coordinator"].depth == 1
    board = {"comms:" + n for n in ("channels", "create_channel", "join_channel", "search", "react",
                                    "post", "read")}  # the board tools the prompts name
    assert board <= set(ALL_COMMS_TOOLS)
    for t in spec.types.values():
        assert board <= set(comms_tools(t, spec.comms))
    coord, worker = spec.types["coordinator"].system_prompt, spec.types["worker"].system_prompt
    assert "enforce" not in coord.lower() and "enforce" not in worker.lower()
    assert "flock prompt" not in coord + worker  # agents talk on the board only
    assert "flock-agent spawn worker" in coord and "flock-agent spawn" not in worker
    assert "You cannot start other agents" in worker and "$FLOCK_PARENT" in worker
    def team(role):  # the shared rules: the same text for both roles
        return (exp / f"prompts/{role}.md").read_text().split("## Team rules")[1].split("\n", 1)[1].split("\n## ")[0]
    repo = REPOS[exp.name]
    assert f"improving {repo}." in team("coordinator") and team("coordinator") == team("worker")
    system = (exp / "prompts/system.md").read_text()
    assert f"`{repo}` on Hugging Face" in system and f"`https://huggingface.co/datasets/{repo}` into `chatbot-swarm`" in system
    assert "[PROJECT" not in coord + worker
    assert not any(r in coord + worker for r in REPOS.values() if r != repo)  # never the other run's repo
    return spec, coord, worker


def test_types_prompts_and_spawn_policy():
    spec, coord, worker = check_types(EXP)
    assert "vote" not in coord.lower() and "vote" not in worker.lower()  # voting's prompts carry it
    assert "The swarm works in teams" in coord and "# Project: Python Chatbot" in worker
    for text in ("benefits the goal of the task", "join that channel", "up to 4 new workers",
                 "opens and merges the PRs", "After every merge, assess"):
        assert text in coord, text
    for text in (SHARED, "Establish the goal", "Build the shared plan", "`edit_doc`",
                 "Claim and execute", "Reassess", "Verify and hand off"):
        assert text in coord and text in worker, text
    assert spec.limits.alive >= 50 + 4 * 50
    assert spec.types["coordinator"].workers.initial * 12 >= 50
    assert spec.types["worker"].workers.max_extra * 12 >= 200


def test_one_spec_with_the_ramp(monkeypatch):
    check_one_spec(EXP, monkeypatch, "coordinator", [(600 * i, 10 * (i + 1)) for i in range(5)])


async def test_coordinator_spawns_four_workers_and_workers_cannot_spawn(client, app, store, tick):
    from flock.controller import spawn as spawning

    spec = load_spec(str(EXP / "spec.yaml"))
    spec.types["coordinator"].replicas = 1
    data = spec.model_dump(mode="json")  # no world or board in the in-process controller
    data.update(world=None, comms=None)
    for t in data["types"].values():
        t.update(tools=[], comms=False, replicas_ramp=None)
    spec = type(spec).model_validate(data)
    assert (await client.put(f"/swarms/{spec.swarm}", json=data)).status_code == 200
    tick()
    [coord] = [a["name"] for a in (await client.get(f"/swarms/{spec.swarm}/agents")).json()["agents"]]
    async with as_agent(app, spawning.agent_token(store, coord, spec.swarm, 0)) as c:
        kids = []
        for i in range(4):
            r = await c.post(f"/agents/{coord}/spawn", json={"type": "worker", "key": f"k{i}"})
            assert r.status_code == 201, r.text
            kids.append(r.json()["name"])
        r = await c.post(f"/agents/{coord}/spawn", json={"type": "worker", "key": "k4"})
        assert r.status_code == 409 and r.json()["reason"] == "max_children"
    async with as_agent(app, spawning.agent_token(store, kids[0], spec.swarm, 0)) as c:
        r = await c.post(f"/agents/{kids[0]}/spawn", json={"type": "worker"})
        assert r.status_code == 409 and r.json()["reason"] in {"depth", "type_not_allowed"}
