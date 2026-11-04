"""The levels experiment: leads spawn up to five managers, managers up to five workers;
workers cannot spawn."""
from __future__ import annotations

from pathlib import Path

from flock.cli import load_spec

from .test_hierarchy_experiment import module
from .test_run4_spawner import as_agent

ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXP = ROOT / "levels"
REPO = "HuggingFaceSwarms/SwarmChat-v10"
SUPERVISION = ("You are responsible for the work of the agents you started: give each a clear scope, "
               "check their progress each round, review or get reviews for their PRs, unblock them, "
               "and report decisions, blockers and verified milestones to ")


def rules(role):
    return (EXP / f"prompts/{role}.md").read_text().split("## Team rules")[1].split("\n", 1)[1]


def test_types_prompts_and_spawn_policy(monkeypatch):
    monkeypatch.delenv("CHATBOT_BOARD_URL", raising=False)
    spec = load_spec(str(EXP / "spec.yaml"))
    hierarchy = load_spec(str(ROOT / "hierarchy/spec.yaml")).types["worker"]
    assert set(spec.types) == {"lead", "manager", "worker"}
    skip = {"system_prompt": True, "system_prompt_append_file": True, "workers": True}
    for t in spec.types.values():  # hierarchy's agent: same model, tools, loop
        assert t.replicas == 0 and t.model_dump(exclude=skip) == hierarchy.model_dump(exclude=skip)
    assert set(spec.spawn) == {"lead", "manager"}  # workers: depth 0, no spawning
    assert spec.spawn["lead"].types == {"manager": 5} and spec.spawn["lead"].depth == 2
    assert spec.spawn["manager"].types == {"worker": 5} and spec.spawn["manager"].depth == 1
    slots = {n: (t.workers.initial + t.workers.max_extra) * t.agents_per_worker
             for n, t in spec.types.items()}
    assert slots["lead"] >= 20 and slots["manager"] >= 100 and slots["worker"] >= 500
    assert spec.limits.alive >= 620 and spec.limits.spawned_total > 600
    assert spec.comms.rules.max_channels == 100
    assert spec.comms.url == "https://huggingfaceswarms-chatbot-v10-msgboard.hf.space/boards/chatbot-v10"

    system = (EXP / "prompts/system.md").read_text()
    assert f"`{REPO}` on Hugging Face" in system and f"datasets/{REPO}` into `chatbot-swarm`" in system
    assert "three levels" in system and "Whoever starts agents supervises them" in system
    assert "Not every agent needs to write features" in system
    assert "understanding engine" in system and "general mechanisms" in system
    lead, manager, worker = (spec.types[n].system_prompt for n in ("lead", "manager", "worker"))
    everything = lead + manager + worker
    for gone in ("## Interviews", "flock spawn interviewer", "verdict", "vote", "voting",
                 "flock prompt", "SwarmChat-v9", "chatbot-v5"):
        assert gone not in everything, gone
    assert rules("lead") == rules("manager") == rules("worker")
    assert f"improving {REPO}." in rules("lead")
    for text in ("Establish the goal", "Build the shared plan", "Claim and execute", "Reassess",
                 "Verify and hand off", "say so in the team channel"):
        assert text in rules("lead"), text
    assert SUPERVISION + "#general. Remove an agent that is done with `flock remove`." in lead
    assert SUPERVISION + "your lead in the team channel." in manager
    assert "between 1 and 5" in lead and "flock spawn manager --prompt" in lead
    assert "between 1 and 5" in manager and "flock spawn worker --prompt" in manager
    assert "flock spawn" not in worker and "You cannot start other agents" in worker
    assert "$FLOCK_PARENT" in manager and "$FLOCK_PARENT" in worker


def test_render_and_waves_read_this_folder():
    data = module(EXP, "render").render()  # jobs: the Slurm overlay
    assert data["swarm"] == "levels" and data["compute"]["kind"] == "slurm"
    assert set(data["types"]) == {"lead", "manager", "worker"}
    smoke = module(EXP, "render").render("smoke", "slurm")
    assert smoke["compute"]["kind"] == "slurm"
    assert all(t["loop"]["while"]["until"] == "30m" for t in smoke["types"].values())
    waves = module(EXP, "waves").load(EXP / "waves.yaml")
    assert waves.type == "lead" and waves.cap == 20 and waves.events == [(0, 10), (600, 20)]


async def test_three_levels_spawn_with_caps(client, app, store, tick):
    from flock.controller import spawn as spawning

    spec = load_spec(str(EXP / "spec.yaml"))
    spec.types["lead"].replicas = 1
    data = spec.model_dump(mode="json")  # no world or board in the in-process controller
    data.update(world=None, comms=None)
    for t in data["types"].values():
        t.update(tools=[], comms=False)
    assert (await client.put(f"/swarms/{spec.swarm}", json=data)).status_code == 200
    tick()
    [lead] = [a["name"] for a in (await client.get(f"/swarms/{spec.swarm}/agents")).json()["agents"]]

    async def spawn_n(parent, type_, n):
        async with as_agent(app, spawning.agent_token(store, parent, spec.swarm, 0)) as c:
            kids = []
            for i in range(n):
                r = await c.post(f"/agents/{parent}/spawn", json={"type": type_, "key": f"k{i}"})
                assert r.status_code == 201, r.text
                kids.append(r.json())
            r = await c.post(f"/agents/{parent}/spawn", json={"type": type_, "key": "extra"})
            assert r.status_code == 409 and r.json()["reason"] == "max_children"
            return kids

    managers = await spawn_n(lead, "manager", 5)
    assert all(m["depth"] == 1 for m in managers)
    async with as_agent(app, spawning.agent_token(store, lead, spec.swarm, 0)) as c:
        r = await c.post(f"/agents/{lead}/spawn", json={"type": "worker"})  # not a lead's child
        assert r.status_code == 409 and r.json()["reason"] == "type_not_allowed"
    workers = await spawn_n(managers[0]["name"], "worker", 5)
    assert all(w["depth"] == 0 for w in workers)
    w = workers[0]["name"]
    async with as_agent(app, spawning.agent_token(store, w, spec.swarm, 0)) as c:
        r = await c.post(f"/agents/{w}/spawn", json={"type": "worker"})
        assert r.status_code == 409 and r.json()["reason"] in {"depth", "type_not_allowed"}
