"""The interviewer experiment: hierarchy's teams plus ephemeral interviewers that coordinators
and workers spawn to judge a bot."""
from __future__ import annotations

from pathlib import Path

from flock.cli import load_spec
from flock.models import comms_tools

from .test_hierarchy_experiment import check_one_spec
from .test_run4_spawner import as_agent

ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXP = ROOT / "interviewer"
REPO = "HuggingFaceSwarms/SwarmChat-v9"


def test_types_prompts_and_spawn_policy(monkeypatch):
    monkeypatch.delenv("CHATBOT_BOARD_URL", raising=False)
    spec = load_spec(str(EXP / "spec.yaml"))
    hierarchy = load_spec(str(ROOT / "hierarchy/spec.yaml"))
    assert set(spec.types) == {"coordinator", "worker", "interviewer"}
    skip = {"system_prompt": True, "tools": True}
    for name in ("coordinator", "worker"):  # hierarchy's types plus the verdicts tool
        t = spec.types[name]
        assert t.model_dump(exclude=skip) == hierarchy.types[name].model_dump(exclude=skip)
        assert t.tools == [*hierarchy.types[name].tools, "world:verdicts"]
    iv = spec.types["interviewer"]
    assert iv.ephemeral and iv.replicas == 0 and iv.harness == "llm"
    assert spec.models[iv.model].extra["chat_template_kwargs"] == {"thinking": False}
    assert {"world:run_bot", "world:record_verdict"} <= set(iv.tools)
    assert {"comms:post", "comms:dm", "comms:join_channel"} <= set(comms_tools(iv, spec.comms))
    assert "interviewer" not in spec.spawn  # interviewers cannot spawn
    assert spec.spawn["coordinator"].types == {"worker": 4, "interviewer": 2}
    assert spec.spawn["worker"].types == {"interviewer": 2}
    assert spec.limits.alive >= 50 + 200 + iv.workers.max_extra * iv.agents_per_worker
    assert spec.limits.spawned_total >= 400 + 150 * 48  # ~150 interviews an hour for 48 h
    assert iv.workers.max_extra * iv.agents_per_worker <= 26  # run_bot's run slots
    assert spec.comms.url.endswith("chatbot-v9-msgboard.hf.space/boards/chatbot-v9")
    system = (EXP / "prompts/system.md").read_text()
    assert f"`{REPO}` on Hugging Face" in system and f"datasets/{REPO}` into `chatbot-swarm`" in system
    assert "chatbot-v5" not in system + spec.types["coordinator"].system_prompt
    assert "## Interviews" in system and "flock spawn interviewer" in system
    assert "Treat a `script` verdict as a bug to fix" in system and "team channel" in system
    assert "The real test: a strong interviewer" in system
    assert "understanding engine" in system and "general mechanisms" in system
    assert "## Interviews" in spec.types["worker"].system_prompt
    interviewer = iv.system_prompt
    assert "new key per interview" in system
    for text in ("run_bot", "record_verdict", "`post` in it", "thread: \"new\"", "says busy",
                 "@mentions", "`dm`", "8 to 15"):
        assert text in interviewer, text


def test_one_spec_with_the_ramp_and_the_interviewer_size(monkeypatch):
    spec = check_one_spec(EXP, monkeypatch, "coordinator", [(600 * i, 10 * (i + 1)) for i in range(5)])
    assert set(spec.types) == {"coordinator", "worker", "interviewer"}
    assert spec.compute.types["interviewer"].model_dump(exclude_none=True) == {"cpus": 2, "mem": "3800M"}
    monkeypatch.setenv("RUN3_INTERVIEWER_MEM", "3G")  # run.env sizes it; the spec holds it
    assert load_spec(str(EXP / "spec.yaml")).compute.types["interviewer"].mem == "3G"


async def test_coordinators_and_workers_spawn_interviewers_with_caps(client, app, store, tick):
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

    async def spawn(c, parent, type_, key):
        return await c.post(f"/agents/{parent}/spawn", json={"type": type_, "key": key})

    async with as_agent(app, spawning.agent_token(store, coord, spec.swarm, 0)) as c:
        workers = [(await spawn(c, coord, "worker", f"w{i}")).json()["name"] for i in range(4)]
        assert (await spawn(c, coord, "worker", "w4")).json()["reason"] == "max_children"
        for i in range(2):
            assert (await spawn(c, coord, "interviewer", f"i{i}")).status_code == 201
        r = await spawn(c, coord, "interviewer", "i2")
        assert r.status_code == 409 and r.json()["reason"] == "max_children"
    async with as_agent(app, spawning.agent_token(store, workers[0], spec.swarm, 0)) as c:
        ivs = []
        for i in range(2):
            r = await spawn(c, workers[0], "interviewer", f"i{i}")
            assert r.status_code == 201 and r.json()["depth"] == 0, r.text
            ivs.append(r.json()["name"])
        assert (await spawn(c, workers[0], "interviewer", "i2")).json()["reason"] == "max_children"
        r = await spawn(c, workers[0], "worker", "w")
        assert r.status_code == 409 and r.json()["reason"] == "type_not_allowed"
    async with as_agent(app, spawning.agent_token(store, ivs[0], spec.swarm, 0)) as c:
        r = await spawn(c, ivs[0], "interviewer", "x")
        assert r.status_code == 409 and r.json()["reason"] in {"depth", "type_not_allowed"}
