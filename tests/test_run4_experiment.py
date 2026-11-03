"""Run-4 configuration (one spec.yaml: cluster settings and the two-phase ramp) and notifications."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from flock.cli import load_spec
from flock.controller.loop import notice_line

EXP = Path(__file__).resolve().parents[1] / "experiments/run4"


def module(name):
    spec = importlib.util.spec_from_file_location("run4_" + name, EXP / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def test_spec_fixed_prompt_and_ramp():
    spec = load_spec(str(EXP / "spec.yaml"))
    assert set(spec.types) == {"agent", "agent-spawner", "agent-child"}  # test_run4_spawner
    agent = spec.types["agent"]
    assert agent.harness.value == "pi" and agent.agents_per_worker == 12
    assert agent.max_turns == 150 and agent.turn_timeout == "2h"
    assert agent.worker_lifetime == "54h"
    assert agent.loop.notices
    assert agent.loop.prompt.strip() == "Continue your work."
    assert "Keep the seed's input/output interface fixed" in agent.system_prompt
    assert "5 min in total" in agent.system_prompt
    assert not spec.comms.protocols
    assert set(agent.tools) == {"world:run_bot", "world:open_pr", "world:review_pr",
                                "world:merge_pr", "world:list_prs", "world:close_pr",
                                "world:get_pr"}
    ramp = agent.replicas_ramp  # was waves.yaml: 20 every 5 min to 100, then +20 every 10 min
    events = [(t, n) for t, n, _ in ramp.schedule()]
    assert ramp.max == 300 and events[:6] == [(0, 20), (300, 40), (600, 60), (900, 80),
                                              (1200, 100), (1800, 120)]
    assert events[-1] == (7200, 300) and [g for _, _, g in ramp.schedule()][:2] == [10, 0]
    assert ramp.hold_if.above == 100 and ramp.hold_if.request_s == 15
    assert spec.types["agent-spawner"].replicas_ramp is None
    assert spec.types["agent-child"].replicas_ramp is None


def test_env_file_fills_the_cluster_settings(monkeypatch, tmp_path):
    """One spec.yaml: --env run.env fills compute, bucket and endpoints; the file wins over the
    shell; COMPUTE_KIND=local and LOOP_UNTIL make the local smoke."""
    for name in ("RUN3_BUCKET", "RUN3_NAMESPACE", "CHATBOT_RUN", "MODEL_WORKER", "COMPUTE_KIND",
                 "LOOP_UNTIL", "LOOP_MAX_ROUNDS", "WORKER_MEM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URLS", "http://shell/v1")
    monkeypatch.setenv("WORKER_CPUS", "8")
    (tmp_path / "run.env").write_text("MODEL_WORKER=fake-model\nRUN3_BUCKET=hf://buckets/x/y\n"
                                      "OPENAI_BASE_URLS=http://first/v1,http://second/v1\n")
    spec = load_spec(str(EXP / "spec.yaml"), [str(tmp_path / "run.env")])
    assert spec.compute.model_dump(exclude_none=True) == {
        "kind": "slurm", "partition": "hopper-cpu", "cpus": 8, "mem": "7G", "time": "54:00:00"}
    assert spec.bucket == "hf://buckets/x/y" and spec.namespace == "cmpatino"
    assert spec.world.url == "http://${WORLD_HOST}:7860" and spec.world.token_env == "CHATBOT_WORLD_TOKEN"
    assert spec.models["worker-pi"].model == "vllm:fake-model"
    assert spec.providers["vllm"].base_urls == ["http://first/v1", "http://second/v1"]
    tools = {"world:run_bot", "world:open_pr", "world:review_pr", "world:merge_pr",
             "world:list_prs", "world:close_pr", "world:get_pr"}
    for t in spec.types.values():
        assert t.replicas == 0 and t.harness.value == "pi" and set(t.tools) == tools
        assert t.loop.while_.until == "48h" and t.loop.while_.max_rounds == 10000
    monkeypatch.setenv("COMPUTE_KIND", "local")
    monkeypatch.setenv("LOOP_UNTIL", "30m")
    smoke = load_spec(str(EXP / "spec.yaml"))
    assert smoke.compute.kind == "local" and smoke.world.command
    assert all(t.loop.while_.until == "30m" for t in smoke.types.values())


def test_notices_group_dm_channels_and_remain_bounded():
    line = notice_line([{"kind": "dm", "thread": 1, "from": "cedar", "channel": "dm-1"},
                        {"kind": "mention", "thread": 9, "from": "birch", "channel": "memory"}])
    assert "DM from cedar" in line and "#memory" in line
    many = [{"kind": "channel", "thread": i, "from": "cedar", "channel": f"topic-{i}"}
            for i in range(20)]
    assert "+15 more groups" in notice_line(many)
