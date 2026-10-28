"""Run-4 configuration, all-node capacity guard and notification integration."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import yaml

from flock.cli import load_spec
from flock.controller.loop import notice_line

EXP = Path(__file__).resolve().parents[1] / "experiments/run4"


def module(name):
    spec = importlib.util.spec_from_file_location("run4_" + name, EXP / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def test_spec_fixed_prompt_and_waves():
    spec = load_spec(str(EXP / "spec.yaml"))
    assert set(spec.types) == {"agent", "agent-spawner", "agent-child"}  # test_run4_spawner
    agent = spec.types["agent"]
    assert agent.harness.value == "pi" and agent.agents_per_worker == 12
    assert agent.max_turns == 150 and agent.turn_timeout == "2h"
    assert agent.worker_lifetime == "54h"
    assert agent.loop.first is None and agent.loop.notices
    assert agent.loop.prompt.strip() == "Continue your work."
    assert "Keep the seed's input/output interface fixed" in agent.system_prompt
    assert "5 min in total" in agent.system_prompt
    assert not spec.comms.protocols
    assert set(agent.tools) == {"world:run_bot", "world:open_pr", "world:review_pr",
                                "world:merge_pr", "world:list_prs", "world:close_pr",
                                "world:get_pr"}
    waves = module("waves").load(EXP / "waves.yaml")
    assert waves.cap == 300 and waves.split(300) == {"agent": 300}
    assert waves.events[:5] == [(0, 20), (300, 40), (600, 60), (900, 80), (1200, 100)]
    assert waves.events[-1] == (7200, 300)


def test_render_multiple_endpoints_and_cluster_lifetime(monkeypatch, tmp_path):
    """render.py is flock's own --env/--overlay plus OPENAI_BASE_URLS; its output reads back
    unchanged (the rendered file goes through substitution again at `flock apply`)."""
    for name in ("RUN3_BUCKET", "RUN3_NAMESPACE", "CHATBOT_RUN", "MODEL_WORKER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URLS", "http://first/v1,http://second/v1")
    (tmp_path / "run.env").write_text("MODEL_WORKER=fake-model\nRUN3_BUCKET=hf://buckets/x/y\n")
    data = module("render").render(env_files=[tmp_path / "run.env"])
    smoke = module("render").render("smoke", env_files=[tmp_path / "run.env"])
    assert all(t["loop"]["while"] == {"until": "30m", "max_rounds": 100}
               for t in smoke["types"].values()) and smoke["compute"] == "local"
    assert data["compute"] == {"kind": "slurm", "partition": "hopper-cpu", "cpus": 4,
                               "mem": "7G", "time": "54:00:00"}
    assert data["bucket"] == "hf://buckets/x/y" and data["namespace"] == "cmpatino"
    assert data["world"] == {"url": "http://${WORLD_HOST}:7860", "token_env": "CHATBOT_WORLD_TOKEN"}
    assert data["models"]["worker-pi"]["model"] == "vllm:fake-model"
    tools = {"world:run_bot", "world:open_pr", "world:review_pr", "world:merge_pr",
             "world:list_prs", "world:close_pr", "world:get_pr"}
    for t in data["types"].values():  # the overlay keeps every type's settings from spec.yaml
        assert t["replicas"] == 0 and t["harness"] == "pi" and set(t["tools"]) == tools
    out = tmp_path / "experiment" / "spec.slurm.local.yaml"
    out.parent.mkdir()
    (out.parent / "prompts").symlink_to(EXP / "prompts")
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    spec = load_spec(str(out))  # as controller.sbatch applies it
    assert spec.world.url == "http://${WORLD_HOST}:7860" and spec.compute.kind == "slurm"
    assert set(spec.types["agent-child"].tools) == tools
    assert spec.providers["vllm"].base_urls == ["http://first/v1", "http://second/v1"]
    assert spec.providers["vllm"].base_url == "http://first/v1"


def test_capacity_guard_checks_each_node_and_holds_on_missing_metrics():
    guard = module("load_guard")
    def response(waiting, completed):
        return SimpleNamespace(raise_for_status=lambda: None, text=(
            f"vllm:num_requests_running 10\nvllm:num_requests_waiting {waiting}\n"
            f"vllm:request_success_total{{finished_reason=\"stop\"}} {completed}\n"))
    clocks = iter([0, 0, 60, 60, 120, 120])
    counters = {"one": 0, "two": 0}
    def healthy(url):
        counters[url] += 60
        return response(0, counters[url])
    ok, _ = guard.capacity_ok(["one", "two"], get=healthy, sleep=lambda _: None,
                             clock=lambda: next(clocks))
    assert ok and counters == {"one": 180, "two": 180}
    clocks = iter([0, 0, 60, 60, 120, 120])
    ok, message = guard.capacity_ok(["one", "two"], get=lambda url: response(url == "two", 100),
                                   sleep=lambda _: None, clock=lambda: next(clocks))
    assert not ok and message.startswith("HOLD")
    ok, _ = guard.capacity_ok(["one"], get=lambda _: (_ for _ in ()).throw(httpx.ConnectError("down")))
    assert not ok


def test_notices_group_dm_channels_and_remain_bounded():
    line = notice_line([{"kind": "dm", "thread": 1, "from": "cedar", "channel": "dm-1"},
                        {"kind": "mention", "thread": 9, "from": "birch", "channel": "memory"}])
    assert "DM from cedar" in line and "#memory" in line
    many = [{"kind": "channel", "thread": i, "from": "cedar", "channel": f"topic-{i}"}
            for i in range(20)]
    assert "+15 more groups" in notice_line(many)


def test_a_wave_raises_the_stored_replicas(tmp_path, monkeypatch, app, store, tick):
    """`flock update` keeps live replicas by default; a wave must apply its own counts."""
    from flock import cli

    from .test_cli import _wire, runner
    from .test_live import _experiment

    _wire(monkeypatch, app)
    f = _experiment(tmp_path / "exp", replicas=2)
    assert runner.invoke(cli.app, ["apply", str(f)]).exit_code == 0
    tick()
    waves = module("waves")
    sched = waves.parse({"cap": 300, "type": "researcher"})
    rec = waves.apply(cli.client(), f, sched, 5, "test", tmp_path / "w.jsonl")
    assert "error" not in rec, rec
    assert store.get_swarm("demo")["spec"].types["researcher"].replicas == 5
