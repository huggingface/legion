"""Run-4 configuration, all-node capacity guard and notification integration."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx

from flock.controller.loop import notice_line

EXP = Path(__file__).resolve().parents[1] / "experiments/run4"


def module(name):
    spec = importlib.util.spec_from_file_location("run4_" + name, EXP / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


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
