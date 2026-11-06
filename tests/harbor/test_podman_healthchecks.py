# from shepherd f3970bc:tests/test_podman_healthchecks.py
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "podman_healthchecks", Path(__file__).parents[2] / "src/flock/harbor/podman_healthchecks.py"
)
healthchecks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(healthchecks)


async def test_checks_use_declared_interval_and_skip_containers_without_checks(
    monkeypatch,
):
    calls = []
    ticks = iter([0, 2, 4])
    container = {
        "Id": "a" * 64,
        "Config": {"Healthcheck": {"Test": ["CMD", "true"], "Interval": 1_000_000_000}},
    }

    async def podman(*args):
        calls.append(args)
        if args[0] == "ps":
            return 0, b"aaaaaaaaaaaa\nbbbbbbbbbbbb\n", b""
        if args[0] == "inspect":
            return (
                0,
                json.dumps([container, {"Id": "b" * 64, "Config": {}}]).encode(),
                b"",
            )
        # Unhealthy status is recorded by Podman; it must not crash the scheduler.
        return 1, b"", b""

    original_sleep = asyncio.sleep
    sleeps = 0

    async def sleep(_):
        nonlocal sleeps
        sleeps += 1
        await original_sleep(0)
        if sleeps == 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(healthchecks, "podman", podman)
    # Replace the module's asyncio reference, not the event loop's clock/sleep.
    from types import SimpleNamespace

    monkeypatch.setattr(
        healthchecks, "time", SimpleNamespace(monotonic=lambda: next(ticks))
    )
    monkeypatch.setattr(
        healthchecks,
        "asyncio",
        SimpleNamespace(sleep=sleep, create_task=asyncio.create_task),
    )
    with pytest.raises(asyncio.CancelledError):
        await healthchecks.main()
    assert calls.count(("healthcheck", "run", "aaaaaaaaaaaa")) == 2
    assert not any(
        call[:2] == ("healthcheck", "run") and call[2].startswith("b") for call in calls
    )
