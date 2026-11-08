import pytest

from flock.resources import Resources


@pytest.mark.parametrize(
    "options",
    [
        {"url": "https://secret.example.com"},  # removed with the external mode
        {"space_id": "ns/not a space"},
        {"space_id": "ns/board/extra"},
        {"private": True},
        {"channels": []},
        {"channels": ["general", "general"]},
        {"token": "secret"},
    ],
)
def test_board_options_reject_invalid_config_without_echoing_values(
    message_board_provider, tmp_path, options
):
    with pytest.raises(ValueError) as exc:
        message_board_provider(options, state_dir=tmp_path, agents=[], secrets=[])
    assert "secret" not in str(exc.value)


async def test_generic_resources_cleanup_after_partial_startup(tmp_path):
    module = """
class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.state_dir = state_dir
        self.fail = options.get("fail", False)
        self.metadata = {"provider": "sample"}
    async def start(self):
        self.state_dir.mkdir(parents=True)
        if self.fail:
            raise RuntimeError("startup failed")
    async def wait(self):
        import asyncio
        await asyncio.Future()
    async def export(self, directory):
        (directory / "result.txt").write_text("result")
        return {"result": "result.txt"}
    async def stop(self):
        (self.state_dir / "stopped").touch()
"""
    for name in ("first", "second"):
        (tmp_path / f"{name}.py").write_text(module)
    resources = Resources(
        tmp_path,
        {"first": {}, "second": {"fail": True}},
        run_dir=tmp_path / "run",
        agents=[],
        secrets=[],
    )
    with pytest.raises(RuntimeError, match="startup failed"):
        await resources.start()
    assert await resources.finish() is False
    for name in ("first", "second"):
        assert (tmp_path / "run/state/resources" / name / "stopped").exists()
        assert (
            resources.records[name]["artifacts"]["result"]
            == f"resources/{name}/result.txt"
        )


async def test_resource_limits_are_the_resources_own(tmp_path):
    (tmp_path / "slow.py").write_text(
        """
import asyncio
class Resource:
    startup_timeout_seconds = 0.2
    stop_timeout_seconds = 0.3
    def __init__(self, options, *, state_dir, agents, secrets):
        self.metadata = {}
    async def start(self):
        self.metadata["deployed"] = "somewhere"
        await asyncio.sleep(10)
    async def export(self, directory):
        return {}
    async def stop(self):
        await asyncio.sleep(10)
"""
    )
    resources = Resources(
        tmp_path, {"slow": {}}, run_dir=tmp_path / "run", agents=[], secrets=[]
    )
    with pytest.raises(RuntimeError, match="slow did not start within 0.2 s"):
        await resources.start()
    assert resources.records["slow"]["status"] == "failed"
    assert resources.records["slow"]["metadata"] == {"deployed": "somewhere"}
    assert await resources.finish() is True
    assert resources.records["slow"]["cleanup_error"].endswith(
        "Stop did not finish within 0.3 s"
    )


async def test_finish_keeps_a_storage_error_and_still_stops(tmp_path):
    module = """
class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.state_dir = state_dir
        self.metadata = {}
    async def start(self):
        self.state_dir.mkdir(parents=True)
    async def wait(self):
        import asyncio
        await asyncio.Future()
    async def export(self, directory):
        return {}
    async def stop(self):
        (self.state_dir / "stopped").touch()
"""
    for name in ("first", "second"):
        (tmp_path / f"{name}.py").write_text(module)
    resources = Resources(
        tmp_path,
        {"first": {}, "second": {}},
        run_dir=tmp_path / "run",
        agents=[],
        secrets=[],
    )
    await resources.start()
    # A file where the first export directory goes: creating it fails.
    (tmp_path / "run/resources").mkdir(parents=True)
    (tmp_path / "run/resources/first").touch()
    assert await resources.finish() is True
    assert resources.records["first"]["export_error"].startswith("FileExistsError")
    assert "export_error" not in resources.records["second"]
    for name in ("first", "second"):
        assert (tmp_path / "run/state/resources" / name / "stopped").exists()
