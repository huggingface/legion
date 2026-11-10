import pytest

from flock.harbor.resources import Resources


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

