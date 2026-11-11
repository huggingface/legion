"""Tools folder shipping: `flock launch` packs and uploads it, the controller stores and checks it,
assigns and heartbeats carry the version."""

from __future__ import annotations

import textwrap
from pathlib import Path

import yaml
from typer.testing import CliRunner

from flock import cli
from flock.models import AssignCommand, SwarmSpec
from flock.tools.bundle import pack

from .conftest import FAKE_PROVIDERS, HB, wire

runner = CliRunner()

ADD_V1 = '''
def add(a: int, b: int) -> int:
    """Add two integers (v1)."""
    return a + b
'''
ADD_V2 = ADD_V1 + '''

def mul(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
'''


def folder(root: Path, src: str = ADD_V1, name: str = "tools") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "calc.py").write_text(textwrap.dedent(src))
    return d


def raw_spec(tools=None, **extra) -> dict:
    t = {**HB, "model": "fake:m", "replicas": 1, "agents_per_worker": 2}
    if tools is not None:
        t["tools"] = tools
    return {"swarm": "demo", "namespace": "ns", "bucket": "hf://buckets/ns/sh", "providers": FAKE_PROVIDERS,
            "types": {"dev": t}, **extra}


def write_spec(tmp_path: Path, raw: dict) -> Path:
    f = tmp_path / "spec.yaml"
    f.write_text(yaml.safe_dump(raw))
    return f


# --------------------------------------------------------------------------- apply + controller


def test_apply_uploads_the_folder_and_pins_its_version(tmp_path, monkeypatch, app, store):
    wire(monkeypatch, app)
    folder(tmp_path)
    f = write_spec(tmp_path, raw_spec(tools=["add"]))
    r = runner.invoke(cli.app, ["launch", str(f)])
    assert r.exit_code == 0, r.output
    v1 = pack(tmp_path / "tools").version
    assert f"tools: version {v1}" in r.output and ": add" in r.output
    sw = store.get_swarm("demo")["spec"]
    assert sw.tools_version == v1 and sw.types["dev"].tools == ["add"]
    assert store.get_tool_bundle("demo", v1)["tools"] == ["add"]

    # same folder: a resume, same version; changed folder: `launch` refuses, `update` ships it
    assert runner.invoke(cli.app, ["launch", str(f)]).exit_code == 0
    assert store.get_swarm("demo")["spec"].tools_version == v1
    folder(tmp_path, ADD_V2)
    f = write_spec(tmp_path, raw_spec(tools="all"))
    r = runner.invoke(cli.app, ["launch", str(f)])
    assert r.exit_code == 1 and "run `flock update` first" in r.output
    r = runner.invoke(cli.app, ["update", str(f)])
    assert r.exit_code == 0, r.output
    v2 = store.get_swarm("demo")["spec"].tools_version
    assert v2 != v1 and store.get_tool_bundle("demo", v2)["tools"] == ["add", "mul"]
    assert [b["version"] for b in store.list_tool_bundles("demo")] == [v2, v1]


def test_apply_refuses_unknown_tools_and_bad_modules(tmp_path, monkeypatch, app, store):
    wire(monkeypatch, app)
    folder(tmp_path)
    f = write_spec(tmp_path, raw_spec(tools=["add", "mul"]))
    r = runner.invoke(cli.app, ["launch", str(f)])
    assert r.exit_code == 1
    assert "types.dev.tools: unknown tool(s) mul; the tools folder has: add" in r.output
    assert store.get_swarm("demo") is None  # nothing applied
    (tmp_path / "tools" / "broken.py").write_text("def f(:\n")
    f = write_spec(tmp_path, raw_spec(tools=["add"]))
    r = runner.invoke(cli.app, ["launch", str(f)])
    assert r.exit_code == 1 and "broken.py: syntax error line 1" in r.output


def test_existing_specs_are_unchanged(tmp_path, monkeypatch, app, store):
    """No type selects tools: an unrelated tools/ next to the spec is not shipped."""
    wire(monkeypatch, app)
    folder(tmp_path)
    f = write_spec(tmp_path, raw_spec())
    r = runner.invoke(cli.app, ["launch", str(f)])
    assert r.exit_code == 0 and "tools:" not in r.output
    assert store.get_swarm("demo")["spec"].tools_version is None
    assert store.list_tool_bundles("demo") == []


async def test_controller_checks_bundles_and_tells_workers(tmp_path, client, store, tick):
    b = pack(folder(tmp_path))
    spec = SwarmSpec.model_validate(raw_spec(tools=["add"]))
    r = await client.put("/swarms/demo", json=spec.model_dump(mode="json"))
    assert r.status_code == 400 and "no tools_version" in r.text
    spec.tools_version = b.version
    r = await client.put("/swarms/demo", json=spec.model_dump(mode="json"))
    assert r.status_code == 400 and "was not uploaded" in r.text
    r = await client.post("/swarms/demo/tools", content=b"garbage")
    assert r.status_code == 400 and "not a tools bundle" in r.text
    r = await client.post("/swarms/demo/tools", content=b.data,
                          headers={"content-type": "application/gzip"})
    assert r.status_code == 201 and r.json() == {"swarm": "demo", "version": b.version,
                                                 "size": len(b.data), "tools": ["add"]}
    spec.types["dev"].tools = ["add", "nope"]
    r = await client.put("/swarms/demo", json=spec.model_dump(mode="json"))
    assert r.status_code == 400 and "types.dev.tools: unknown tool(s) nope" in r.text
    spec.types["dev"].tools = ["add"]
    r = await client.put("/swarms/demo", json=spec.model_dump(mode="json"))
    assert r.status_code == 200, r.text
    r = await client.get(f"/swarms/demo/tools/{b.version}")
    assert r.status_code == 200 and r.content == b.data
    assert (await client.get("/swarms/demo/tools/0000000000000000")).status_code == 404
    assert (await client.get("/swarms/demo/tools")).json()["current"] == b.version

    # the assign carries the version
    tick()
    [w] = store.list_workers("demo")
    [cmd] = [AssignCommand.model_validate_json(c["body"]) for c in store.commands_for(w["id"])]
    assert cmd.tools_version == b.version and cmd.spec.tools == ["add"]


async def test_bundle_retention_keeps_the_current_version(store):
    from flock.controller.store import Store

    spec = SwarmSpec.model_validate({**raw_spec(), "tools_version": f"{0:016x}"})
    store.put_swarm(spec, 0.0)
    for i in range(Store.TOOL_BUNDLES_KEPT + 3):
        store.put_tool_bundle("demo", f"{i:016x}", b"x", [], float(i))
    kept = {b["version"] for b in store.list_tool_bundles("demo")}
    assert len(kept) == Store.TOOL_BUNDLES_KEPT + 1 and f"{0:016x}" in kept
    store.delete_swarm("demo")
    assert store.list_tool_bundles("demo") == []


# --------------------------------------------------------------------------- CLI harnesses
