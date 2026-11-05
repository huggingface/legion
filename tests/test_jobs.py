"""HubJobsClient launch arguments: bootstrap mode vs a prebuilt image."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from flock.controller.jobs import (
    BOOTSTRAP_IMAGE,
    HF_HUB_VERSION,
    PI_VERSION,
    WORKER_COMMAND,
    HubJobsClient,
    bootstrap_command,
)
from flock.controller.main import parse_labels
from flock.models import TypeSpec

WHEEL = "hf://buckets/ns/sh/dist/abc/flock-0.0.1-py3-none-any.whl"


def _spec(**kw) -> TypeSpec:
    return TypeSpec(harness="pi", model="claude-haiku-5-5", **kw)


def test_bootstrap_command_installs_pinned_tools_then_execs_worker(tmp_path):
    cmd = bootstrap_command()
    assert cmd[:2] == ["bash", "-c"]
    script = cmd[2]
    assert f"@earendil-works/pi-coding-agent@{PI_VERSION}" in script
    assert "astral.sh/uv/install.sh" in script
    assert 'hf buckets cp "$1" "$2"' in script and 'fetch "$FLOCK_WHEEL_URL"' in script
    assert "download_bucket_files" in script  # Python API fallback
    assert 'uv tool install -q "$WHEEL[worker]"' in script and "pip install" in script
    for env in ("WORK_ROOT=/work/agents", "PI_CODING_AGENT_DIR=/state/pi"):
        assert env in script
    assert script.rstrip().endswith("exec flock-worker")
    assert "{" + "pi}" not in script and "{" + "hub}" not in script  # all pins filled in
    f = tmp_path / "bootstrap.sh"
    f.write_text(script)
    subprocess.run(["bash", "-n", str(f)], check=True)


def _client(monkeypatch, image, bootstrap_env=None, wheel=WHEEL) -> HubJobsClient:
    if bootstrap_env is None:
        monkeypatch.delenv("FLOCK_WORKER_BOOTSTRAP", raising=False)
    else:
        monkeypatch.setenv("FLOCK_WORKER_BOOTSTRAP", bootstrap_env)
    return HubJobsClient(token="x", image=image, wheel_url=wheel)


def test_bootstrap_installs_pi_only_on_pi_workers(monkeypatch):
    script = bootstrap_command()[2]
    assert 'if [ "${FLOCK_HARNESS:-}" = pi ]' in script
    assert f"@earendil-works/pi-coding-agent@{PI_VERSION}" in script
    for env in ("PI_CODING_AGENT_DIR=/state/pi", "PI_OFFLINE=1", "PI_TELEMETRY=0", "FLOCK_PI_CLI="):
        assert env in script
    c = _client(monkeypatch, image=None)
    args = c.launch_args(TypeSpec(harness="pi", model="m"), {"WORKER_ID": "w"})
    assert args["env"]["FLOCK_HARNESS"] == "pi"


MSGBOARD = "msgboard-0.1.0-py3-none-any.whl"
# what git asks a credential helper for a huggingface.co clone
FILL = ("printf 'protocol=https\\nhost=huggingface.co\\n\\n' | git credential fill 2>/dev/null"
        " | grep password= >> LOG; exit 0")


def fake_wheel(path: Path, requires: tuple[str, ...] = ()) -> Path:
    """A wheel uv can read: name and version from the file name, metadata only."""
    name, version = path.name.split("-")[:2]
    info = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{info}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                   + "".join(f"Requires-Dist: {r}\n" for r in requires))
        z.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        z.writestr(f"{info}/RECORD", "")
    return path


@pytest.mark.parametrize("harness", ["pi", "harbor"])
def test_bootstrap_installs_each_harness_its_extra(tmp_path, harness):
    """The bootstrap script on stubs (uv, hf from a local bucket, npm, curl, mkdir, flock,
    flock-worker). A harbor worker installs flock[harbor] on Python 3.12 with uv.lock's versions
    as constraints and msgboard overridden to the wheel shipped next to flock's (never its
    private git source), gets shepherd's git credential helper for huggingface.co, and no Pi
    CLI (Pi runs inside each Trial's environment)."""
    stubs, log, bucket = tmp_path / "stubs", tmp_path / "calls.log", tmp_path / "bucket"
    wheels = tmp_path / "wheels"  # the script's /opt/wheels
    (bucket / "ns/sh/dist").mkdir(parents=True)
    (bucket / "ns/sh/dist/w.whl").write_text("")
    deps = tmp_path / "deps"
    deps.mkdir()
    fake_wheel(deps / MSGBOARD)
    (deps / "constraints.txt").write_text("harbor==0.25.0 ; python_full_version >= '3.12'\n")
    with tarfile.open(bucket / "ns/sh/dist/harbor-deps.tar", "w") as tar:
        tar.add(deps, arcname=".")
    stubs.mkdir()
    wheels.mkdir()
    for name, body in [
        ("uv", 'for a; do [ "${prev:-}" = --overrides ] && cat "$a" >> LOG; prev=$a; done'),
        ("hf", f'cp "{bucket}/${{3#hf://buckets/}}" "$4"'),
        ("flock-worker", FILL),
        *((name, "") for name in ("npm", "curl", "mkdir", "flock")),
    ]:
        (stubs / name).write_text(f'#!/bin/sh\necho "{name} $*" >> {log}\n'
                                  + body.replace("LOG", str(log)) + "\n")
        (stubs / name).chmod(0o755)
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path), "FLOCK_HARNESS": harness,
           "FLOCK_WHEEL_URL": "hf://buckets/ns/sh/dist/w.whl", "HF_TOKEN": "hf_test_token"}
    cmd = bootstrap_command()
    r = subprocess.run([*cmd[:2], cmd[2].replace("/opt/wheels", str(wheels))], env=env,
                       capture_output=True, text=True, timeout=30, check=False)
    assert r.returncode == 0, r.stderr
    calls = log.read_text().splitlines()
    assert any(c.startswith("npm i -g") for c in calls) == (harness == "pi")
    if harness == "pi":
        assert f"uv tool install -q {wheels}/w.whl[worker] --with huggingface_hub=={HF_HUB_VERSION}" in calls
        assert calls[-1] == "flock-worker "  # no git helper
        return
    i = calls.index(f"uv tool install -q --python 3.12 {wheels}/w.whl[harbor] -c "
                    f"{wheels}/constraints.txt --overrides {wheels}/overrides.txt")
    assert calls[i + 1] == f"msgboard @ file://{wheels}/{MSGBOARD}"  # the overrides file
    assert (wheels / "constraints.txt").read_text().startswith("harbor==0.25.0")
    assert calls[-2:] == ["flock-worker ", "password=hf_test_token"]


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
def test_the_override_replaces_msgboards_git_source(tmp_path):
    """What the harbor bootstrap relies on: uv's --overrides replaces the harbor extra's
    `msgboard @ git+` requirement (pyproject's, in flock's wheel metadata) with the shipped
    wheel. Offline and without the override, uv cannot resolve it."""
    harbor = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())[
        "project"]["optional-dependencies"]["harbor"]
    (git,) = [r for r in harbor if r.startswith("msgboard @ git+")]
    flock = fake_wheel(tmp_path / "flock-0.0.1-py3-none-any.whl", (f'{git} ; extra == "harbor"',))
    (tmp_path / "req.in").write_text(f"flock[harbor] @ file://{flock}\n")
    (tmp_path / "overrides.txt").write_text(f"msgboard @ file://{fake_wheel(tmp_path / MSGBOARD)}\n")

    def compile_(*extra):
        return subprocess.run(["uv", "pip", "compile", "-q", "--offline", "--no-index",
                               "--no-cache", *extra, str(tmp_path / "req.in")],
                              capture_output=True, text=True, timeout=60, check=False,
                              env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})

    r = compile_("--overrides", str(tmp_path / "overrides.txt"))
    assert r.returncode == 0, r.stderr
    assert f"msgboard @ file://{tmp_path}/{MSGBOARD}" in r.stdout and "git+" not in r.stdout
    assert compile_().returncode != 0


def test_launch_args_bootstrap_flag_overrides_default_image(monkeypatch):
    c = _client(monkeypatch, image="ghcr.io/x/worker:1", bootstrap_env="1")
    assert c.launch_args(_spec(), {})["image"] == BOOTSTRAP_IMAGE
    # a type's own image always wins
    args = c.launch_args(_spec(image="me/custom:2"), {"A": "1"})
    assert args == {"image": "me/custom:2", "command": WORKER_COMMAND, "env": {"A": "1"}}


def test_launch_args_prebuilt_image(monkeypatch):
    c = _client(monkeypatch, image="ghcr.io/x/worker:1", wheel=None)
    args = c.launch_args(_spec(), {"A": "1"})
    assert args == {"image": "ghcr.io/x/worker:1", "command": WORKER_COMMAND, "env": {"A": "1"}}


def test_bootstrap_without_wheel_url_fails_fast(monkeypatch):
    monkeypatch.delenv("FLOCK_WHEEL_URL", raising=False)
    with pytest.raises(ValueError, match="FLOCK_WHEEL_URL"):
        _client(monkeypatch, image=None, wheel=None)


def test_parse_job_labels():
    assert parse_labels("experiment=flock-m1, team = eval") == {
        "experiment": "flock-m1",
        "team": "eval",
    }
    assert parse_labels("") == {}
