"""Spec rendering: ${VAR} substitution, overlays and --ref (flock.specrender)."""
from __future__ import annotations

import json
import subprocess

import pytest
import yaml

from flock import cli
from flock.specrender import merge, read_env_file, render, substitute

from .conftest import make_spec
from .test_cli import _wire, runner


def test_substitution_types_interpolates_defaults_and_escapes():
    env = {"N": "7", "HOST": "gpu1", "FLAG": "false", "BLANK": ""}
    data = {"replicas": "${N}", "other": "${M:-20}", "ratio": "${R:-0.5}", "on": "${FLAG}",
            "url": "http://${HOST}:${PORT:-8000}/v1", "model": "vllm:${MODEL:-MODEL_WORKER}",
            "world": "http://$${OTHER}:7860", "empty": "${E:-}", "${N}": ["${HOST}"],
            "blank": "${BLANK}", "blank_default": "${BLANK:-d}", "runtime": "${WORLD_HOST}"}
    assert substitute(data, env) == {
        "replicas": 7, "other": 20, "ratio": 0.5, "on": False,
        "url": "http://gpu1:8000/v1", "model": "vllm:MODEL_WORKER",
        "world": "http://${OTHER}:7860", "empty": "", "${N}": ["gpu1"],  # keys untouched
        "blank": "", "blank_default": "d", "runtime": "${WORLD_HOST}"}
    assert substitute({"m": "${HOST}"}, {"HOST": "a: b"}) == {"m": "a: b"}  # only scalars typed
    # inline prompts: lowercase loop placeholders and a shell `$$` stay; WORLD_HOST is never ours
    prompts = {"p": "spent ${spend} of ${budget:-x}", "cmd": "echo $$ $${HOME}",
               "url": "http://${WORLD_HOST}:7860"}
    assert substitute(prompts, {"WORLD_HOST": "node1"}) == {
        "p": "spent ${spend} of ${budget:-x}", "cmd": "echo $$ ${HOME}",
        "url": "http://${WORLD_HOST}:7860"}


def test_unset_variables_are_listed():
    with pytest.raises(ValueError, match="unset variables without a default: A, B"):
        substitute({"x": "${B}", "y": ["${A}", "${C:-ok}", "${A}"]}, {})


def test_env_files_and_process_environment_precedence(tmp_path, monkeypatch):
    one, two = tmp_path / "one.env", tmp_path / "two.env"
    one.write_text("# run\nA=1\nB='from one'\nexport C=\"c\"\n\n")
    two.write_text("A=2\n")
    assert read_env_file(one) == {"A": "1", "B": "from one", "C": "c"}
    spec = tmp_path / "spec.yaml"
    spec.write_text("a: ${A}\nb: ${B}\nc: ${C}\n")
    monkeypatch.delenv("A", raising=False)
    monkeypatch.setenv("C", "env")
    assert render(spec, [one, two]) == {"a": 2, "b": "from one", "c": "env"}
    (tmp_path / "bad.env").write_text("A=1\nnot a line\n")
    with pytest.raises(ValueError, match="bad.env:2: expected NAME=value"):
        read_env_file(tmp_path / "bad.env")


def test_overlay_merges_maps_replaces_lists_and_null_deletes():
    base = {"compute": "local", "world": {"dir": "w", "command": ["x"], "token_env": "T"},
            "types": {"a": {"tools": ["x", "y"], "replicas": 0}}}
    over = {"compute": {"kind": "slurm"}, "world": {"dir": None, "command": None, "url": "u"},
            "types": {"a": {"tools": ["z"]}}, "bucket": "b"}
    assert merge(base, over) == {
        "compute": {"kind": "slurm"}, "world": {"token_env": "T", "url": "u"},
        "types": {"a": {"tools": ["z"], "replicas": 0}}, "bucket": "b"}
    assert base["world"]["dir"] == "w"  # the base is not modified


def test_cli_apply_with_env_and_overlay(tmp_path, monkeypatch, app, store):
    _wire(monkeypatch, app)
    data = make_spec().model_dump(mode="json")
    data["types"]["researcher"]["replicas"] = "${REPLICAS:-1}"
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(data))
    (tmp_path / "run.env").write_text("REPLICAS=4\n")
    (tmp_path / "over.yaml").write_text("namespace: ${NS}\n")
    r = runner.invoke(cli.app, ["apply", str(spec), "--overlay", str(tmp_path / "over.yaml")])
    assert r.exit_code == 1 and "unset variables without a default: NS" in r.output
    (tmp_path / "ns.env").write_text("NS=other\n")
    r = runner.invoke(cli.app, ["apply", str(spec), "--env", str(tmp_path / "run.env"),
                                "--env", str(tmp_path / "ns.env"),
                                "--overlay", str(tmp_path / "over.yaml")])
    assert r.exit_code == 0, r.output
    stored = store.get_swarm("demo")["spec"]
    assert stored.types["researcher"].replicas == 4 and stored.namespace == "other"


def test_update_ref_applies_the_committed_folder(tmp_path, monkeypatch, app, store):
    _wire(monkeypatch, app)
    monkeypatch.setenv("USER", "carlos")
    repo = tmp_path / "repo"
    exp = repo / "exp"
    exp.mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.name=t",
                                     "-c", "user.email=t@t", *a], check=True,
                                    capture_output=True, text=True).stdout.strip()
    git("init", "-q", "-b", "main")
    data = make_spec().model_dump(mode="json")
    (exp / "spec.yaml").write_text(yaml.safe_dump(data))
    (exp / "over.yaml").write_text("namespace: committed\n")
    git("add", ".")
    git("commit", "-qm", "v1")
    git("tag", "v1")
    sha = git("rev-parse", "HEAD")
    data["types"]["researcher"]["replicas"] = 9  # uncommitted: --ref must not see it
    (exp / "spec.yaml").write_text(yaml.safe_dump(data))
    (exp / "over.yaml").write_text("namespace: working-tree\n")
    r = runner.invoke(cli.app, ["apply", str(exp / "spec.yaml"), "--ref", "v1",
                                "--overlay", str(exp / "over.yaml")])
    assert r.exit_code == 0, r.output
    (gen1,) = store.spec_history("demo")
    assert gen1["source"] == {"git_sha": sha, "ref": "v1", "dirty": False, "by": "carlos"}
    stored = store.get_swarm("demo")["spec"]
    assert stored.types["researcher"].replicas == 3 and stored.namespace == "committed"
    # update --check with REF:PATH (the positional only names the repo); nothing stored
    r = runner.invoke(cli.app, ["update", str(repo), "--ref", "main:exp", "--check",
                                "--overlay", str(exp / "over.yaml")])
    assert r.exit_code == 0 and "check ok" in r.output, r.output
    assert len(store.spec_history("demo")) == 1
    r = runner.invoke(cli.app, ["update", str(exp), "--ref", "nope"])
    assert r.exit_code == 1 and "--ref nope" in r.output
    hdr = cli.source_header(exp / "spec.yaml")["X-Flock-Source"]
    assert json.loads(hdr)["dirty"] is True  # without --ref: the working tree, as before


def test_base_urls_take_a_comma_separated_string():
    from flock.models import ProviderSpec

    p = ProviderSpec(type="openai_compat", api_key_env="K", base_urls="http://a/v1, http://b/v1")
    assert p.base_urls == ["http://a/v1", "http://b/v1"] and p.base_url == "http://a/v1"
    one = ProviderSpec(type="openai_compat", api_key_env="K", base_url="http://c/v1", base_urls="")
    assert one.base_urls is None and one.endpoints == ["http://c/v1"]
