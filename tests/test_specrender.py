"""Spec rendering: ${VAR} substitution, value sources and --ref (flock.specrender)."""
from __future__ import annotations

import json
import subprocess

import pytest
import yaml

from flock import cli, ops
from flock.specrender import read_env_file, render, substitute

from .conftest import make_spec, wire
from .test_cli import runner


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


def test_env_files_win_over_the_process_environment(tmp_path, monkeypatch):
    one, two = tmp_path / "one.env", tmp_path / "two.env"
    one.write_text("# run\nA=1\nB='from one'\nexport C=\"c\"\n\n")
    two.write_text("A=2\n")
    assert read_env_file(one) == {"A": "1", "B": "from one", "C": "c"}
    spec = tmp_path / "spec.yaml"
    monkeypatch.delenv("A", raising=False)
    monkeypatch.setenv("C", "env")
    monkeypatch.setenv("D", "shell")
    spec.write_text("a: ${A}\nb: ${B}\nc: ${C}\nd: ${D}\n")
    assert render(spec, [one, two]) == {"a": 2, "b": "from one", "c": "c", "d": "shell"}
    (tmp_path / "bad.env").write_text("A=1\nnot a line\n")
    with pytest.raises(ValueError, match="bad.env:2: expected NAME=value"):
        read_env_file(tmp_path / "bad.env")


def test_env_file_trailing_comments_as_a_shell_reads_them(tmp_path):
    f = tmp_path / "cluster.env"
    f.write_text("P=cpu   # the partition\nU=https://h/a#b\nQ='x # y'  # quoted\nE=  # empty\n")
    assert read_env_file(f) == {"P": "cpu", "U": "https://h/a#b", "Q": "x # y", "E": ""}


def test_cli_apply_with_env_files_and_no_overlay(tmp_path, monkeypatch, app, store):
    wire(monkeypatch, app)
    monkeypatch.delenv("NS", raising=False)
    data = make_spec().model_dump(mode="json")
    data["types"]["researcher"]["replicas"] = "${REPLICAS:-1}"
    data["namespace"] = "${NS}"
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(data))
    (tmp_path / "run.env").write_text("REPLICAS=4\n")
    r = runner.invoke(cli.app, ["launch", str(spec)])
    assert r.exit_code == 1 and "unset variables without a default: NS" in r.output
    (tmp_path / "ns.env").write_text("NS=other\n")
    r = runner.invoke(cli.app, ["launch", str(spec), "--env", str(tmp_path / "run.env"),
                                "--env", str(tmp_path / "ns.env")])
    assert r.exit_code == 0, r.output
    stored = store.get_swarm("demo")["spec"]
    assert stored.types["researcher"].replicas == 4 and stored.namespace == "other"
    for cmd in ("launch", "update"):  # one spec per run: no overlays on the live path
        r = runner.invoke(cli.app, [cmd, str(spec), "--overlay", str(spec)])
        assert r.exit_code == 2 and "--overlay" in r.output


def test_value_sources_lines_env_files_defaults(tmp_path, monkeypatch):
    from flock.specrender import source_of

    monkeypatch.setenv("MEM", "shell")
    monkeypatch.delenv("CPUS", raising=False)
    spec = tmp_path / "spec.yaml"
    spec.write_text("swarm: s\ncompute:\n  cpus: ${CPUS:-4}\n  mem: ${MEM}\n"
                    "types:\n  a: &a\n    tools: [x]\n  b:\n    <<: *a\n    model: m\n")
    (tmp_path / "run.env").write_text("MEM=7G\n")
    src: dict[str, str] = {}
    render(spec, [tmp_path / "run.env"], sources=src)
    assert source_of(src, "compute.cpus") == "spec.yaml:3 default of CPUS"
    assert source_of(src, "compute.mem") == "run.env MEM"
    assert source_of(src, "types.b.tools") == "spec.yaml:7"  # merged from the anchor
    assert source_of(src, "types.b.model") == "spec.yaml:10"
    assert source_of(src, "types.b.loop.while_.until") == "spec.yaml:9"  # nearest parent
    assert source_of(src, "limits.alive") == "default"
    src.clear()
    render(spec, [], sources=src)
    assert source_of(src, "compute.mem") == "env MEM"


def test_update_ref_applies_the_committed_folder(tmp_path, monkeypatch, app, store):
    wire(monkeypatch, app)
    monkeypatch.setenv("USER", "alice")
    repo = tmp_path / "repo"
    exp = repo / "exp"
    exp.mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.name=t",
                                     "-c", "user.email=t@t", *a], check=True,
                                    capture_output=True, text=True).stdout.strip()
    git("init", "-q", "-b", "main")
    data = make_spec().model_dump(mode="json")
    data["namespace"] = "committed"
    (exp / "spec.yaml").write_text(yaml.safe_dump(data))
    git("add", ".")
    git("commit", "-qm", "v1")
    git("tag", "v1")
    sha = git("rev-parse", "HEAD")
    data["types"]["researcher"]["replicas"] = 9  # uncommitted: --ref must not see it
    data["namespace"] = "working-tree"
    (exp / "spec.yaml").write_text(yaml.safe_dump(data))
    r = runner.invoke(cli.app, ["launch", str(exp / "spec.yaml"), "--ref", "v1"])
    assert r.exit_code == 0, r.output
    (gen1,) = store.spec_history("demo")
    assert gen1["source"] == {"git_sha": sha, "ref": "v1", "dirty": False, "by": "alice"}
    stored = store.get_swarm("demo")["spec"]
    assert stored.types["researcher"].replicas == 3 and stored.namespace == "committed"
    # update --check with REF:PATH (the positional only names the repo); nothing stored
    r = runner.invoke(cli.app, ["update", str(repo), "--ref", "main:exp", "--check"])
    assert r.exit_code == 0 and "check ok" in r.output, r.output
    assert len(store.spec_history("demo")) == 1
    r = runner.invoke(cli.app, ["update", str(exp), "--ref", "nope"])
    assert r.exit_code == 1 and "--ref nope" in r.output
    hdr = ops.source_header(exp / "spec.yaml")["X-Flock-Source"]
    assert json.loads(hdr)["dirty"] is True  # without --ref: the working tree


def test_urls_take_a_comma_separated_string():
    from flock.models import Inference

    i = Inference(urls="http://a/v1, http://b/v1/")  # `urls: ${VLLM_URLS}`
    assert i.urls == i.endpoints == ["http://a/v1", "http://b/v1"]
