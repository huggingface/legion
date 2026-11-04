"""Cluster review regressions using fake commands, virtual time and no cluster access."""
from __future__ import annotations

import asyncio
import importlib
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest

from flock.models import CommsSpec
from flock.worker.comms import CommsHub, Member
from flock.worker.identity import MsgboardIdentity

ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "experiments/run4/slurm"


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o700)
    return path


def shell_environment(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    executable(venv / "bin/sleep", '#!/bin/bash\ncase "$1" in 60) exec /bin/sleep 0.02;; *) echo "$1" >> "$SLEEPS";; esac\n')
    env = {"PATH": f"{venv / 'bin'}:{os.environ['PATH']}", "HOME": str(tmp_path),
           "VENV": str(venv), "SLEEPS": str(tmp_path / "sleeps"), "SLURM_JOB_ID": "123",
           "RUN_DIR": str(tmp_path / "run"), "REPO": str(tmp_path / "repo"),
           "SLURM_DIR": str(tmp_path / "slurm"), "SLURM_RESTART_COUNT": "0",
           "CONTAINER_IMAGE": "fake.sqsh"}
    (tmp_path / "run/logs").mkdir(parents=True)
    (tmp_path / "slurm").mkdir()
    executable(venv / "bin/scontrol", '#!/bin/bash\nprintf "%s\\n" "$*" >> "$REQUEUE_LOG"\n')
    executable(venv / "bin/squeue", '#!/bin/bash\n[[ "$*" == *%T* ]] && echo RUNNING || echo 00:10:00\n')
    env["REQUEUE_LOG"] = str(tmp_path / "requeues")
    return env


def run_script(tmp_path, name, env):
    source = (SLURM / name).read_text()
    # Only replace deployment paths; all process/exit/retry logic is the real script.
    # /work first: pytest's tmp dir may itself be under /work (a later pass would rewrite it).
    source = source.replace('/work', str(tmp_path / "work"))
    source = source.replace('/fsx/carlos_patino/prj-eliza/slurm/common.sh', str(tmp_path / "common.sh"))
    target = tmp_path / name
    target.write_text(source)
    return subprocess.run(["bash", str(target), "run4"], env=env, text=True,
                          capture_output=True, timeout=4, check=False)


def test_restore_retries_before_serving_and_preserves_missing_archive_semantics(tmp_path):
    env = shell_environment(tmp_path)
    env.update(WORLD_SRC=str(tmp_path / "source"), CHATBOT_WS=str(tmp_path / "work/world/ws"),
               WORLD_PORT="7860", WORLD_CTL=str(tmp_path), B="hf://buckets/fake/run4",
               WS_SYNC_S="60", FINAL_SYNC_S="1", FINAL_SYNC_LONG_S="1", CANCEL_SYNC_S="1",
               COUNTER=str(tmp_path / "counter"))
    executable(tmp_path / "venv/bin/flock", '#!/bin/bash\nexit 0\n')
    executable(tmp_path / "source/snapshot.py", '''from pathlib import Path
import os, sys
if sys.argv[1] == 'restore':
    p = Path(os.environ['COUNTER'])
    n = int(p.read_text()) + 1 if p.exists() else 1
    p.write_text(str(n))
    raise SystemExit(1 if n < 5 else 0)
''')
    result = run_script(tmp_path, "world-inner.sh", env)
    assert (tmp_path / "counter").read_text() == "5", result.stderr
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "sleeps").read_text().splitlines() == ["30"] * 4


@pytest.mark.parametrize(("code", "restarts", "stable_s", "stopping", "requeue", "count", "sleeps"), [
    (37, 0, "600", False, True, 1, []), (37, 4, "600", False, True, 5, []),
    (37, 5, "600", False, True, 5, ["300"]),    # past the cap: back off, then requeue anyway
    (37, 5, "0", False, True, 0, []),           # ran >= WORLD_STABLE_S: the counter resets
    (0, 0, "600", False, False, 0, []), (37, 0, "600", True, False, 0, []),
])
def test_world_failure_counts_fast_crashes_and_never_gives_up(tmp_path, code, restarts, stable_s, stopping,
                                                              requeue, count, sleeps):
    env = shell_environment(tmp_path)
    env["WORLD_STABLE_S"] = stable_s
    env.update(RUN3_BUCKET="hf://buckets/fake/b", HF_TOKEN="fake-shared",
               CHATBOT_WORLD_TOKEN="fake-world", SLURM_RESTART_COUNT="9")  # Spot requeues do not count
    (tmp_path / "run/world").mkdir(parents=True)
    (tmp_path / "run/world/crash-requeues").write_text(f"{restarts}\n")
    (tmp_path / "common.sh").write_text('''log() { echo "$*"; }
die() { echo "$*" >&2; exit 2; }
require_var() { :; }
publish_host() { :; }
http_status() { echo 000; }
container_srun_args() { CTR_SRUN=(); }
''')
    executable(tmp_path / "venv/bin/srun", '#!/bin/bash\nexec "$@"\n')
    executable(tmp_path / "slurm/world-inner.sh",
               '#!/bin/bash\n' + ('kill -TERM "$PPID"\n' if stopping else '') + f'exit {code}\n')
    result = run_script(tmp_path, "world.sbatch", env)
    marker = tmp_path / "requeues"
    assert marker.exists() == requeue, result.stdout + result.stderr
    if requeue:
        assert marker.read_text().splitlines() == ["requeue 123"]
        assert result.returncode == 0
        assert (tmp_path / "run/world/crash-requeues").read_text().strip() == str(count)
    sleep_log = tmp_path / "sleeps"
    assert ([x for x in sleep_log.read_text().split() if int(x) >= 300] if sleep_log.exists() else []) == sleeps


@pytest.mark.parametrize("cycle", [None, 60.0])
async def test_busy_board_polling_rotates_once_per_cycle_on_a_busy_feed(monkeypatch, cycle):
    import flock.worker.comms_extended as module

    now = 0.0
    starts = []
    hub = CommsHub(swarm="run4")
    spec = CommsSpec(url="http://board", invite_env="INVITE", channel="general",
                     channel_id=12, extended=True, **({"poll_cycle_s": cycle} if cycle else {}))
    assert spec.poll_cycle_s == cycle  # unset: each identity every 5 minutes (300 s)
    hub.configure(spec, ["comms:read"])
    for i in range(12):
        name = f"agent-{i:04d}"
        hub.members[name] = Member(name, ident=MsgboardIdentity("http://board", "fake", name, i), cursor=0)

    async def sleep(delay):
        nonlocal now
        now += delay

    async def activity(*args, **kwargs):
        starts.append(now)
        if len(starts) == 48:
            hub.entries = []
        return {"messages": [], "next_cursor": 1}

    monkeypatch.setattr(module, "monotonic", lambda: now, raising=False)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(hub, "_board", lambda _: SimpleNamespace(activity=activity))
    await hub._poll_extended()
    assert len(starts) == 48
    spacing = (cycle or 300.0) / 12  # 25 s per worker by default: ~1 req/s for 25 workers
    assert all(b - a >= spacing for a, b in pairwise(starts))
    assert starts[-1] - starts[0] < 47 * spacing + 1


def test_world_backoff_doubles_to_the_cap(tmp_path):
    env = shell_environment(tmp_path)
    env.update(RUN3_BUCKET="hf://buckets/fake/b", HF_TOKEN="x", CHATBOT_WORLD_TOKEN="x", WORLD_STABLE_S="600")
    (tmp_path / "run/world").mkdir(parents=True)
    (tmp_path / "run/world/crash-requeues").write_text("5\n")
    (tmp_path / "common.sh").write_text('''log() { echo "$*"; }
die() { echo "$*" >&2; exit 2; }
require_var() { :; }
publish_host() { :; }
http_status() { echo 000; }
container_srun_args() { CTR_SRUN=(); }
''')
    executable(tmp_path / "venv/bin/srun", '#!/bin/bash\nexec "$@"\n')
    executable(tmp_path / "slurm/world-inner.sh", '#!/bin/bash\nexit 37\n')
    for _ in range(5):
        assert run_script(tmp_path, "world.sbatch", env).returncode == 0
    backoffs = [x for x in (tmp_path / "sleeps").read_text().split() if int(x) >= 300]  # not the 2 s manifest poll
    assert backoffs == ["300", "600", "1200", "1800", "1800"]


@pytest.mark.parametrize(("swarm_code", "waited", "calls"), [(200, False, ["pin", "run"]), (404, True, [])])
def test_controller_waits_for_the_world_only_on_the_first_launch(tmp_path, swarm_code, waited, calls):
    import shlex
    env = shell_environment(tmp_path)
    env.update(RUN3_BUCKET="hf://buckets/fake/b", RUN3_NAMESPACE="fake", OPENAI_BASE_URLS="http://fake/v1",
               CHATBOT_BOARD_URL="http://fake-board", CHATBOT_CHANNEL="general", MODEL_WORKER="fake",
               WORLD_WAIT_S="2", HOSTS_DIR=str(tmp_path / "run/hosts"), CLOCK=str(tmp_path / "clock"),
               CALLS=str(tmp_path / "calls"), SECRETS_FILE="s", REPO=str(tmp_path))
    (tmp_path / "run/hosts").mkdir()
    (tmp_path / "run/hosts/world.host").write_text("w\n")
    exp = tmp_path / "run/experiment"
    exp.mkdir()
    (exp / "render.py").write_text("import sys; open(sys.argv[sys.argv.index('--out') + 1], 'w').write('swarm: s\\n')")
    (exp / "waves.py").write_text("import os, sys; open(os.environ['CALLS'], 'a').write(sys.argv[1] + '\\n')")
    (tmp_path / "common.sh").write_text(f'''log() {{ echo "$*"; }}
die() {{ echo "$*" >&2; exit 2; }}
require_var() {{ :; }}
publish_host() {{ :; }}
http_status() {{ case $1 in */healthz) echo 200;; */swarms/*) echo {swarm_code};; *) echo 000;; esac; }}
''')
    (tmp_path / "venv/bin/python").unlink()
    executable(tmp_path / "venv/bin/python", f'#!/bin/bash\nexec {shlex.quote(sys.executable)} "$@"\n')
    for name in ("sbatch", "sacct"):
        executable(tmp_path / "venv/bin" / name, '#!/bin/bash\nexit 97\n')
    executable(tmp_path / "venv/bin/flock", '#!/bin/bash\necho "$1" >> "$CALLS"\n')
    executable(tmp_path / "venv/bin/flock-controller", '#!/bin/bash\n/bin/sleep 1\n')
    executable(tmp_path / "venv/bin/date", '#!/bin/bash\nn=$(cat "$CLOCK" 2>/dev/null || echo 0)\nn=$((n+1))\necho "$n" > "$CLOCK"\necho "$n"\n')
    result = run_script(tmp_path, "controller.sbatch", env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("world unavailable after 2s" in result.stdout) == waited
    assert not (tmp_path / "requeues").exists()  # never requeues itself over a missing world
    called = tmp_path / "calls"
    assert (called.read_text().split() if called.exists() else []) == calls


def test_role_preparation_plan_and_existing_checkout_protection(tmp_path):
    import shlex

    from .test_chatbot4 import git

    project = tmp_path / "project"
    source = project / "flock"
    source.mkdir(parents=True)
    git(source, "init", "-b", "main")
    (source / "marker").write_text("code")
    git(source, "add", ".")
    git(source, "-c", "user.name=test", "-c", "user.email=test@flock.invalid", "commit", "-m", "code")
    commit = git(source, "rev-parse", "HEAD")
    bin_dir = tmp_path / "bin"
    log = tmp_path / "uv.log"
    python_wrapper = '#!/bin/bash\nexec ' + shlex.quote(sys.executable) + ' "$@"\n'
    executable(bin_dir / "uv", f'''#!{sys.executable}
import os, sys
from pathlib import Path
with open(os.environ['UV_LOG'], 'a') as log: log.write(' '.join(sys.argv[1:])+'\\n')
if sys.argv[1] == 'venv':
    target = Path(sys.argv[-1]) / 'bin'
    target.mkdir(parents=True)
    (target / 'python').write_text({python_wrapper!r})
    (target / 'python').chmod(0o700)
    for name in ['flock', 'flock-controller', 'flock-worker', 'hf', 'pytest', 'ruff']:
        (target / name).write_text('#!/bin/sh\\nexit 0\\n')
        (target / name).chmod(0o700)
''')
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "UV_LOG": str(log)}
    cmd = ["bash", str(ROOT / "experiments/run4/prepare_roles.sh"), commit, "--project", str(project)]
    planned = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
    assert "Plan only" in planned.stdout and not log.exists()
    assert not (project / "flock-run4-ctl").exists()
    applied = subprocess.run([*cmd, "--apply"], env=env, text=True, capture_output=True, check=False)
    assert applied.returncode == 0, applied.stderr
    assert len(log.read_text().splitlines()) == 6
    assert all("--all-extras" in line for line in log.read_text().splitlines() if line.startswith("sync"))
    (project / "flock-run4-worker/marker").write_text("uncommitted")
    refused = subprocess.run([*cmd, "--apply"], env=env, text=True, capture_output=True, check=False)
    assert refused.returncode == 2 and "refusing" in refused.stderr
    assert len(log.read_text().splitlines()) == 6  # prechecks ran before touching any role


def test_preflight_is_read_only_and_does_not_echo_credentials(tmp_path):
    from .test_run4_experiment import module

    env = {key: "private-marker" for key in ["HF_TOKEN", "OPENAI_API_KEY", "CHATBOT_BOARD_INVITE", "CHATBOT_WORLD_TOKEN", "FLOCK_TOKEN"]}
    for role in ["CTL", "WORLD", "WORKER"]:
        repo, venv = tmp_path / role / "repo", tmp_path / role / "venv"
        (repo / "experiments/run4").mkdir(parents=True)
        (repo / "experiments/run4/spec.yaml").touch()
        for name in ["python", "flock", "hf", "pytest", "ruff"]:
            executable(venv / "bin" / name, '#!/bin/sh\nexit 0\n')
        env[role + "_REPO"], env[role + "_VENV"] = str(repo), str(venv)
    preflight = module("preflight")
    results = preflight.check(env, get=lambda *_: pytest.fail("offline preflight made a request"))
    assert all(check["ok"] for check in results)
    assert "private-marker" not in str(results)


@pytest.mark.parametrize(("restore_code", "attempts", "expected_code"), [(3, 1, 0), (1, 5, 2)])
def test_restore_missing_archive_or_exhausted_retries(tmp_path, restore_code, attempts, expected_code):
    env = shell_environment(tmp_path)
    env.update(WORLD_SRC=str(tmp_path / "source"), CHATBOT_WS=str(tmp_path / "work/world/ws"),
               WORLD_PORT="7860", WORLD_CTL=str(tmp_path), B="hf://buckets/fake/run4",
               WS_SYNC_S="60", FINAL_SYNC_S="1", FINAL_SYNC_LONG_S="1", CANCEL_SYNC_S="1",
               COUNTER=str(tmp_path / "counter"), SERVED=str(tmp_path / "served"))
    executable(tmp_path / "venv/bin/flock", '#!/bin/bash\ntouch "$SERVED"\n')
    executable(tmp_path / "source/snapshot.py", f'''from pathlib import Path
import os, sys
if sys.argv[1] == 'restore':
    p = Path(os.environ['COUNTER'])
    p.write_text(str(int(p.read_text()) + 1 if p.exists() else 1))
    raise SystemExit({restore_code})
''')
    result = run_script(tmp_path, "world-inner.sh", env)
    assert result.returncode == expected_code, result.stderr
    assert (tmp_path / "counter").read_text() == str(attempts)
    assert (tmp_path / "served").exists() == (restore_code == 3)


def test_shared_token_is_explicit_and_does_not_fall_back_to_cached_credentials(monkeypatch):
    from . import test_chatbot4  # installs the world module path

    assert test_chatbot4.Chatbot4World
    repository = importlib.import_module("repository")
    monkeypatch.setattr(repository, "HfApi", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setenv("HF_TOKEN", "shared-fixture")
    monkeypatch.delenv("CHATBOT_REPO_TOKEN", raising=False)
    assert repository.Hub("fake/dataset").api.token == "shared-fixture"
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(repository.ToolError, match="HF_TOKEN is required"):
        repository.Hub("fake/dataset")


def test_worker_paths_trim_slashes_and_accept_only_shared_hub_token(tmp_path):
    env = shell_environment(tmp_path)
    env.update(RUN="run4", LOG_TAG="worker")
    project = tmp_path / "project"
    secrets = project / "secrets/secrets.env"
    secrets.parent.mkdir(parents=True)
    secrets.write_text("HF_TOKEN=shared-fixture\n")
    secrets.chmod(0o600)
    run_env = project / "runs/run4/run.env"
    run_env.parent.mkdir(parents=True)
    run_env.write_text(f"WORKER_VENV={tmp_path}/venv/\nWORKER_REPO={tmp_path}/repo/\n")
    run_env.chmod(0o600)
    for binary in ["flock", "flock-controller", "flock-worker", "hf"]:
        executable(tmp_path / "venv/bin" / binary, '#!/bin/sh\nexit 0\n')
    common = tmp_path / "common.sh"
    common.write_text((SLURM / "common.sh").read_text().replace(
        "PRJ=/fsx/carlos_patino/prj-eliza", "PRJ=" + str(project)))
    result = subprocess.run(["bash", "-c", '. "$1"; printf "%s\\n%s\\n" "$VENV" "$REPO"', "test", str(common)],
                            env=env, text=True, capture_output=True, timeout=4, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-2:] == [str(tmp_path / "venv"), str(tmp_path / "repo")]
    assert "shared-fixture" not in result.stdout + result.stderr


def test_calibrated_capacity_threshold_and_invalid_values(monkeypatch):
    from .test_run4_experiment import module

    guard = module("load_guard")
    def measure(threshold):
        times = iter([0, 60, 120])
        completed = iter([0, 30, 60])
        monkeypatch.setenv("RUN4_MAX_REQUEST_SECONDS", str(threshold))
        return guard.capacity_ok(["fake"], sleep=lambda _: None, clock=lambda: next(times),
                                 get=lambda _: SimpleNamespace(raise_for_status=lambda: None, text=(
                                     "vllm:num_requests_running 10\nvllm:num_requests_waiting 0\n"
                                     f"vllm:request_success_total {next(completed)}\n")))
    assert not measure(15)[0]  # observed 20 s/request
    assert measure(25)[0]
    for invalid in [0, -1, "bad", "nan", "inf"]:
        ok, message = measure(invalid)
        assert not ok and "finite and positive" in message


@pytest.mark.parametrize("stop_signal", ["USR1", "TERM"])
def test_restore_backoff_is_interruptible_and_never_uploads_partial_state(tmp_path, stop_signal):
    env = shell_environment(tmp_path)
    env.update(WORLD_SRC=str(tmp_path / "source"), CHATBOT_WS=str(tmp_path / "work/world/ws"),
               WORLD_PORT="7860", WORLD_CTL=str(tmp_path), B="hf://buckets/fake/run4",
               WS_SYNC_S="60", FINAL_SYNC_S="1", FINAL_SYNC_LONG_S="1", CANCEL_SYNC_S="1",
               OPERATIONS=str(tmp_path / "operations"))
    executable(tmp_path / "source/snapshot.py", '''from pathlib import Path
import os, sys
with Path(os.environ['OPERATIONS']).open('a') as out: out.write(sys.argv[1]+'\\n')
raise SystemExit(1)
''')
    executable(tmp_path / "venv/bin/sleep", f'#!/bin/bash\nkill -{stop_signal} "$PPID"\nexec /bin/sleep 1\n')
    result = run_script(tmp_path, "world-inner.sh", env)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "operations").read_text().splitlines() == ["restore"]
    assert "flock world serve pid" not in result.stderr


def test_online_preflight_only_reads_metadata_and_sanitizes_failures(tmp_path):
    import httpx

    from .test_run4_experiment import module

    preflight = module("preflight")
    env = {"OPENAI_BASE_URLS": "http://model/v1", "MODEL_WORKER": "model-id",
           "CHATBOT_BOARD_URL": "http://board", "RUN3_BUCKET": "hf://buckets/fake/b", "RUN": "run4",
           "HF_TOKEN": "private-marker"}
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        if url.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "model-id"}]}, request=httpx.Request("GET", url))
        return httpx.Response(200, text="vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n",
                              request=httpx.Request("GET", url))
    class ReadOnlyApi:
        def repo_info(self, *args, **kwargs):
            assert kwargs == {"repo_type": "dataset"}
        def get_bucket_paths_info(self, bucket_id, paths):
            assert bucket_id == "fake/b" and paths == ["run4/world.tar.gz"]
            return []
    results = preflight.check(env, online=True, get=get, api=ReadOnlyApi())
    assert calls == ["http://model/v1/models", "http://model/metrics", "http://board/v1/guide"]
    assert all(row["ok"] for row in results if row["check"] in ["model_1", "metrics_1", "board", "hub_metadata"])
    def error(*args, **kwargs):
        raise httpx.ConnectError("private-marker")
    results = preflight.check(env, online=True, get=error, api=ReadOnlyApi())
    assert not any(row["ok"] for row in results if row["check"] in ["model_1", "metrics_1", "board"])
    assert "private-marker" not in str(results)
