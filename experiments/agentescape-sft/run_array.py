"""Run-local AgentEscape sweep. No edits to Legion or Hub task definitions."""

import fcntl
import json
import os
import secrets
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import yaml
from prepare_slurm import prepare

from legion.controller.slurm import scrubbed_env
from legion.harbor.spec import Spec, resource_secrets
from legion.specrender import read_env_file

ROOT = Path(__file__).resolve().parent
STUDY = json.loads((ROOT / "study.json").read_text())
ROWS = json.loads((ROOT / "manifest.json").read_text())
SMOKES = json.loads((ROOT / "smokes.json").read_text())
PY = sys.executable
os.umask(0o077)


class InfraError(Exception):
    pass


def save(p, v):
    p = Path(p)
    tmp = p.with_name(p.name + "." + str(os.getpid()) + ".tmp")
    tmp.write_text(json.dumps(v, indent=2) + "\n")
    tmp.replace(p)


def log(**kw):
    print(json.dumps({"at": time.time(), **kw}), flush=True)


def row_for(index):
    return SMOKES[index] if str(index).startswith("smoke") else ROWS[int(index)]


def directory(row, attempt):
    return ROOT / (row["prefix"] + f"-a{attempt}")


def jobs(r):
    raw = subprocess.check_output(["squeue", "--me", "-h", "-o", "%i|%j|%T"], text=True, timeout=30)
    return [
        (j, n, s)
        for line in raw.splitlines()
        for j, n, s in [line.split("|")]
        if n.startswith(r.name + "-") or n == "legion-ctl-" + r.name
    ]


def cleanup(r):
    for _ in range(18):
        live = jobs(r)
        if not live:
            return True
        for ids in (
            [j for j, n, s in live if n == "legion-ctl-" + r.name],
            [j for j, n, s in live if n != "legion-ctl-" + r.name],
        ):
            if ids:
                subprocess.run(["scancel", *ids], timeout=30, check=False)
        time.sleep(5)
    return not jobs(r)


def command(r, args, name, timeout):
    with (r / name).open("a") as f:
        return subprocess.run(
            [str(r / "legion-run"), "--run", str(r), *args],
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        ).returncode


def spec_for(row, attempt):
    r = directory(row, attempt)
    n = row["agents"]
    s = json.loads((ROOT / "spec-template.json").read_text())
    s.update(
        name=r.name,
        bucket=STUDY["bucket"],
        namespace=STUDY["bucket"].split("/")[0],
        model="openai/" + STUDY["model"],
    )
    s["inference"] = {
        "base_url": STUDY["router_url"],
        "api_key_env": "ROUTER_API_KEY",
        "max_working": n,
        "reasoning_effort": "off",
    }
    s["compute"].update(
        cpus=min(30, max(2, n)), mem=f"{min(30, max(2, n)) * 1900}M", time="3:00:00"
    )
    s["resources_dir"] = "resources"
    s["resources"]["start_barrier"] = {"agents": n}
    s["agents"]["count"] = n
    s["task"] = dict(
        repo="https://huggingface.co/datasets/" + STUDY["dataset"],
        revision=STUDY["revision"],
        dataset="agentescape-sft@1.0.0",
        selector=row["id"],
    )
    check = (
        'import json,urllib.request; s=json.load(urllib.request.urlopen("http://127.0.0.1:8000/state",timeout=3)); print(json.dumps({"success":s.get("won") is True and s.get("scenario_sha256")=='
        + repr(row["scenario_sha256"])
        + ',"evidence":s}))'
    )
    s["success_check"]["command"] = "python -c " + shlex.quote(check)
    s["workers"].update(
        agents_per_worker=n,
        max_working_per_worker=n,
        worker_lifetime="165m",
        turn_timeout="145m",
        drain_window="15m",
        max_extra=0,
    )
    s["limits"].update(
        run_timeout_seconds=10800, agent_timeout_seconds=9900, trial_hours=3, agent_max_turns=200
    )
    if row.get("startup_only"):
        s["stop_on_success"] = False
    if n == 1:
        s["loop"]["prompt"] = (
            "Start solving your room with escape observe. Solve independently and submit the victory answer. The controller stops the run after a verified win."
        )
    return s


def prepare_files(row, attempt):
    from task_splits import verify_task

    verify_task(Path(STUDY["task_snapshot"]) / row["task_path"], row)
    if row["split"] != "sft":
        raise ValueError("Only SFT scenarios are allowed")
    r = directory(row, attempt)
    r.mkdir(exist_ok=True)
    (r / "experiment").mkdir(exist_ok=True)
    save(r / "design.json", {**row, "attempt": attempt})
    (r / "experiment/spec.yaml").write_text(yaml.safe_dump(spec_for(row, attempt), sort_keys=False))
    team = (ROOT / "team.md").read_text()
    if row["agents"] == 1:
        start = team.index("Keep notes on initial values")
        team = (
            "Solve your escape-room puzzle independently. Run shell command `escape observe` using the bash tool; escape is not a model tool.\n\n"
            + team[start:]
        )
        team = team.replace(
            "A teammate's success is useful\nevidence but does not solve your copy.", ""
        )
    (r / "experiment/team.md").write_text(team)
    import shutil

    shutil.copy2(ROOT / "legion-run", r / "legion-run")
    shutil.copytree(ROOT / "resources", r / "experiment/resources", dirs_exist_ok=True)
    (r / "cluster.env").write_text(
        "SLURM_PARTITION=hopper-cpu\nWORKER_CPUS=2\nWORKER_MEM=3800M\nSLURM_MAX_MEM_PER_CPU_MB=2000\nLEGION_SLURM_MAX_CPUS=32\nLEGION_SLURM_MAX_MEM=64G\nCTL_PARTITION=hopper-cpu\nCTL_CPUS=2\nCTL_MEM=3800M\n"
        f"VENV={shlex.quote(sys.prefix)}\nREPO={shlex.quote(STUDY['repo'])}\n"
    )
    (r / "cluster.env").chmod(0o600)
    resource_secrets(Spec.load(r / "experiment/spec.yaml"))
    return r


def admission(r):
    while True:
        if datetime.fromisoformat(STUDY["router_end"]).timestamp() - time.time() < 10800:
            raise InfraError("Insufficient router lifetime; do not launch")
        with (ROOT / "admission.lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if len(list(ROOT.glob("*/reservation.json"))) < STUDY["concurrency"]:
                if (r / "reservation.json").exists():
                    raise InfraError("Duplicate attempt reservation")
                save(
                    r / "reservation.json",
                    {
                        "owner": os.environ.get("SLURM_ARRAY_JOB_ID", os.environ["SLURM_JOB_ID"])
                        + (
                            "_" + os.environ["SLURM_ARRAY_TASK_ID"]
                            if "SLURM_ARRAY_TASK_ID" in os.environ
                            else ""
                        ),
                        "at": time.time(),
                    },
                )
                return
        time.sleep(10)


def setup(r):
    key = Path(STUDY["key_file"]).read_text().strip()
    with httpx.Client(headers={"Authorization": "Bearer " + key}, trust_env=False, timeout=30) as c:
        resp = c.get(STUDY["router_url"] + "/models")
        resp.raise_for_status()
        assert any(
            m["id"] == STUDY["model"] and m["max_model_len"] == 65536 for m in resp.json()["data"]
        )
    e = read_env_file(STUDY["template_secrets"])
    e = {k: v for k, v in e.items() if k in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]}
    e.update(
        LEGION_TOKEN=secrets.token_urlsafe(32),
        RUN_BUCKET=STUDY["bucket"],
        ROUTER_API_KEY=key,
        ROUTER_URL=STUDY["router_url"],
    )
    for n in ["secrets.env", "run.env"]:
        p = r / n
        p.write_text("".join(k + "=" + shlex.quote(v) + "\n" for k, v in e.items()))
        p.chmod(0o600)
    if command(r, ["init", str(r)], "init.log", 1800):
        raise InfraError("Legion init failed")
    prepare(r, STUDY["network_helpers"])
    p = r / "slurm/worker.sbatch"
    s = p.read_text()
    old = '  printf \'unqualified-search-registries = ["docker.io"]\\n\' > "$CONTAINERS_REGISTRIES_CONF"'
    assert old in s
    s = s.replace(
        old, "  cp " + shlex.quote(str(ROOT / "registries.conf")) + ' "$CONTAINERS_REGISTRIES_CONF"'
    )
    s = s.replace(
        "--storage-driver overlay --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs",
        "--storage-driver vfs",
    )
    s = s.replace(
        '  "$VENV/bin/python" -m legion.harbor.podman_healthchecks &',
        '  (while true; do "$VENV/bin/python" -m legion.harbor.podman_healthchecks; sleep 2; done) &',
    )
    p.write_text(s)
    subprocess.run(["bash", "-n", str(p)], check=True)
    p = r / "slurm/controller.sbatch"
    p.write_text(p.read_text().replace("#SBATCH --time=60:00:00", "#SBATCH --time=3:00:00"))
    rc = command(r, ["launch", str(r / "experiment/spec.yaml")], "launch.log", 1800)
    if rc and "ReadTimeout" in (r / "launch.log").read_text():
        rc = command(r, ["launch", str(r / "experiment/spec.yaml")], "launch-retry.log", 1800)
    if rc:
        raise InfraError("Launch/preflight failed")
    save(r / "launched.json", {"at": time.time()})
    return e


def monitor(r, row, e):
    start = time.time()
    first = None
    bad = None
    seenjobs = set()
    capture = None
    starts = {}
    cursors = {}
    all_started_at = None
    reason = None
    success = None
    with httpx.Client(
        timeout=15, trust_env=False, headers={"Authorization": "Bearer " + e["LEGION_TOKEN"]}
    ) as c:
        while True:
            now = time.time()
            live = jobs(r)
            workers = [j for j, n, s in live if n.startswith(r.name + "-agent-w")]
            if set(workers) - seenjobs and seenjobs:
                raise InfraError(
                    "Worker replaced after model start" if first else "Worker replaced during setup"
                )
            seenjobs.update(workers)
            if capture is None or capture.poll() is not None:
                running = [
                    j for j, n, s in live if n.startswith(r.name + "-agent-w") and s == "RUNNING"
                ]
                if running:
                    capture = subprocess.Popen(
                        [
                            "srun",
                            "--jobid=" + running[0],
                            "--overlap",
                            "--ntasks=1",
                            "--cpus-per-task=1",
                            "--cpu-bind=none",
                            "--gpus=0",
                            PY,
                            str(ROOT / "capture.py"),
                            str(r),
                            str(row["agents"]),
                        ],
                        stdout=(r / "capture.log").open("a"),
                        stderr=subprocess.STDOUT,
                        env=scrubbed_env(),
                    )
            if now - start > 2700 and len(starts) < row["agents"]:
                raise InfraError("Not all agents started within45min")
            try:
                base = "http://" + (r / "hosts/controller.host").read_text().strip()
                resp = c.get(base + "/swarms/" + r.name)
                resp.raise_for_status()
                status = resp.json()
                save(r / "status.json", status)
                st = status["status"]
                success = st.get("success")
                resp = c.get(base + "/swarms/" + r.name + "/agents")
                resp.raise_for_status()
                agents = resp.json()["agents"]
                save(r / "agents.json", agents)
                for agent in agents:
                    name = agent["name"]
                    if name in starts:
                        continue
                    resp = c.get(
                        base + "/agents/" + name + "/events",
                        params={"after": cursors.get(name, 0), "limit": 500},
                    )
                    resp.raise_for_status()
                    page = resp.json()
                    cursors[name] = page["next"]
                    with (r / (name + "-events.jsonl")).open("a") as f:
                        for event in page["events"]:
                            f.write(json.dumps(event) + "\n")
                            detail = event.get("data", {}).get("data", {})
                            if detail.get(
                                "dropped_prompts"
                            ) and "Cannot create trial network" in detail.get("error", ""):
                                raise InfraError(detail["error"])
                            if event.get("type") == "turn_started" and event.get("data", {}).get(
                                "data", {}
                            ).get("trial"):
                                starts[name] = datetime.fromisoformat(event["ts"]).timestamp()
                if starts:
                    first = min(starts.values())
                    save(
                        r / "work-start.json",
                        {
                            "unix": first,
                            "agent_turn_starts": starts,
                            "clock_source": "Harbor turn_started before Pi request",
                        },
                    )
                if len(starts) == row["agents"] and all_started_at is None:
                    audits = list((r / "native").glob("*/agent/request-settings.jsonl"))
                    if not row.get("startup_only") or len(audits) == row["agents"]:
                        all_started_at = now
                save(
                    r / "progress.json",
                    {
                        "phase": "work" if first else "startup",
                        "at": now,
                        "started_agents": len(starts),
                        "elapsed": now - first if first else None,
                    },
                )
                # Detect installation failures without treating legitimate model/budget endings as infrastructure.
                for p in (r / "native").glob("*/result.json"):
                    try:
                        result = json.loads(p.read_text())
                    except (OSError, ValueError):
                        continue
                    exc = result.get("exception_type") or ""
                    if exc in [
                        "AgentSetupError",
                        "NetworkConnectionError",
                        "EnvironmentStartError",
                        "EnvironmentBuildError",
                        "ModelProviderError",
                    ]:
                        raise InfraError("Agent setup/environment failure: " + exc)
                if success:
                    save(r / "success.json", success)
                    if len(starts) < row["agents"]:
                        raise InfraError("Success before all configured agents started")
                    reason = "success"
                    break
                if row.get("startup_only") and all_started_at and now - all_started_at >= 60:
                    reason = "startup_smoke_pass"
                    break
                if first and now - first >= row.get("work_seconds", 3600):
                    reason = "time_limit"
                    break
                loop = st["types"]["agent"].get("loop") or {}
                if first and loop.get("state") == "done":
                    reason = "agents_finished"
                    break
                bad = None
            except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
                bad = bad or now
                log(run=r.name, error=type(exc).__name__)
                if now - bad > 120:
                    raise InfraError("Controller unavailable for120s") from exc
            time.sleep(5)
    save(
        r / "model-finished.json",
        {
            "at": time.time(),
            "first_model_turn": first,
            "reason": reason,
            "startup_skew_s": max(starts.values()) - min(starts.values()) if starts else None,
            "started_agents": len(starts),
        },
    )
    rc = command(
        r, ["stop", r.name, "--interrupt", "--release", "--reason", reason], "release.log", 900
    )
    if rc:
        raise InfraError("Release failed")
    if capture:
        try:
            capture.wait(timeout=30)
        except subprocess.TimeoutExpired:
            capture.terminate()
    return reason


def summarize(r, row, status, reason):
    native = []
    for p in (r / "native").glob("*/result.json"):
        try:
            x = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        v = (x.get("verifier_result") or {}).get("rewards") or {}
        native.append(
            {"trial": p.parent.name, "rewards": v, "exception_type": x.get("exception_type")}
        )
    success = (
        json.loads((r / "success.json").read_text()) if (r / "success.json").exists() else None
    )
    started = (
        json.loads((r / "work-start.json").read_text()) if (r / "work-start.json").exists() else {}
    )
    proven = []
    for candidate in native:
        evidence_path = r / "native" / candidate["trial"] / "artifacts/evidence/final.json"

        try:
            evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
        except (OSError, ValueError):
            evidence = {}
        if (
            candidate["rewards"].get("reward") == 1
            and evidence.get("won") is True
            and evidence.get("scenario_sha256") == row["scenario_sha256"]
        ):
            proven.append(candidate["trial"])
    verified = bool(proven)
    if success:
        verified = (
            success.get("trial_id") in proven
            and success.get("evidence", {}).get("scenario_sha256") == row["scenario_sha256"]
        )
        if verified:
            success["verification"] = "passed"
            success["verification_source"] = "captured native reward and final evidence"
            save(r / "success.json", success)
    if verified and not success and status == "finished":
        reason = "native_success_without_controller_timestamp"
    if status == "finished" and len(native) < row["agents"]:
        status = "infra_failed"
        reason = "Missing native results"
    if status == "finished" and reason == "success" and not verified:
        status = "infra_failed"
        reason = "Success not natively verified"
    # Pi completed response records give usage at timestamps, including interrupted-turn partial histories.
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    before = dict(usage)
    count = 0
    agents_with_usage = set()
    boundary = (
        datetime.fromisoformat(success["accepted_at"]).timestamp() if verified and success else None
    )
    for p in (r / "native").glob("*/agent/pi.txt"):
        for line in p.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            m = event.get("message", {})
            if (
                event.get("type") != "message_end"
                or m.get("role") != "assistant"
                or not m.get("usage")
            ):
                continue
            u = m["usage"]
            ts = m.get("timestamp")
            count += 1
            agents_with_usage.add(p.parent.parent.name)
            for key, alias in [
                ("input", "input"),
                ("output", "output"),
                ("cache_read", "cacheRead"),
                ("cache_write", "cacheWrite"),
            ]:
                value = u.get(alias, 0)
                usage[key] += value
                if boundary and isinstance(ts, (int, float)) and ts / 1000 <= boundary:
                    before[key] += value
    for u in (usage, before):
        u["total"] = sum(u.values())
    result = {
        **{k: row[k] for k in ["difficulty", "scenario_index", "agents", "index"]},
        "run": r.name,
        "status": status,
        "reason": reason,
        "verified_success": verified,
        "native_results": len(native),
        "trials": native,
        "time_to_success_s": boundary - started["unix"] if boundary and started else None,
        "recorded_tokens": usage,
        "recorded_tokens_before_success": before if boundary else None,
        "agents_with_usage": len(agents_with_usage),
        "responses_with_usage": count,
        "accounting_note": "Completed response usage only; in-flight generation may be unaccounted. End totals include draining. Before-success tokens use assistant response timestamp.",
    }
    if (r / "board.db").exists():
        with sqlite3.connect("file:" + str(r / "board.db") + "?mode=ro", uri=True) as board:
            messages, senders = board.execute(
                "select count(*),count(distinct sender_id) from messages"
            ).fetchone()
        result.update(board_messages=messages, board_senders=senders)
    save(r / "summary.json", result)
    return result


def run_one(row, attempt):
    r = prepare_files(row, attempt)
    if (r / "completion.json").exists():
        return json.loads((r / "completion.json").read_text())
    admission(r)
    save(r / "owner.json", json.loads((r / "reservation.json").read_text()))
    status = "infra_failed"
    reason = None

    def interrupt(sig, frame):
        raise InfraError("Scheduler interrupted attempt")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupt)
    try:
        e = setup(r)
        reason = monitor(r, row, e)
        status = "finished"
    except Exception as exc:  # noqa: BLE001 -- record external job failure before cleanup
        reason = type(exc).__name__ + ": " + str(exc)
        log(run=r.name, error=reason)
    finally:
        clean = cleanup(r)
        result = summarize(r, row, status, reason)
        result["cleanup_complete"] = clean
        save(r / "completion.json", result)
        if clean:
            (r / "reservation.json").unlink(missing_ok=True)
    return result


def main(index):
    row = row_for(index)
    for attempt in range(int(os.environ.get("ATTEMPT_START", "0")), 2):
        result = run_one(row, attempt)
        if result["status"] == "finished" or not result["cleanup_complete"]:
            break
    log(done=index, status=result["status"])


if __name__ == "__main__":
    main(sys.argv[1])
