"""Run one CIB dimension through `legion launch`, case by case, for the parity check against
the dataset's published results (until `legion sweep` exists).

Every solo case runs first; then the three team seeds of each scenario whose solo passed, as
the benchmark's gate does. Each case is one local run of spec.yaml (CASE set) in its own folder,
`<output>/cases/<task id>/`, with at most `--concurrency` at once. A case with a verifier result
backed by a healthy host run is kept. An unavailable attempt is moved aside as
`<id>.failed-N` on the next invocation once its controller has stopped. The driver writes the
dataset scorer's report over admitted results.

Only manifests without aliases and reader grading (Sharing: `manifest.json`) are supported.

    uv run --with ../cib-harbor/adapter python experiments/cib/parity.py --dataset ../cib-harbor \
        --output results/cib/parity-sharing --concurrency 2
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from legion.current import read_current
from legion.harbor.spec import Spec

SPEC = Path(__file__).resolve().with_name("spec.yaml")
POLL_S = 30
WAIT_S = 75 * 60
RUNNER = "legion-launch"


def read_json(path: Path):
    return json.loads(path.read_text())


def completed_trial(case: Path) -> Path | None:
    """Find the Harbor Trial backed by the completed Legion host turn."""
    hosts = list(case.glob(".legion/*/run.json"))
    if len(hosts) != 1:
        return None
    host = read_json(hosts[0])
    run = hosts[0].parent
    agents = host.get("agents", [])
    if host.get("stop_reason") != "idle" or not host.get("ended_at") or len(agents) != 1:
        return None
    [agent] = agents
    workers = [w for w in host.get("workers", []) if w.get("id") == agent.get("worker")]
    if len(workers) != 1 or workers[0].get("crashes") or workers[0].get("last_error"):
        return None
    if agent.get("turns_completed") != 1 or agent.get("state") != "idle":
        return None
    events = run / "events" / f"{agent['name']}.jsonl"
    if not events.is_file():
        return None
    observed = [json.loads(line) for line in events.read_text().splitlines()]
    if any(e.get("type") in {"turn_interrupted", "error"} for e in observed):
        return None
    turns = [e for e in observed if e.get("type") == "turn_ended"]
    if len(turns) != 1:
        return None
    status = (turns[0].get("data") or {}).get("data") or {}
    if status.get("status") != "completed" or status.get("error"):
        return None
    trials = list((run / "agents" / "_harbor" / agent["name"] / "trials").glob("*/result.json"))
    return trials[0] if len(trials) == 1 else None


def result(case: Path, identity: dict, gateway: dict) -> Path | None:
    """Admit only a verifier envelope backed by a completed host turn and Harbor Trial."""
    if (trial_path := completed_trial(case)) is None:
        return None
    trial = read_json(trial_path)
    path = trial_path.parent / "verifier" / "result.json"
    if not path.is_file() or not trial.get("finished_at") or trial.get("exception_info"):
        return None
    envelope = read_json(path)
    if envelope.get("identity") != identity or envelope.get("fixture") is not False:
        raise ValueError(f"{path}: unexpected benchmark identity or fixture")
    metadata = (envelope.get("configuration") or {}).get("agent") or {}
    if metadata.get("runner") != RUNNER or metadata.get("model_gateway") != gateway:
        raise ValueError(f"{path}: gateway configuration differs from this run")
    if type(envelope.get("reward")) not in (int, float) or (
            (trial.get("verifier_result") or {}).get("rewards") or {}).get("reward") != envelope["reward"]:
        return None
    if envelope.get("status") == "finished":
        return path
    if envelope.get("status") == "abandoned" and envelope.get("terminal_reason") not in {
            "unanswered_request_timeout", "episode_timeout"}:
        evidence = envelope.get("partial_evidence") or {}
        if not (evidence.get("observations") and "delivered_request_ids" in evidence
                and not evidence["delivered_request_ids"] and not evidence.get("actions")):
            return path  # the benchmark admits abandonment after a healthy host turn
    return None


def wait_current(case: Path) -> bool:
    """Wait for grading, leaving a still-live case intact when the wait expires."""
    current = case / ".legion" / "current"
    host = case / "host"
    if current.exists() and (not host.is_file() or host.read_text().strip() != socket.gethostname()):
        return False  # a local PID cannot establish liveness on another node
    deadline = time.monotonic() + WAIT_S
    while read_current(current) is not None:
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_S)
    if current.exists():  # failed parsing is not proof of a dead controller
        try:
            pid = read_json(current)["pid"]
            if not isinstance(pid, int) or pid <= 0:
                return False
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except (KeyError, OSError, TypeError, ValueError):
            pass
        return False
    return True


def run_case(output: Path, task: dict, env: dict, gateway: dict, lock_fd: int) -> Path | None:
    task_id, identity = task["task_id"], task["identity"]
    case = output / "cases" / task_id
    if case.exists():
        if not wait_current(case):
            print(f"{task_id}: controller still live or unverified; kept {case}", flush=True)
            return None
        if (found := result(case, identity, gateway)) is not None:
            return found
        if (case / "launch-uncertain").exists() and not any(
                read_json(path).get("ended_at") for path in case.glob(".legion/*/run.json")):
            print(f"{task_id}: interrupted launch has no terminal evidence; kept {case}", flush=True)
            return None
        # An earlier attempt without an admissible result is kept for the record.
        n = 1
        while case.with_name(f"{task_id}.failed-{n}").exists():
            n += 1
        case.rename(case.with_name(f"{task_id}.failed-{n}"))
    case.mkdir(parents=True)
    (case / "host").write_text(socket.gethostname() + "\n")
    env = {**env, "CASE": task_id,
           "RUN_NAME": "cib-" + hashlib.sha256(task_id.encode()).hexdigest()[:8]}
    launch = case / "launch-uncertain"
    launch.touch()
    try:
        with (case / "launch.out").open("w") as out:
            code = subprocess.run([sys.executable, "-m", "legion", "launch", str(SPEC)], cwd=case,
                                  env=env, stdout=out, stderr=subprocess.STDOUT, check=False,
                                  timeout=180, pass_fds=(lock_fd,)).returncode
    except subprocess.TimeoutExpired:  # the detached controller may still be starting
        print(f"{task_id}: launcher timed out; kept {case}", flush=True)
        return None
    launch.unlink()
    found = result(case, identity, gateway) if code == 0 and wait_current(case) else None
    print(f"{task_id}: {'graded' if found else f'unavailable (see {case})'}", flush=True)
    return found


def solo_passed(path: Path | None, manifest: dict) -> bool:
    # These helpers come from the pinned dataset selected by main().
    from collab_harbor.takeover_verdicts import effective_record
    from collab_intel.inspect_eval.scoring import item_result

    if path is None or (run := read_json(path)).get("status") != "finished":
        return False
    return item_result([effective_record(run, manifest)], [], [],
                       baseline_optional=True)["gated"] is True


def configuration(dataset: Path, manifest_path: Path,
                  single_case: str | None) -> tuple[dict, dict, str]:
    """Validate and record the gateway settings without reading credentials into the record."""
    from collab_harbor.model_gateway import request_defaults, routing_headers, validate_upstream

    env = dict(os.environ)
    # Only a local, case-owned controller may be used for parity runs.
    for name in ("LEGION_URL", "LEGION_RUN_DIR", "LEGION_CURRENT"):
        env.pop(name, None)
    spec = Spec.load(SPEC, env)
    env_spec = spec.workers.env
    gateway = {
        "upstream": validate_upstream(env_spec["CI_MODEL_UPSTREAM"]).geturl(),
        "routing_headers": routing_headers(env_spec["CI_MODEL_HEADERS"]),
        "request_fields": request_defaults(env_spec["CI_MODEL_BODY"]),
        "private_readout_api": env_spec["CI_PRIVATE_READOUT_API"],
    }
    env["CIB_GATEWAY_CONFIGURATION"] = json.dumps(gateway, sort_keys=True, allow_nan=False)
    env_spec = Spec.load(SPEC, env).workers.env
    revision = subprocess.check_output(["git", "-C", str(dataset), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != spec.task.revision:
        raise ValueError(f"check out the dataset at the spec's pinned revision {spec.task.revision}")
    details = json.dumps({
        "revision": revision,
        "manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "spec": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        "settings": env_spec,
        "backend": spec.environment.backend,
        "model_key_env": spec.inference.api_key_env,
        "single_case": single_case,
    }, sort_keys=True, allow_nan=False)
    return env, gateway, hashlib.sha256(details.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True,
                        help="a clone of the dataset at the spec's task revision")
    parser.add_argument("--manifest", default="manifest.json", help="relative to --dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--case", help="run and score one task from the manifest")
    args = parser.parse_args()
    manifest_path = args.dataset / args.manifest
    manifest = read_json(manifest_path)
    tasks = manifest["tasks"]
    if manifest.get("grading_protocol") or any("canonical_task_id" in t for t in tasks):
        parser.error(f"{args.manifest}: aliases and reader grading need the dataset's own sweep")
    if any(Path(t["task_id"]).name != t["task_id"] or t["task_id"] in ("", ".", "..") for t in tasks):
        parser.error("the manifest contains an invalid task id")
    solos = [t for t in tasks if t["identity"]["variant"] == "solo"]
    selected = [t for t in tasks if t["task_id"] == args.case] if args.case else []
    if not solos or (args.case and len(selected) != 1):
        parser.error("the manifest must contain solos and the selected case")
    from collab_harbor.compact_runtime import ensure_upstream
    from collab_harbor.legion_results import load_runs
    from collab_harbor.scoring import IncompleteRunsError, score_runs

    tests = args.dataset.resolve() / solos[0]["path"] / "tests"
    upstream = tests if (tests / "src/collab_intel").is_dir() else ensure_upstream(tests)
    sys.path.insert(0, str(upstream / "src"))
    env, gateway, fingerprint = configuration(args.dataset, manifest_path, args.case)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"{output}: another driver or launcher is operating on this directory")
        receipt = output / "configuration.sha256"
        if receipt.exists():
            if receipt.read_text().strip() != fingerprint:
                parser.error(f"{output}: configuration changed; use a fresh output directory")
        elif (output / "cases").exists():
            parser.error(f"{output}: existing cases have no configuration fingerprint")
        else:
            receipt.write_text(fingerprint + "\n")
        with ThreadPoolExecutor(max(1, args.concurrency)) as pool:
            def run(task):
                return run_case(output, task, env, gateway, lock.fileno())

            first = selected or solos
            results = list(pool.map(run, first))
            if not selected:
                passed = {t["identity"]["scenario_id"] for t, path in zip(solos, results, strict=True)
                          if solo_passed(path, manifest)}
                seeds = [t for t in tasks if t["identity"]["variant"] in {"manipulation", "control"}
                         and t["identity"]["scenario_id"] in passed]
                results += list(pool.map(run, seeds))
        runs = [path for path in results if path is not None]
        (output / "runs.txt").write_text("".join(f"{path}\n" for path in runs))
        print(f"{len(runs)}/{len(results)} cases admitted for scoring")
        if not runs:
            return 1
        try:
            report = score_runs(load_runs(runs), manifest,
                                allow_incomplete=bool(selected) or len(runs) != len(results))
        except IncompleteRunsError as exc:
            report = exc.report
            incomplete = True
        else:
            incomplete = len(runs) != len(results)
        (output / "scores.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return int(incomplete)


if __name__ == "__main__":
    raise SystemExit(main())
