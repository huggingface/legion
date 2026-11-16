"""The s3-aggregation scaling sweep on the HF cluster, one command for the whole grid:

    uv run experiments/s3-aggregation/sweep/sweep.py --urls http://<vllm node>:8000/v1 \\
        [--problems 1-32] [--agents 1,2,4,8,16] [--parallel 8] [--root .flock/s3-sweep] \\
        [--bucket lewtun/s3-aggregation-sweep] [--dataset HuggingFaceSwarms/s3-aggregation-results]

Each (problem, N) team is its own flock run (`compute: slurm`, podman agents on hopper-cpu, a
local board on its one worker), from team.py's cluster spec with the sweep's model on `urls`
(the same board rules, loop and limits at every N). Per team: `flock init` (a worker sized to
N), `flock launch`, wait until its loop is done and its agents are idle, `flock stop --release`
(each Trial is graded as the worker drains; the results land in the run directory), then
experiments/s3/metrics.py's row plus the wall time goes to `<root>/rows.jsonl`. Until #121
is fixed, a team must run on one worker incarnation (a relaunch or a preemption overwrites
results): one that didn't runs again as `<task>-rK` (its row: `reruns`). A team that
has a row is skipped, so the same command resumes. At the end plot.py writes the figure, CSV
and table, and with --dataset each run directory (spec, results, logs; never run.env) goes to
that private dataset under `<prefix>/<task>/`.

Needs HF_TOKEN (or `hf auth login`): the workers clone the private task dataset and sync the
private state bucket with it. The agents get no Hub token; the vLLM server takes no key.
"""

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import yaml
from huggingface_hub import HfApi, get_token

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODEL = "Qwen/Qwen3.5-27B"
SECRET = re.compile(r"hf_[A-Za-z0-9]{30,}")
lock, init_lock = threading.Lock(), threading.Lock()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


team = load("team", HERE.parent / "team.py")
metrics = load("metrics", REPO / "experiments" / "s3" / "metrics.py")


def say(*parts) -> None:
    with lock:
        print(datetime.now(UTC).strftime("%H:%M:%S"), *parts, flush=True)


def numbers(text: str) -> list[int]:
    """`1-32` or `1,3,7` -> the numbers."""
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def flock(run_dir: Path, *args: str, timeout: float = 3600) -> subprocess.CompletedProcess:
    env = {**os.environ, "FLOCK_RUN_DIR": str(run_dir)}
    env.pop("VIRTUAL_ENV", None)
    return subprocess.run([str(Path(sys.executable).with_name("flock")), *args], cwd=REPO,
                          env=env, capture_output=True, text=True, timeout=timeout, check=False)


def write_experiment(src: Path, problem: int, n: int, name: str, urls: list[str],
                     model: str) -> None:
    """team.py's cluster spec and role files, with the sweep's model on its servers, as swarm
    `name`."""
    team.main(["--problem", str(problem), "--agents", str(n), "--where", "cluster",
               "--out", str(src)])
    spec = yaml.safe_load((src / "spec.yaml").read_text())
    spec["name"] = name
    spec["model"] = model
    spec["inference"] = {"urls": urls}  # no key: the servers take none
    (src / "spec.yaml").write_text(
        f"# {spec['task']['selector']} for the scaling sweep (sweep.py): team.py's cluster spec "
        f"with {model} on the sweep's servers.\n" + yaml.safe_dump(spec, sort_keys=False))


def profile(root: Path, n: int) -> Path:
    """A worker sized to the team: a CPU per agent (at least 4), 1900 MB per CPU (hopper-cpu
    allows 2000)."""
    cpus = max(4, n)
    path = root / "profiles" / f"n{n:02d}.env"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"SLURM_PARTITION=hopper-cpu\nWORKER_CPUS={cpus}\nWORKER_MEM={cpus * 1900}M\n"
                    f"SLURM_MAX_MEM_PER_CPU_MB=2000\nVENV={REPO / '.venv'}\n")
    return path


def wait_done(run_dir: Path, task: str, n: int, timeout: float) -> str:
    """Poll `flock status` until the loop is done and every agent is idle: 'done', or why not."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = flock(run_dir, "status", task, timeout=120).stdout
        if re.search(r"crashloop|launch_refused", text):
            return "worker failed: " + " ".join(text.split())[:300]
        if re.search(r"^loop agent: done", text, re.MULTILINE) and re.search(
                rf"^agent .*\bidle:{n}\b", text, re.MULTILINE):
            return "done"
        time.sleep(20)
    return f"not done after {timeout:.0f} s"


def attempt(args, root: Path, secrets: Path, problem: int, n: int, name: str) -> dict | None:
    """One run of a team as swarm `name`: its row, or None when it could not start."""
    run_dir, src = root / "runs" / name, root / "src" / name
    if run_dir.exists():
        if any((run_dir / "launch").glob("*.env")):  # a worker ran: look before running again
            say(name, "has a run directory but no row: remove", run_dir, "to run it again")
            return None
        shutil.rmtree(run_dir)  # init failed before: nothing ran
    started = time.time()
    write_experiment(src, problem, n, name, args.urls, args.model)
    log = root / "logs" / f"{name}-init.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for tries in range(3):
        with init_lock:  # one at a time: concurrent inits race on the bucket's write check
            init = flock(run_dir, "init", str(run_dir), "--from", str(src), "--profile",
                         str(profile(root, n)), "--secrets", str(secrets))
        log.write_text(init.stdout + init.stderr)
        if init.returncode == 0:
            break
        time.sleep(10 * (tries + 1))
    else:
        failed = [x for x in init.stdout.splitlines() if x.startswith("FAIL")]
        say(name, "init failed 3 times:", failed or init.stdout[-300:], "(log:", log, ")")
        return None
    say(name, "launching")
    launch = flock(run_dir, "launch", str(run_dir / "experiment"))
    if launch.returncode != 0:
        say(name, "launch failed:", launch.stdout[-600:], launch.stderr[-600:])
    why = wait_done(run_dir, name, n, args.team_timeout) if launch.returncode == 0 else "launch"
    say(name, "loop:", why, "- releasing")
    release = flock(run_dir, "stop", name, "--release")
    ended = time.time()
    return metrics.row(run_dir) | {
        "problem": problem, "wall_s": round(ended - started, 1), "outcome": why,
        "released": release.returncode == 0,
        # each launch of the worker slot writes one env file: more than one is a relaunch or a
        # preemption, whose results a later incarnation overwrites (#121)
        "incarnations": len(list((run_dir / "launch").glob("*.env"))),
        "started_at": datetime.fromtimestamp(started, UTC).isoformat()}


def run_team(args, root: Path, secrets: Path, problem: int, n: int) -> dict | None:
    """A team, run again (as `<task>-rK`) until it runs on one worker incarnation (#121)."""
    task = f"agg-p{problem:02d}-n{n:02d}"
    for reruns in range(args.max_reruns + 1):
        row = attempt(args, root, secrets, problem, n, task if reruns == 0 else f"{task}-r{reruns}")
        if row is None:
            return None
        row["reruns"] = reruns
        if row["incarnations"] == 1:
            break
        say(task, f"ran on {row['incarnations']} worker incarnations: running it again")
        with lock, (root / "rows-discarded.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
    with lock, (root / "rows.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    say(task, f"success={row['success']} rewards={list(row['rewards'].values())} "
        f"posts={row['posts']} wall={row['wall_s']} s reruns={row['reruns']} "
        f"incarnations={row['incarnations']} missing={row['missing']}")
    return row


def upload(root: Path, dataset: str, prefix: str) -> None:
    """Each run directory's spec, results and logs, and the rows, to a private dataset; never
    run.env, and nothing that holds a Hub token."""
    api = HfApi()
    api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
    if not api.dataset_info(dataset).private:
        raise SystemExit(f"{dataset} is public: refusing to upload")
    allow = ["*.jsonl", "*.csv", "*.md", "*.png", "runs/*/experiment/**", "runs/*/results/**",
             "runs/*/logs/**"]
    leaks = [str(p) for p in (root / "runs").rglob("*") if p.is_file()
             and not p.name.endswith(".env") and SECRET.search(p.read_text(errors="ignore"))]
    if leaks:
        raise SystemExit(f"refusing to upload: Hub tokens in {leaks[:5]}")
    api.upload_folder(repo_id=dataset, repo_type="dataset", folder_path=root,
                      path_in_repo=prefix, allow_patterns=allow,
                      commit_message=f"s3-aggregation sweep: {prefix}")
    say("uploaded", root, "to", f"{dataset}/{prefix}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--urls", required=True, help="the model servers, comma-separated")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--problems", default="1-32")
    parser.add_argument("--agents", default="1,2,4,8,16")
    parser.add_argument("--parallel", type=int, default=8, help="teams at once")
    parser.add_argument("--team-timeout", type=float, default=2400, help="s until a release")
    parser.add_argument("--max-reruns", type=int, default=2,
                        help="runs again of a team that ran on more than one worker (#121)")
    parser.add_argument("--root", type=Path, default=REPO / ".flock" / "s3-sweep")
    parser.add_argument("--bucket", default="lewtun/s3-aggregation-sweep",
                        help="private bucket for the runs' state (a new one per root: a "
                             "swarm's name is used once in a bucket)")
    parser.add_argument("--namespace", default="lewtun")
    parser.add_argument("--dataset", help="private dataset for the run directories")
    parser.add_argument("--prefix", default="grid", help="folder in the dataset")
    args = parser.parse_args(argv)
    args.urls = args.urls.split(",")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    api.create_bucket(args.bucket, private=True, exist_ok=True)
    if not api.bucket_info(args.bucket).private:
        raise SystemExit(f"bucket {args.bucket} is public: refusing")
    token = os.environ.get("HF_TOKEN") or get_token()
    secrets = root / "secrets.env"
    old = os.umask(0o077)
    secrets.write_text(f"HF_TOKEN={token}\nNAMESPACE={args.namespace}\n"
                       f"BUCKET=hf://buckets/{args.bucket}\n")
    os.umask(old)
    rows_file = root / "rows.jsonl"
    done = {(r["problem"], r["agents"]) for r in map(json.loads, rows_file.read_text().splitlines())
            } if rows_file.exists() else set()
    teams = [(p, n) for p in numbers(args.problems) for n in numbers(args.agents)
             if (p, n) not in done]
    say(f"{len(teams)} teams to run ({len(done)} done), {args.parallel} at once, on {args.urls}")
    try:
        with ThreadPoolExecutor(args.parallel) as pool:
            list(pool.map(lambda t: run_team(args, root, secrets, *t), teams))
    finally:
        secrets.unlink(missing_ok=True)
    subprocess.run(["uv", "run", str(HERE / "plot.py"), str(root)], check=False)  # matplotlib
    if args.dataset:
        upload(root, args.dataset, args.prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
