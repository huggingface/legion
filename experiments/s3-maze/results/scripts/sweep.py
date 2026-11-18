"""The s3-maze scaling sweep on the HF cluster, one command per model for the whole grid, run
from Flock's checkout (its environment, and its `.venv` for the workers):

    uv run python <this dataset>/flock/sweep/sweep.py --urls http://<vllm node>:8000/v1 \\
        --model Qwen/Qwen3.5-27B --tag q27- --revision <sha> [--trees 1-32] \\
        [--agents 1,2,4,8,16] [--parallel 20] [--max-agents 64] [--root .flock/s3-maze-sweep/q27]

s3-aggregation's flock/sweep/sweep.py for the maze. Each (tree, N) team is its own flock run
(`compute: slurm`, podman agents on hopper-cpu, a local board on its one worker), from team.py's
cluster spec with the sweep's model on `urls`: the same board rules, loop, limits and move
budget at every N, and `stop_on_success`. The worker gets a CPU per agent, a quarter of one per
maze service and two more (at least 4), at 1900 MB per CPU. Per team: `flock init`,
`flock launch`, then wait until the first agent's exit stops the run (and every Trial is
graded) or the loop is done and the agents are idle; `flock stop --release` (which grades the
Trials of a run nobody won, and ends the controller in either case); then eval/metrics.py's row
plus the wall time and the tree's exit branch go to `<root>/rows.jsonl`.

A run counts only when its worker started within WAIT_LIMIT of its launch (the loop's time
counts from the launch), on one worker incarnation (flock#121), with its controller kept and
nothing missing (every agent's reward and move log, the board export). Anything else is the
infrastructure's, so the team runs again as `<task>-rK` and the attempt goes to
`rows-discarded.jsonl`; what the model does (no move, no post, no exit) is a result. A team with
a row is skipped, so the same command resumes; a team a stopped driver launched is adopted and
keeps its own start time for its time cap.

At most `--parallel` teams and `--max-agents` agents run at once (the agents of the teams in
flight; `<root>/max_agents`, when present, overrides it while the driver runs): one GPU's KV
cache holds a few dozen agents' contexts of a large model, and an overloaded server would make
the loop's time limit, not the move budget, decide the result.

Needs HF_TOKEN (or `hf auth login`): the workers clone the private task dataset and sync the
private state bucket with it. The agents get no Hub token; the vLLM servers take no key.
"""

import argparse
import importlib.util
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from huggingface_hub import HfApi, get_token

from flock.models import Inference

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parents[1]
REPO = Path.cwd()  # Flock's checkout
SECRET = re.compile(r"hf_[A-Za-z0-9]{30,}")
lock, init_lock = threading.Lock(), threading.Semaphore(4)
LOST, GONE = "controller lost", "no controller left"
WAIT_LIMIT = 120  # s: a worker that started later than this after the launch doesn't count
MB_PER_CPU = 1900
# runs started in this window (`--overloaded START/END`, ISO times) ran on a model server whose
# KV cache was thrashing (too many agents for its GPU): the server, not the model, decided them
OVERLOADED: tuple[datetime, datetime] | None = None
# the servers' health log (`--health`: one JSON line per server and minute, with its prefix-cache
# hit rate): a run in which an agent's server thrashed this many minutes doesn't count
HEALTH: Path | None = None
THRASH_LIMIT = 5


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


team = load("team", HERE.parent / "team.py")
metrics = load("metrics", DATASET_DIR / "eval" / "metrics.py")


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


CPUS_PER_AGENT = 1  # --cpus-per-agent; memory follows at MB_PER_CPU
MEMORY_LADDER: dict | None = None  # --memory-ladder: the workers' rotate/drain/preempt fractions


def worker_cpus(n: int) -> int:
    """CPUS_PER_AGENT per agent's container, a quarter of one per maze service, two for the
    worker, podman and the image builds."""
    return max(4, CPUS_PER_AGENT * n + 2 + (n + 3) // 4)


def exit_branch(tree: int) -> int:
    """Which of the start room's doors (1, 2 or 3, in the listed order) leads to the exit."""
    layout = json.loads((DATASET_DIR / "tasks" / f"maze-t{tree:02d}" / "environment" / "maze"
                         / "tree.json").read_text())
    path = load("generate", DATASET_DIR / "generate.py").shortest_path(layout)
    return layout["doors"][layout["start"]].index(path[0]) + 1


def write_experiment(src: Path, tree: int, n: int, name: str, urls: list[str], model: str,
                     revision: str) -> None:
    """team.py's cluster spec at the dataset `revision`, with the sweep's model on its servers
    and a worker sized to the team, as swarm `name`."""
    team.main(["--tree", str(tree), "--agents", str(n), "--where", "cluster",
               "--revision", revision, "--out", str(src)])
    spec = yaml.safe_load((src / "spec.yaml").read_text())
    spec["name"] = name
    spec["model"] = model
    spec["inference"] = {"urls": urls}  # no key: the servers take none
    cpus = worker_cpus(n)
    spec["compute"] |= {"cpus": cpus, "mem": f"{cpus * MB_PER_CPU}M"}
    if MEMORY_LADDER:  # an agent's runaway process spikes the worker's memory (podman can't
        # cap a container here): drain or rotate later, so the spike doesn't end the run
        spec["workers"].setdefault("worker", {})["memory"] = dict(MEMORY_LADDER)
    (src / "spec.yaml").write_text(
        f"# {spec['task']['selector']} with N = {n} for the scaling sweep (sweep.py): team.py's "
        f"cluster spec with {model} on the sweep's servers.\n" + yaml.safe_dump(spec,
                                                                                sort_keys=False))


def profile(root: Path) -> Path:
    """The cluster profile: hopper-cpu, the venv, podman pulling Docker Hub images through a
    mirror (registries.conf), and 15 minutes (not 5) for a controller to come up: with a hundred
    teams starting at once, its imports from the shared disk can take minutes."""
    path = root / "profiles" / "cluster.env"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"SLURM_PARTITION=hopper-cpu\nSLURM_MAX_MEM_PER_CPU_MB=2000\n"
                    f"VENV={REPO / '.venv'}\n"
                    f"CONTAINERS_REGISTRIES_CONF={HERE.parent / 'registries.conf'}\n"
                    "HEALTH_WAIT_S=900\n")
    path.chmod(0o644)
    return path


def serving(url: str) -> bool:
    """The model server answers `GET /models`."""
    try:
        return httpx.get(f"{url.rstrip('/')}/models", timeout=10).status_code == 200
    except httpx.HTTPError:
        return False


def job_state(name: str) -> str:
    """The Slurm state of the job called `name`, '' when there is none."""
    out = subprocess.run(["squeue", "-h", "-n", name, "-o", "%T"], capture_output=True,
                         text=True, check=False).stdout.split()
    return out[0] if out else ""


def wait_done(run_dir: Path, task: str, n: int, timeout: float) -> str:
    """Poll `flock status` until the first exit stopped the run and its Trials are graded
    ('success'), or the loop is done and every agent is idle ('done'): or why not. A controller
    that stops answering and isn't running (a spot reclaim requeued it, or it is gone) is LOST."""
    deadline, unanswered, stopped = time.monotonic() + timeout, 0, None
    while time.monotonic() < deadline:
        status = flock(run_dir, "status", task, timeout=120)
        text = status.stdout
        unanswered = unanswered + 1 if status.returncode != 0 else 0
        if unanswered >= 3 and job_state(f"flock-ctl-{task}") != "RUNNING":
            return LOST
        if re.search(r"crashloop|launch_refused", text):
            return "worker failed: " + " ".join(text.split())[:300]
        if m := re.search(r"STOPPED ON SUCCESS: .*verification: (\w+)", text):
            stopped = stopped or time.monotonic()
            graded = m[1] != "pending" and not job_state(f"{task}-agent-w0000")
            if graded or time.monotonic() - stopped > 600:  # the cleanup budget is 5 min
                return "success"
        elif re.search(r"^loop agent: done", text, re.MULTILINE) and (agents := re.search(
                r"^agent .*$", text, re.MULTILINE)) and "working:" not in agents[0]:
            return "done"  # no agent in a turn (idle, or stuck: a turn that timed out)
        time.sleep(20)
    return f"not done after {timeout:.0f} s"


def release(run_dir: Path, name: str) -> bool:
    """`flock stop --release`: drains (and so grades) a run nobody won and ends its
    controller; then make sure no job of the run is left."""
    out = flock(run_dir, "stop", name, "--release")
    # once the worker is gone its results are on the shared disk: a controller still up a
    # minute later (it sometimes outlives the release) only holds the team's slot, and SIGTERM
    # makes it push its state and exit
    for job, wait in ((f"{name}-agent-w0000", 900), (f"flock-ctl-{name}", 60)):
        for _ in range(wait // 10):  # a draining worker is still grading its Trials
            if not job_state(job):
                break
            time.sleep(10)
        else:
            say(name, f"{job} still there after the release: cancelling it")
            subprocess.run(["scancel", "-n", job], check=False)
    return out.returncode == 0


def attempt(args, root: Path, secrets: Path, tree: int, n: int, name: str,
            urls: list[str]) -> dict | None:
    """One run of a team as swarm `name` on the servers `urls`: its row, or None when it could
    not start."""
    run_dir, src = root / "runs" / name, root / "src" / name
    if run_dir.exists() and (any((run_dir / "launch").glob("*.env"))
                             or (run_dir / "hosts" / "controller.host").exists()):
        # a previous driver launched it and stopped before its row: finish it from here (its
        # time cap counts from its own start, cluster.env's), or, with its controller gone,
        # score what it left
        alive = bool(job_state(f"flock-ctl-{name}"))
        say(name, "adopting the run a previous driver launched" + ("" if alive else
                                                                    " (no controller left)"))
        return finish(args, run_dir, name, tree, n, (run_dir / "cluster.env").stat().st_mtime,
                      None if alive else GONE)
    if run_dir.exists():
        shutil.rmtree(run_dir)  # init failed before: nothing ran
    started = time.time()
    # servers retired while the grid drained (or down) are left out, e.g. for a rerun
    urls = [u for u in urls if serving(u)] or [u for u in args.urls if serving(u)] or urls
    write_experiment(src, tree, n, name, urls, args.model, args.revision)
    log = root / "logs" / f"{name}-init.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for tries in range(6):  # its checks probe the model servers, which may be busy
        with init_lock:  # a few at a time: many at once race on the bucket's write check
            init = flock(run_dir, "init", str(run_dir), "--from", str(src), "--profile",
                         str(profile(root)), "--secrets", str(secrets))
        log.write_text(init.stdout + init.stderr)
        if init.returncode == 0:
            break
        time.sleep(30 * (tries + 1))
    else:
        failed = [x for x in init.stdout.splitlines() if x.startswith("FAIL")]
        say(name, "init failed 6 times:", failed or init.stdout[-300:], "(log:", log, ")")
        return None
    say(name, "launching")
    launch = flock(run_dir, "launch", str(run_dir / "experiment"))
    if launch.returncode != 0:
        say(name, "launch failed:", launch.stdout[-600:], launch.stderr[-600:])
        return finish(args, run_dir, name, tree, n, started, "launch")
    return finish(args, run_dir, name, tree, n, started)


def finish(args, run_dir: Path, name: str, tree: int, n: int, started: float,
           why: str | None = None) -> dict:
    """Wait for a launched team, release it (its Trials are graded, its controller ends) and
    make its row."""
    why = why or wait_done(run_dir, name, n, max(0.0, args.team_timeout - (time.time() - started)))
    if why == LOST:  # nothing to release: end its jobs, and the team runs again
        say(name, "controller lost (spot reclaim?): cancelling its jobs")
        for job in (f"flock-ctl-{name}", f"{name}-agent-w0000"):
            subprocess.run(["scancel", "-n", job], check=False)
        released = False
    elif why == GONE:  # its rewards and export are on disk, or it doesn't count
        released = False
    else:
        say(name, "outcome:", why, "- releasing")
        released = release(run_dir, name)
    ended = time.time()
    return metrics.row(run_dir) | {
        "model": args.model, "tree": tree, "exit_branch": exit_branch(tree),
        "wall_s": round(ended - started, 1), "outcome": why, "released": released,
        # each launch of the worker slot writes one env file: more than one is a relaunch or a
        # preemption, whose results a later incarnation overwrites (flock#121)
        "incarnations": len(list((run_dir / "launch").glob("*.env"))),
        "worker_wait_s": worker_wait(run_dir),
        "thrashed_min": thrashed_minutes(run_dir, n, started, ended),
        "started_at": datetime.fromtimestamp(started, UTC).isoformat()}


def thrashed_minutes(run_dir: Path, n: int, start: float, end: float) -> int | None:
    """Minutes, within the run, in which a server one of its agents calls was thrashing: a
    prefix-cache hit rate under 50% while prefilling at least 200k prompt tokens (every call
    recomputing an agent's long context; a cold server's first small prompts don't count), from
    the health log (`--health`); None without one."""
    spec = run_dir / "experiment" / "spec.yaml"
    if HEALTH is None or not HEALTH.is_file() or not spec.is_file():
        return None
    urls = yaml.safe_load(spec.read_text())["inference"]["urls"]
    servers = {u.split("//")[1].removesuffix("/v1") for u in agents_on(urls, n)}
    minutes = set()
    for line in HEALTH.read_text().splitlines():
        s = json.loads(line)
        if (start <= s["t"] <= end and s["server"] in servers and s["hit_rate"] is not None
                and s["hit_rate"] < 0.5 and s["prompt"] >= 200_000):
            minutes.add(int(s["t"] // 60))
    return len(minutes)


def worker_wait(run_dir: Path) -> float | None:
    """Seconds from the controller writing the worker's first launch file to the worker job's
    first log line (None: not known)."""
    launches = sorted((run_dir / "launch").glob("*.env"), key=lambda p: p.stat().st_mtime)
    logs = sorted((run_dir / "logs").glob("*-agent-w*.out"))
    if not launches or not logs:
        return None
    first = logs[0].read_text(errors="replace").split("\n", 1)[0]
    try:
        began = datetime.fromisoformat(first.split(" ", 1)[0])
    except ValueError:
        return None
    return round(began.timestamp() - launches[0].stat().st_mtime, 1)


def why_invalid(row: dict) -> str:
    """Why a team's run doesn't count ('' when it does): the infrastructure's failures, never the
    model's."""
    wait = row.get("worker_wait_s")
    started = datetime.fromisoformat(row["started_at"]) if row.get("started_at") else None
    overloaded = bool(OVERLOADED and started and OVERLOADED[0] <= started < OVERLOADED[1])
    thrashed = row.get("thrashed_min") or 0
    return "; ".join(why for why, bad in (
        ("ran on an overloaded model server (KV-cache thrash)", overloaded),
        (f"an agent's model server thrashed for {thrashed} min", thrashed >= THRASH_LIMIT),
        (f"{row.get('incarnations')} worker incarnations", row.get("incarnations") != 1),
        ("worker never started", wait is None),
        (f"worker started {wait or 0:.0f} s after the launch", (wait or 0) > WAIT_LIMIT),
        (LOST, row.get("outcome") == LOST),
        (f"missing {', '.join(row['missing'])}"[:160], bool(row["missing"]))) if bad)


def valid(row: dict) -> bool:
    return not why_invalid(row)


def run_team(args, root: Path, secrets: Path, tree: int, n: int,
             urls: list[str]) -> dict | None:
    """A team, run again (as `<task>-rK`) until its run counts."""
    task = f"{args.tag}maze-t{tree:02d}-n{n:02d}"
    row = None
    for reruns in range(args.max_reruns + 1):
        row = attempt(args, root, secrets, tree, n, task if reruns == 0 else f"{task}-r{reruns}",
                      urls)
        if row is None:
            return None
        row["reruns"] = reruns
        if valid(row):
            break
        say(task, f"run doesn't count ({why_invalid(row)}): running it again")
        row["set_aside"] = why_invalid(row)
        with lock, (root / "rows-discarded.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
    else:
        say(task, f"gave up after {args.max_reruns} reruns")
        return row
    with lock, (root / "rows.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    first = row.get("first_exit") or {}
    say(task, f"success={row['success']} exit={first.get('agent')} after "
        f"{first.get('after_s')} s, {first.get('team_moves')} team moves; posts={row['posts']} "
        f"overlap={row.get('room_overlap')} wall={row['wall_s']} s reruns={row['reruns']}")
    return row


def agents_on(urls: list[str], n: int) -> Counter:
    """How many of a team's agents (agent-001 .. agent-00N) call each server: flock hashes each
    agent's name over the team's URLs (Inference.endpoint_for), so with consecutive names
    agent-001 calls the same server in every team that has the same URLs."""
    block = Inference(urls=urls)
    return Counter(block.endpoint_for(f"agent-{k:03d}") for k in range(1, n + 1))


class Slots:
    """At most `parallel` teams, `max_agents` agents and `per_server` agents per model server
    (`<root>/max_agents` and `<root>/per_server` override them while the driver runs) in flight,
    granted in order. Each new team gets the subset of the servers that spreads its agents onto
    the least-loaded ones (one server's KV cache holds about a dozen long agent contexts)."""

    def __init__(self, root: Path, urls: list[str], parallel: int, max_agents: int,
                 per_server: int):
        self.root, self.urls, self.parallel = root, urls, parallel
        self.defaults = {"max_agents": max_agents, "per_server": per_server}
        self.cond, self.teams, self.agents, self.load = threading.Condition(), 0, 0, Counter()

    def cap(self, name: str) -> int:
        try:
            return int((self.root / name).read_text().strip())
        except (OSError, ValueError):
            return self.defaults[name]

    def subsets(self):
        """Every subset of the servers when there are at most 12; else (2^24 is too many to
        try per team) the k least-loaded servers for each k and 3000 random subsets."""
        if len(self.urls) <= 12:
            for size in range(1, len(self.urls) + 1):
                yield from itertools.combinations(self.urls, size)
            return
        by_load = sorted(self.urls, key=lambda u: self.load[u])
        for k in range(1, len(by_load) + 1):
            yield tuple(by_load[:k])
        rng = random.Random(len(self.urls) * 1000 + sum(self.load.values()))
        for _ in range(3000):
            yield tuple(rng.sample(self.urls, rng.randint(1, len(self.urls))))

    def choose(self, n: int) -> list[str] | None:
        """The subset of the servers whose mapping keeps every server within its cap and the
        busiest as idle as possible; None when none fits now."""
        best, cap = None, self.cap("per_server")
        for subset in self.subsets():
            added = agents_on(list(subset), n)
            busiest = max(self.load[u] + k for u, k in added.items())  # where it adds only
            after = self.load + added
            score = (busiest, sum(v * v for v in after.values()))
            if busiest <= cap and (best is None or score < best[0]):
                best = (score, list(subset))
        return best[1] if best else None

    def take(self, n: int, urls: list[str] | None = None) -> list[str]:
        """Wait for room for a team of n; `urls`: an adopted team's own (it already runs)."""
        with self.cond:
            while True:
                if self.teams < self.parallel and (not self.agents or urls is not None
                                                   or self.agents + n <= self.cap("max_agents")):
                    chosen = urls or self.choose(n)
                    if chosen is not None or not self.agents:
                        chosen = chosen or list(self.urls)
                        break
                self.cond.wait(30)
            self.teams, self.agents = self.teams + 1, self.agents + n
            self.load += agents_on(chosen, n)
            return chosen

    def give(self, n: int, urls: list[str]) -> None:
        with self.cond:
            self.teams, self.agents = self.teams - 1, self.agents - n
            self.load -= agents_on(urls, n)
            self.cond.notify_all()


def live_urls(root: Path, tag: str, tree: int, n: int) -> list[str] | None:
    """The URLs of a team's run that a stopped driver launched and whose controller is still
    there (it keeps them), else None."""
    for spec in sorted((root / "src").glob(f"{tag}maze-t{tree:02d}-n{n:02d}*/spec.yaml")):
        if job_state(f"flock-ctl-{spec.parent.name}"):
            return yaml.safe_load(spec.read_text())["inference"]["urls"]
    return None


def upload(root: Path, dataset: str, prefix: str) -> None:
    """Each run directory's spec, results and logs, and the rows, to a private dataset; never
    run.env, and nothing that holds a Hub token."""
    api = HfApi()
    api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
    if not api.dataset_info(dataset).private:
        raise SystemExit(f"{dataset} is public: refusing to upload")
    allow = ["*.jsonl", "*.csv", "*.md", "*.png", "runs/*/experiment/**", "runs/*/results/**",
             "runs/*/logs/**", "driver.log"]
    leaks = [str(p) for p in root.rglob("*") if p.is_file() and not p.name.endswith(".env")
             and SECRET.search(p.read_text(errors="ignore"))]
    if leaks:
        raise SystemExit(f"refusing to upload: Hub tokens in {leaks[:5]}")
    api.upload_folder(repo_id=dataset, repo_type="dataset", folder_path=root,
                      path_in_repo=prefix, allow_patterns=allow,
                      commit_message=f"s3-maze sweep: {prefix}")
    say("uploaded", root, "to", f"{dataset}/{prefix}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--urls", required=True, help="the model servers, comma-separated")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True,
                        help="put before every run and swarm name (e.g. q27-): two grids at "
                             "once then never share a Slurm job name")
    parser.add_argument("--revision", help="the dataset revision every team pins (default: its "
                        "head)")
    parser.add_argument("--trees", default="1-32")
    parser.add_argument("--agents", default="1,2,4,8,16")
    parser.add_argument("--parallel", type=int, default=20, help="teams at once")
    parser.add_argument("--max-agents", type=int, default=1000, help="agents at once")
    parser.add_argument("--per-server", type=int, default=1000,
                        help="agents at once per model server (each team gets the subset of "
                             "--urls that keeps the servers within it)")
    parser.add_argument("--team-timeout", type=float, default=4800,
                        help="s from a team's start until it is released anyway")
    parser.add_argument("--max-reruns", type=int, default=4,
                        help="runs again of a team whose run doesn't count (why_invalid)")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bucket", default="lewtun/s3-maze-sweep",
                        help="private bucket for the runs' state (a swarm's name is used once "
                             "in a bucket)")
    parser.add_argument("--namespace", default="lewtun")
    parser.add_argument("--dataset", help="private dataset for the run directories")
    parser.add_argument("--prefix", help="folder in the dataset (default: the root's name)")
    parser.add_argument("--overloaded", help="START/END (ISO times): runs started then ran on "
                        "an overloaded model server, and run again")
    parser.add_argument("--health", type=Path,
                        help="the servers' health log (health-log.sh's servers.jsonl)")
    parser.add_argument("--cpus-per-agent", type=int, default=1,
                        help="worker CPUs (and so memory, at 1900 MB each) per agent")
    parser.add_argument("--memory-ladder", help="ROTATE,DRAIN,PREEMPT: the workers' memory "
                        "fractions (default: flock's 0.75,0.85,0.93)")
    args = parser.parse_args(argv)
    global HEALTH, CPUS_PER_AGENT, MEMORY_LADDER
    HEALTH = args.health
    CPUS_PER_AGENT = args.cpus_per_agent
    if args.memory_ladder:
        MEMORY_LADDER = dict(zip(("rotate", "drain", "preempt"),
                                 map(float, args.memory_ladder.split(",")), strict=True))
    if args.overloaded:
        global OVERLOADED
        start, end = (datetime.fromisoformat(t) for t in args.overloaded.split("/"))
        OVERLOADED = (start, end)
    args.urls = args.urls.split(",")
    args.revision = team.pinned(args.revision)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    api.create_bucket(args.bucket, private=True, exist_ok=True)
    if not api.bucket_info(args.bucket).private:
        raise SystemExit(f"bucket {args.bucket} is public: refusing")
    token = os.environ.get("HF_TOKEN") or get_token()
    secrets = root / f"secrets-{os.getpid()}.env"  # per driver
    old = os.umask(0o077)
    secrets.write_text(f"HF_TOKEN={token}\nNAMESPACE={args.namespace}\n"
                       f"BUCKET=hf://buckets/{args.bucket}\n")
    os.umask(old)
    rows_file = root / "rows.jsonl"
    rows = [json.loads(x) for x in rows_file.read_text().splitlines()] if rows_file.exists() else []
    for r in rows:  # rows from before the health check get it too
        if r.get("thrashed_min") is None and r.get("started_at") and r.get("wall_s"):
            start = datetime.fromisoformat(r["started_at"]).timestamp()
            r["thrashed_min"] = thrashed_minutes(root / "runs" / r["run"], r["agents"], start,
                                                 start + r["wall_s"])
    if bad := [r for r in rows if not valid(r)]:  # an earlier driver kept them: run them again
        with (root / "rows-discarded.jsonl").open("a") as f:
            f.write("".join(json.dumps(r | {"set_aside": why_invalid(r)}) + "\n" for r in bad))
        rows_file.write_text("".join(json.dumps(r) + "\n" for r in rows if valid(r)))
        say(f"{len(bad)} rows that don't count moved to rows-discarded.jsonl")
    done = {(r["tree"], r["agents"]) for r in rows if valid(r)}
    teams = [(t, n) for t in numbers(args.trees) for n in numbers(args.agents)
             if (t, n) not in done]
    # teams a stopped driver launched first: they hold their own time and their workers
    launched = {(t, n) for t, n in teams if any((root / "runs").glob(
        f"{args.tag}maze-t{t:02d}-n{n:02d}*/launch/*.env"))}
    teams.sort(key=lambda t: t not in launched)
    say(f"{len(teams)} teams to run ({len(done)} done, {len(launched)} to adopt), "
        f"{args.parallel} teams and {args.max_agents} agents at once, on {args.urls}")
    slots = Slots(root, args.urls, args.parallel, args.max_agents, args.per_server)

    def one(t: int, n: int, urls: list[str]):
        try:
            return run_team(args, root, secrets, t, n, urls)
        except Exception as e:  # noqa: BLE001 - one team's failure must not stop the grid
            say(f"{args.tag}maze-t{t:02d}-n{n:02d}", "failed:", repr(e)[:300])
        finally:
            slots.give(n, urls)

    try:
        with ThreadPoolExecutor(args.parallel) as pool:
            for t, n in teams:
                urls = slots.take(n, live_urls(root, args.tag, t, n))
                pool.submit(one, t, n, urls)
    finally:
        secrets.unlink(missing_ok=True)
    if args.dataset:
        upload(root, args.dataset, args.prefix or root.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
