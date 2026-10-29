"""Keep the SmolTalkBench evals of the swarms' chatbots up to date: launch an eval for every
`main` commit of every HuggingFaceSwarms/SwarmChat-v<N> dataset that changes the bot and has no
eval yet, many at once, newest first.

A run keeps no state of its own; the bucket and the Jobs list are the state. It drops the commits
with a summary.json or a skipped.json in the bucket, those with an eval in flight, and those whose
evals already failed --max_attempts times. It then walks the rest newest first, round-robin across
repos, and launches one Job per commit, up to --max_running evals in flight in all. A commit that
leaves bot/ as its parent left it gets a skipped.json instead, since its replies would be its
parent's.

`--schedule CRON` registers this script as a scheduled Job, so new merges are scored without
anyone launching them. The scheduled Job and its evals run the eval code as it is when the
schedule is created; rerun `--schedule` to update it.

Usage:

uv run experiments/run4/eval/sync.py --dry_run
uv run experiments/run4/eval/sync.py
uv run experiments/run4/eval/sync.py --schedule "*/10 * * * *"
"""

import argparse
import dataclasses
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import evaluate
import huggingface_hub
from huggingface_hub import HfApi, Volume, get_token
from huggingface_hub.utils import disable_progress_bars

ORG = "HuggingFaceSwarms"
SWARM_REPO = re.compile(rf"{ORG}/SwarmChat-v\d+")
TASK = "smoltalk_bench_gold"
ACTIVE, FAILED = ("SCHEDULING", "RUNNING"), ("ERROR", "CANCELED")
# Inside the scheduled Job, the volume with the eval code, for the evals to mount too
VOLUME_ENV = "SWARMCHAT_EVAL_VOLUME"
SCHEDULE_LABELS = {"app": "swarmchat-sync"}
LAUNCH_THREADS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repos", nargs="+",
                        help=f"default: every {ORG}/SwarmChat-v<N> dataset")
    parser.add_argument("--max_running", type=int, default=50,
                        help="evals in flight at most, in all (the judge's rate limit)")
    parser.add_argument("--max_attempts", type=int, default=2,
                        help="failed evals of a commit after which it is left alone")
    parser.add_argument("--bucket", default=evaluate.DEFAULT_BUCKET)
    parser.add_argument("--namespace", help="who runs the Jobs (default: the token's user)")
    parser.add_argument("--schedule", metavar="CRON",
                        help='register a scheduled Job that runs this sync, e.g. "*/10 * * * *"')
    parser.add_argument("--dry_run", action="store_true",
                        help="print what a run would launch and skip, and change nothing")
    return parser.parse_args()


def swarm_repos(api: HfApi, repos: list[str] | None) -> list[str]:
    """The SwarmChat-v<N> datasets, in version order."""
    if repos:
        if bad := [repo for repo in repos if not SWARM_REPO.fullmatch(repo)]:
            raise ValueError(f"not a {ORG}/SwarmChat-v<N> dataset: {bad}")
    else:
        repos = [d.id for d in api.list_datasets(author=ORG, search="SwarmChat")
                 if SWARM_REPO.fullmatch(d.id)]
    return sorted(repos, key=lambda repo: int(repo.rsplit("-v", 1)[1]))


def settled(api: HfApi, bucket: str, repo: str) -> set[str]:
    """The commits (first 12 hex digits) with a summary.json or a skipped.json in the bucket."""
    commits = set()
    for entry in api.list_bucket_tree(bucket, prefix=f"{repo}/", recursive=True):
        parts = entry.path.removeprefix(f"{repo}/").split("/")
        if (getattr(entry, "type", "file") == "file"
                and (parts[-1] == "summary.json" and TASK in parts or parts[-1] == "skipped.json")):
            commits.add(parts[0].rsplit("-", 1)[-1])
    return commits


def eval_jobs(api: HfApi, namespace: str) -> tuple[set[tuple[str, str]], Counter, int]:
    """The (repo, commit) pairs with a SwarmChat eval in flight, the failed evals per pair, and
    how many evals of any model are in flight."""
    in_flight, failures, running = set(), Counter(), 0
    for job in api.list_jobs(namespace=namespace, status=[*ACTIVE, *FAILED]):
        command = job.command or []
        if not any(part.endswith("hf_job.py") for part in command):
            continue
        active = job.status.stage in ACTIVE
        running += active
        if "--revision" in command and SWARM_REPO.fullmatch(command[command.index("--model") + 1]):
            pair = (command[command.index("--model") + 1],
                    command[command.index("--revision") + 1][:12])
            if active:
                in_flight.add(pair)
            else:
                failures[pair] += 1
    return in_flight, failures, running


def job_volume(api: HfApi, bucket: str) -> Volume:
    """The scheduled Job's own volume, or a fresh snapshot of this checkout's eval code."""
    if spec := os.environ.get(VOLUME_ENV):
        return Volume(**json.loads(spec))
    return evaluate.snapshot_volume(api, bucket.split("/", 1)[0])


def schedule(api: HfApi, token: str, namespace: str, args: argparse.Namespace) -> None:
    volume = job_volume(api, args.bucket)
    command = ["uv", "run", "--no-project", "--with",
               f"huggingface-hub=={huggingface_hub.__version__}", "python",
               f"{evaluate.JOB_ROOT}/{evaluate.EVAL_DIR}/sync.py", "--bucket", args.bucket,
               "--namespace", namespace, "--max_running", str(args.max_running),
               "--max_attempts", str(args.max_attempts)]
    if args.repos:
        command += ["--repos", *args.repos]
    for old in api.list_scheduled_jobs(labels=SCHEDULE_LABELS, namespace=namespace):
        api.delete_scheduled_job(scheduled_job_id=old.id, namespace=namespace)
        print(f"Deleted the previous schedule {old.id}")
    info = api.create_scheduled_job(
        image=evaluate.CPU_IMAGE, command=command, schedule=args.schedule, concurrency=False,
        flavor="cpu-basic", timeout="15m", secrets={"HF_TOKEN": token},
        env={**evaluate.JOB_ENV, VOLUME_ENV: json.dumps(dataclasses.asdict(volume))},
        volumes=[volume], labels=SCHEDULE_LABELS, namespace=namespace)
    print(f"Scheduled {info.id} ({args.schedule}): https://huggingface.co/jobs/{namespace}")


def main():
    args = parse_args()
    token = get_token()
    if token is None:
        raise RuntimeError("HF Jobs need a Hugging Face token; run `hf auth login` first")
    api = HfApi(token=token)
    namespace = args.namespace or api.whoami()["name"]
    disable_progress_bars()
    if args.schedule:
        schedule(api, token, namespace, args)
        return

    repos = swarm_repos(api, args.repos)
    with ThreadPoolExecutor(LAUNCH_THREADS) as pool:
        histories = dict(zip(repos, pool.map(
            lambda repo: api.list_repo_commits(repo, repo_type="dataset"), repos)))
        done = dict(zip(repos, pool.map(lambda repo: settled(api, args.bucket, repo), repos)))
    in_flight, failures, running = eval_jobs(api, namespace)

    # Per repo, newest first: (commit, its parent on main) still to evaluate
    queues = {repo: [] for repo in repos}
    for repo, commits in histories.items():
        for commit, parent in zip(commits, [*commits[1:], None]):
            pair = (repo, commit.commit_id[:12])
            if (pair[1] not in done[repo] and pair not in in_flight
                    and failures[pair] < args.max_attempts):
                queues[repo].append((commit, parent))

    trees = {}

    def bot_tree(repo, commit):
        if commit.commit_id not in trees:
            trees[commit.commit_id] = next(
                (entry.tree_id for entry in api.list_repo_tree(
                    repo, repo_type="dataset", revision=commit.commit_id)
                 if entry.path == "bot"), None)
        return trees[commit.commit_id]

    # Newest first, round-robin across repos, until the evals in flight reach the cap
    launches, skips, slots = [], [], args.max_running - running
    while len(launches) < slots and any(queues.values()):
        for repo in repos:
            if not queues[repo] or len(launches) >= slots:
                continue
            commit, parent = queues[repo].pop(0)
            tree = bot_tree(repo, commit)
            if tree is None or parent is not None and tree == bot_tree(repo, parent):
                skips.append((repo, commit, parent))
            else:
                launches.append((repo, commit))
    left = sum(len(queue) for queue in queues.values())
    print(f"{len(repos)} repos ({', '.join(r.rsplit('/', 1)[1] for r in repos)}), "
          f"{sum(len(h) for h in histories.values())} main commits, "
          f"{sum(len(d) for d in done.values())} settled, {running} evals in flight "
          f"({len(in_flight)} SwarmChat) | launching {len(launches)}, skipping {len(skips)}, "
          f"{left} left")

    for repo, commit, parent in skips:
        print(f"skip {repo}@{commit.commit_id[:7]}: " + (
            "no bot/" if bot_tree(repo, commit) is None else f"bot/ as at {parent.commit_id[:7]}"))
    for repo, commit in launches:
        print(f"launch {repo}@{commit.commit_id[:7]} ({commit.created_at:%Y-%m-%d %H:%M} UTC)")
    if args.dry_run:
        return

    markers = []
    for repo, commit, parent in skips:
        marker = {"reason": "no bot/" if bot_tree(repo, commit) is None else "bot/ unchanged",
                  "commit": commit.commit_id,
                  "same_as": parent and evaluate.swarmchat_revision(parent.commit_id,
                                                                    parent.created_at)}
        revision = evaluate.swarmchat_revision(commit.commit_id, commit.created_at)
        markers.append(((json.dumps(marker, indent=2) + "\n").encode(),
                        f"{repo}/{revision}/skipped.json"))
    if markers:
        api.batch_bucket_files(args.bucket, add=markers)
    if launches:
        secrets = evaluate.job_secrets(token, [])
        volume = job_volume(api, args.bucket)

        def launch(item):
            repo, commit = item
            try:
                job, _ = evaluate.build_job(evaluate.parse_args(
                    ["--model", repo, "--revision", commit.commit_id, "--bucket", args.bucket,
                     "--namespace", namespace]), api, namespace)
                info = api.run_job(**job, secrets=secrets, volumes=[volume])
                return f"launched {repo}@{commit.commit_id[:7]}: {info.url}"
            except Exception as error:  # noqa: BLE001 - the next run retries this commit
                return f"failed to launch {repo}@{commit.commit_id[:7]}: {error}"

        with ThreadPoolExecutor(LAUNCH_THREADS) as pool:
            for line in pool.map(launch, launches):
                print(line, flush=True)


if __name__ == "__main__":
    main()
