"""Recompute the length-controlled win rate of the SmolTalkBench summaries in the bucket with the
gamed baselines of smoltalk_bench.GAMED, from the preferences and lengths in each run's Inspect
log, as an eval scored now would.

A rewritten summary records the gamed baselines as metadata.lc_gamed, and summaries that have
them are skipped. The first rewrite keeps the summary it replaces beside it, as
summary.minimal.json. A run scored by another judge gets a null LC win rate. The Inspect logs
keep the metrics they were scored with.

Usage:

uv run --group evals experiments/run4/eval/rescore.py --limit 20   # print the first changes
uv run --group evals experiments/run4/eval/rescore.py --write      # rewrite the summaries
uv run --group evals experiments/run4/eval/rescore.py --job        # --write, on a cpu-xl Job
"""

import argparse
import json
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import evaluate
import smoltalk_bench
from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import disable_progress_bars

TASK = "smoltalk_bench_gold"
SCORER = "alpaca_eval_judge"
GAMED = f"{smoltalk_bench.BENCH}/{smoltalk_bench.GAMED}@{smoltalk_bench.BENCH_REVISION}"
BACKUP = "summary.minimal.json"
JOB_FLAVOR = "cpu-xl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", default=evaluate.DEFAULT_BUCKET)
    parser.add_argument("--write", action="store_true", help="rewrite the summaries")
    parser.add_argument("--limit", type=int, help="only the first N summaries")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--job", action="store_true",
                        help=f"run with --write on a {JOB_FLAVOR} Job instead")
    parser.add_argument("--namespace", help="who runs the Job (default: the token's user)")
    return parser.parse_args()


def runs(api: HfApi, bucket: str) -> list[tuple[str, str]]:
    """Each summary.json of the task, and the newest Inspect log in its folder."""
    summaries, logs = [], {}
    for entry in api.list_bucket_tree(bucket, recursive=True):
        if getattr(entry, "type", None) != "file":
            continue
        path = PurePosixPath(entry.path)
        if path.name == "summary.json" and path.parent.parent.name == TASK:
            summaries.append(path)
        elif path.suffix == ".eval":
            logs[path.parent] = max(logs.get(path.parent, ""), entry.path)
    return sorted((str(path), logs[path.parent]) for path in summaries if path.parent in logs)


def rescore(bucket: str, summary_path: str, log_path: str) -> tuple[dict, dict] | str | None:
    """The summary, and that summary with the gamed baselines' LC win rate; None if it has it
    already, or the error."""
    from inspect_ai.log import read_eval_log

    try:
        api = HfApi()
        with tempfile.TemporaryDirectory() as tmp:
            api.download_bucket_files(bucket, [(summary_path, f"{tmp}/summary.json")])
            old = json.loads(Path(f"{tmp}/summary.json").read_text())
            if old["metadata"].get("lc_gamed") == GAMED:
                return None
            api.download_bucket_files(bucket, [(log_path, f"{tmp}/log.eval")])
            samples = read_eval_log(f"{tmp}/log.eval").samples or []
        # Logs from before the judge was in the score metadata name it as the grader role
        grader = old["metadata"].get("model_roles", {}).get("grader")
        meta = [{"judge": grader} | s.scores[SCORER].metadata for s in samples
                if s.scores and SCORER in s.scores]
        lc = smoltalk_bench.length_controlled(meta, smoltalk_bench.GAMED)
    except Exception as exc:  # noqa: BLE001 - one broken log shouldn't stop the others
        return f"{type(exc).__name__}: {exc}"
    lc = {k: None if math.isnan(v) else float(v) for k, v in lc.items()}
    new = json.loads(json.dumps(old))
    new["metrics"].update(lc)
    new["primary_metric"].update(value=lc["length_controlled_win_rate"],
                                 standard_error=lc["lc_standard_error"])
    new["metadata"]["lc_gamed"] = GAMED
    return old, new


def launch(api: HfApi, token: str, args: argparse.Namespace) -> None:
    """Runs this script with --write on a Job, from a snapshot of this checkout's eval code."""
    namespace = args.namespace or api.whoami()["name"]
    volume = evaluate.snapshot_volume(api, args.bucket.split("/", 1)[0])
    script = (f"cp -r {evaluate.JOB_ROOT} /tmp/flock && cd /tmp/flock && "
              f"uv sync --frozen --only-group evals && "
              f".venv/bin/python {evaluate.EVAL_DIR}/rescore.py --write --bucket {args.bucket}")
    job = api.run_job(image=evaluate.CPU_IMAGE, command=["bash", "-c", script],
                      flavor=JOB_FLAVOR, timeout="3h", secrets={"HF_TOKEN": token},
                      env=evaluate.JOB_ENV, volumes=[volume], namespace=namespace)
    print(f"Job: {job.url}")


def main():
    args = parse_args()
    token = get_token()
    if token is None:
        raise RuntimeError("the bucket needs a Hugging Face token; run `hf auth login` first")
    api = HfApi(token=token)
    disable_progress_bars()
    if args.job:
        launch(api, token, args)
        return
    todo = runs(api, args.bucket)[:args.limit]
    print(f"{len(todo)} summaries, gamed baselines {GAMED}")
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(rescore, [args.bucket] * len(todo), *zip(*todo, strict=True)))
    adds, rewrites, errors = [], 0, []
    for (summary_path, _), result in zip(todo, results, strict=True):
        if isinstance(result, str):
            errors.append(f"{summary_path}: {result}")
            continue
        if result is None:
            continue
        old, new = result
        print(f"{summary_path}: LC {old['metrics'].get('length_controlled_win_rate')} -> "
              f"{new['metrics']['length_controlled_win_rate']}")
        rewrites += 1
        adds.append(((json.dumps(new, indent=2) + "\n").encode(), summary_path))
        if "lc_gamed" not in old["metadata"]:
            adds.append(((json.dumps(old, indent=2) + "\n").encode(),
                         str(PurePosixPath(summary_path).with_name(BACKUP))))
    print(f"{rewrites} to rewrite, {len(todo) - rewrites - len(errors)} already rescored, "
          f"{len(errors)} errors")
    for error in errors:
        print(error)
    if args.write:
        for start in range(0, len(adds), 500):
            api.batch_bucket_files(args.bucket, add=adds[start:start + 500])
        print(f"rewrote {rewrites} summaries")


if __name__ == "__main__":
    main()
