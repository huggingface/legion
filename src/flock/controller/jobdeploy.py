"""Launch the controller as an HF Job (job mode, docs/deploy-job.md). Used by
scripts/deploy-job.sh; stdlib plus huggingface_hub only, so it runs from a bare checkout.

    python -m flock.controller.jobdeploy <wheel-url> [--flavor cpu-upgrade] [--group flock]
        [--timeout 48h] [--namespace NS] [--state-url hf://buckets/...] [--secret NAME ...]
        [--replace] [--dry-run]
    python -m flock.controller.jobdeploy --down --group G [<wheel-url> | --state-url URL]
        cancel the group's controller (final state push), then delete its state prefix, so the
        next controller in the group starts empty
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Mapping
from typing import Any

API_PORT = 7860
STATUS_PORT = 7861
ALIAS = "controller"
IMAGE = "python:3.12-slim"
DATA_DIR = "/state/flock"  # job disk: ephemeral, synced to FLOCK_STATE_URL
ROLE_LABEL = "flock-role"
GROUP_LABEL = "flock-group"
DEFAULT_GROUP = "flock"

# As the workers' (flock.controller.jobs): msgboard is private on GitHub (a job cannot clone it),
# so its wheel and uv.lock's versions come in harbor-deps.tar next to flock's wheel
# (scripts/publish-wheel.sh); uv installs flock with those pins and msgboard's wheel.
_BOOTSTRAP = r"""set -euo pipefail
t0=$(date +%s)
python3 -m pip install -q --no-cache-dir --root-user-action=ignore huggingface_hub uv
mkdir -p /opt/wheels "$FLOCK_DATA"
fetch() {  # fetch hf://buckets/<namespace>/<bucket>/<path> <file>
  python3 -c 'import sys; from huggingface_hub import HfApi
ns, name, path = sys.argv[1].removeprefix("hf://buckets/").split("/", 2)
HfApi().download_bucket_files(f"{ns}/{name}", [(path, sys.argv[2])], raise_on_missing_files=True)
' "$1" "$2"
}
WHEEL=/opt/wheels/$(basename "$FLOCK_WHEEL_URL")
fetch "$FLOCK_WHEEL_URL" "$WHEEL"
fetch "${FLOCK_WHEEL_URL%/*}/harbor-deps.tar" /opt/wheels/harbor-deps.tar || {
  echo "bootstrap: no harbor-deps.tar next to the wheel: scripts/publish-wheel.sh" >&2
  exit 1; }
tar -xf /opt/wheels/harbor-deps.tar -C /opt/wheels
echo "msgboard @ file://$(ls /opt/wheels/msgboard-*.whl)" > /opt/wheels/overrides.txt
uv pip install -q --system --no-cache "$WHEEL[controller]" -c /opt/wheels/constraints.txt \
  --overrides /opt/wheels/overrides.txt
echo "bootstrap: $(($(date +%s) - t0)) s; $(python3 --version)"
exec flock-controller
"""


def bootstrap_command() -> list[str]:
    return ["bash", "-c", _BOOTSTRAP]


def default_state_url(wheel_url: str, group: str) -> str:
    """hf://buckets/<ns>/<bucket>/... (the wheel's bucket) -> .../controller/<group>."""
    rest = wheel_url.removeprefix("hf://buckets/").split("/")
    if not wheel_url.startswith("hf://buckets/") or len(rest) < 3:
        raise ValueError(f"wheel url must be hf://buckets/<ns>/<bucket>/<path>: {wheel_url!r}")
    return f"hf://buckets/{rest[0]}/{rest[1]}/controller/{group}"


def run_job_args(
    *,
    wheel_url: str,
    group: str,
    namespace: str,
    flavor: str,
    timeout: str,
    state_url: str,
    job_labels: str,
    secrets: Mapping[str, str],
) -> dict[str, Any]:
    """Keyword arguments for `HfApi.run_job` that start the controller."""
    env = {
        "FLOCK_MODE": "job",
        "FLOCK_NETWORK_GROUP": group,
        "FLOCK_NETWORK_ALIAS": ALIAS,
        "FLOCK_NAMESPACE": namespace,
        "FLOCK_STATUS_PORT": str(STATUS_PORT),
        "PORT": str(API_PORT),
        "FLOCK_DATA": DATA_DIR,
        "FLOCK_STATE_URL": state_url,
        "FLOCK_WHEEL_URL": wheel_url,
        "FLOCK_WORKER_BOOTSTRAP": "1",
        "FLOCK_JOB_LABELS": job_labels,
        "PYTHONUNBUFFERED": "1",
    }
    out = {
        "image": IMAGE,
        "command": bootstrap_command(),
        "env": env,
        "secrets": dict(secrets),
        "flavor": flavor,
        "timeout": timeout,
        "labels": {ROLE_LABEL: "controller", GROUP_LABEL: group},
        "expose": [API_PORT],
        "network_group": group,
        "network_aliases": [ALIAS],
        "namespace": namespace,
    }
    if public_status_supported():
        out["expose_public"] = [STATUS_PORT]
    return out


def public_status_supported() -> bool:
    """huggingface_hub 1.33 (uv.lock's pin, Harbor's) has no `expose_public`; the public status
    page is optional, so without it the controller starts and the page is just not exposed."""
    import inspect

    from huggingface_hub import HfApi

    return "expose_public" in inspect.signature(HfApi.run_job).parameters


def collect_secrets(env: Mapping[str, str], extra: list[str]) -> dict[str, str]:
    """FLOCK_TOKEN, HF_TOKEN, ANTHROPIC_API_KEY (or ANTHROPIC_KEY, as deploy-space.sh) and
    every `--secret NAME` from the local environment. Values are never printed."""
    out = {}
    for name in ("FLOCK_TOKEN", "HF_TOKEN"):
        if not env.get(name):
            raise SystemExit(f"{name} must be set in the environment")
        out[name] = env[name]
    anthropic = env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_KEY")
    if anthropic:
        out["ANTHROPIC_API_KEY"] = anthropic
    for name in extra:
        if not env.get(name):
            raise SystemExit(f"--secret {name}: not set in the environment")
        out[name] = env[name]
    return out


def _running_controllers(api, namespace: str, group: str) -> list:
    labels = {ROLE_LABEL: "controller", GROUP_LABEL: group}
    jobs = api.list_jobs(labels=labels, namespace=namespace)
    return [j for j in jobs if j.status and str(getattr(j.status.stage, "value", j.status.stage))
            in ("SCHEDULING", "RUNNING")]


def _cancel_and_wait(api, namespace: str, group: str, wait: float = 180, say=print) -> bool:
    """Cancel the group's running controllers (SIGTERM: final state push); True once none run."""
    for j in _running_controllers(api, namespace, group):
        say(f"cancelling controller {j.id} (final state push on SIGTERM)")
        api.cancel_job(job_id=j.id, namespace=namespace)
    deadline = time.monotonic() + wait
    while (old := _running_controllers(api, namespace, group)) and time.monotonic() < deadline:
        time.sleep(5)
    return not old


def down(api, namespace: str, group: str, state_url: str, *, dry_run: bool, env) -> int:
    """Stop the group's controller, then delete `controller/<group>/` from the bucket, so the
    next controller deployed in the group cannot restore a removed swarm."""
    from flock.worker.sync import HubBucketStore

    bucket = HubBucketStore(state_url, token=env.get("HF_TOKEN") or None)
    if dry_run:
        running = [j.id for j in _running_controllers(api, namespace, group)] or "none"
        print(f"would cancel: {running}; would delete {len(bucket.list(''))} files "
              f"under {state_url}")
        return 0
    if not _cancel_and_wait(api, namespace, group):
        print("controller still running after 180 s; state left in place", file=sys.stderr)
        return 1
    files = bucket.list("")  # after the final push
    if files:
        bucket.delete(files)
    print(f"deleted {len(files)} state files under {state_url}")
    return 0


def main(argv: list[str] | None = None, env: Mapping[str, str] = os.environ) -> int:
    p = argparse.ArgumentParser(prog="deploy-job.sh", description=__doc__.split("\n\n")[0])
    p.add_argument("wheel_url", nargs="?",
                   help="hf://buckets/... wheel from scripts/publish-wheel.sh (launch)")
    p.add_argument("--down", action="store_true",
                   help="cancel the group's controller and delete its bucket state")
    p.add_argument("--flavor", default="cpu-upgrade")
    p.add_argument("--group", default=DEFAULT_GROUP, help="network group (workers join it)")
    p.add_argument("--timeout", default="48h", help="job timeout, e.g. 48h or 7d")
    p.add_argument("--namespace", help="HF namespace (default: whoami); specs must match it")
    p.add_argument("--state-url", help="default: <wheel bucket>/controller/<group>")
    p.add_argument("--secret", action="append", default=[], help="extra secret NAME (repeat)")
    p.add_argument("--replace", action="store_true",
                   help="cancel the group's running controller first (SIGTERM: final state push)")
    p.add_argument("--dry-run", action="store_true", help="print the launch, secret names only")
    a = p.parse_args(argv)
    if not a.wheel_url and not (a.down and a.state_url):
        p.error("a wheel url is required (or --down with --state-url)")
    state_url = a.state_url or default_state_url(a.wheel_url, a.group)

    from huggingface_hub import HfApi

    api = HfApi(token=env.get("HF_TOKEN") or None)
    namespace = a.namespace or api.whoami()["name"]
    if a.down:
        return down(api, namespace, a.group, state_url, dry_run=a.dry_run, env=env)
    secrets = collect_secrets(env, a.secret)
    kwargs = run_job_args(
        wheel_url=a.wheel_url, group=a.group, namespace=namespace, flavor=a.flavor,
        timeout=a.timeout, state_url=state_url,
        job_labels=env.get("JOB_LABELS", "experiment=flock-m1"), secrets=secrets,
    )
    if a.dry_run:
        shown = {**kwargs, "secrets": sorted(kwargs["secrets"]), "command": "<bootstrap>"}
        for k, v in shown.items():
            print(f"{k}: {v}")
        return 0
    old = _running_controllers(api, namespace, a.group)
    if old and not a.replace:
        ids = ", ".join(j.id for j in old)
        print(f"group {a.group!r} already has a running controller ({ids}). Two jobs claiming "
              "the alias would both receive worker traffic; pass --replace.", file=sys.stderr)
        return 1
    if old and not _cancel_and_wait(api, namespace, a.group):
        print("old controller still running after 180 s; not launching", file=sys.stderr)
        return 1
    job = api.run_job(**kwargs)
    print(f"controller job: {job.id} ({job.url})")
    print(f"API (HF token via the jobs proxy): https://{job.id}--{API_PORT}.hf.jobs")
    if "expose_public" in kwargs:
        print(f"status page (public, read-only):  https://{job.id}--{STATUS_PORT}.hf.jobs")
    else:
        print("status page: not exposed (this huggingface_hub has no expose_public)")
    print(f"workers: http://${{HF_NETWORK_GROUP_PREFIX}}{ALIAS}:{API_PORT} in group {a.group}")
    print(f"export FLOCK_URL=https://{job.id}--{API_PORT}.hf.jobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
