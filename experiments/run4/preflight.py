"""Read-only run4 preparation checks; --online queries metadata, never inference or jobs."""

import argparse
import json
import os
from pathlib import Path

import httpx
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

try:  # huggingface_hub >= 2 raises errors based on its own httpx fork
    from httpx2 import HTTPError as HTTPX2Error
except ImportError:  # pragma: no cover - older huggingface_hub
    HTTPX2Error = httpx.HTTPError

HUB_ERRORS = (HfHubHTTPError, HTTPX2Error, httpx.HTTPError, OSError, ValueError, KeyError)

from flock.worker.sync import parse_bucket


def world_archive(env):
    """The bucket path world.sbatch restores from: $CHATBOT_BUCKET or $RUN3_BUCKET/$RUN."""
    bucket = (env.get("CHATBOT_BUCKET") or "").rstrip("/")
    if not bucket:
        base, run = (env.get("RUN3_BUCKET") or "").rstrip("/"), env.get("RUN", "")
        if not base or not run:
            raise ValueError("set CHATBOT_BUCKET, or RUN3_BUCKET and RUN, as the world does")
        bucket = f"{base}/{run}"
    bucket_id, prefix = parse_bucket(bucket)
    return bucket_id, prefix + "world.tar.gz"


def check(env, *, online=False, get=httpx.get, api=None):
    results = []

    def record(name, ok, detail):
        results.append({"check": name, "ok": ok, "detail": detail})

    for role in ("CTL", "WORLD", "WORKER"):
        repo, venv = env.get(role + "_REPO"), env.get(role + "_VENV")
        if not repo or not venv:
            record(role.lower(), False, "set the role's REPO and VENV paths")
            continue
        binaries = ["python", "flock", "hf"] + (["pytest", "ruff"] if role == "WORKER" else [])
        ok = (Path(repo, "experiments/run4/spec.yaml").is_file()
              and all(os.access(Path(venv, "bin", binary), os.X_OK) for binary in binaries))
        record(role.lower(), ok, "role checkout and required executables")
    for key in ("HF_TOKEN", "OPENAI_API_KEY", "CHATBOT_BOARD_INVITE", "CHATBOT_WORLD_TOKEN", "FLOCK_TOKEN"):
        record(key, bool(env.get(key)), "set" if env.get(key) else "missing; value not displayed")
    if not online:
        return results
    urls = [u.strip().rstrip("/") for u in
            (env.get("OPENAI_BASE_URLS") or env.get("OPENAI_BASE_URL", "")).split(",") if u.strip()]
    record("model_endpoints", bool(urls), "configure every serving URL; node count must be verified by the operator")
    metrics = [u.strip() for u in env.get("OPENAI_METRICS_URLS", "").split(",") if u.strip()]
    for i, url in enumerate(urls):
        try:
            response = get(url + "/models", headers={"Authorization": "Bearer " + env.get("OPENAI_API_KEY", "")}, timeout=10)
            response.raise_for_status()
            ids = [model.get("id") for model in response.json().get("data", [])]
            record(f"model_{i+1}", env.get("MODEL_WORKER") in ids, "served model id matches MODEL_WORKER")
        except (httpx.HTTPError, ValueError, KeyError):
            record(f"model_{i+1}", False, "metadata request failed; response/credentials omitted")
        try:
            metric_url = metrics[i] if i < len(metrics) else url.removesuffix("/v1") + "/metrics"
            response = get(metric_url, timeout=10)
            response.raise_for_status()
            record(f"metrics_{i+1}", "vllm:num_requests_running" in response.text
                   and "vllm:num_requests_waiting" in response.text, "request metrics reachable")
        except httpx.HTTPError:
            record(f"metrics_{i+1}", False, "metrics request failed")
    try:
        url = env.get("CHATBOT_BOARD_URL", "https://cmpatino-msgboard-chatbot.hf.space").rstrip("/")
        response = get(url + "/v1/guide", timeout=10)
        record("board", response.status_code in (200, 401),
               "reachability only; invite validity, 429 load and DM delivery need the live smoke")
    except httpx.HTTPError:
        record("board", False, "board request failed")
    try:
        api = api or HfApi(token=env.get("HF_TOKEN") or False)
        api.repo_info(env.get("CHATBOT_REPO", "cmpatino/chatbot-swarm"), repo_type="dataset")
        bucket_id, archive = world_archive(env)
        list(api.get_bucket_paths_info(bucket_id, [archive]))
        record("hub_metadata", True, f"dataset/bucket metadata reachable ({archive}); "
               "this does not prove write scope")
    except HUB_ERRORS as exc:
        detail = str(exc) if isinstance(exc, ValueError) and "CHATBOT_BUCKET" in str(exc) else \
            "metadata request failed; inspect token scopes without logging its value"
        record("hub_metadata", False, detail)
    except Exception:  # noqa: BLE001 - never crash preflight or echo a possibly secret message
        record("hub_metadata", False, "metadata check failed unexpectedly; details omitted")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    checks = check(os.environ, online=args.online)
    print(json.dumps({"checks": checks, "manual_checks": [
        "HF_TOKEN has only the intended bucket and dataset grants",
        "native Hub PR write/update/review/merge lifecycle",
        "board/NAT rate test, capacity calibration and restart rehearsal",
        "at least two actual model nodes and enough remaining model-server lifetime"]}, indent=2))
    raise SystemExit(0 if all(item["ok"] for item in checks) else 1)
