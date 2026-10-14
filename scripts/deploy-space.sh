#!/usr/bin/env bash
# Deploy the controller to a public Docker Space and configure it.
#   scripts/deploy-space.sh <wheel-url> [space-id]     (default space cmpatino/flock)
# JOB_LABELS sets the deployment-wide worker job labels (default experiment=flock-m1).
# Needs HF_TOKEN, ANTHROPIC_KEY and FLOCK_TOKEN in the environment; values are never
# printed. The Space is public: the API is protected by the bearer token (a private Space would
# need an HF bearer on every request, which collides with our Authorization header).
set -euo pipefail
WHEEL_URL=${1:?wheel url from scripts/publish-wheel.sh}
SPACE=${2:-cmpatino/flock}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp "$ROOT/Dockerfile.controller" "$STAGE/Dockerfile"
cp "$ROOT/pyproject.toml" "$ROOT/uv.lock" "$STAGE/"
cp -r "$ROOT/src" "$STAGE/src"
find "$STAGE/src" -name __pycache__ -prune -exec rm -rf {} +
cat > "$STAGE/README.md" <<MD
---
title: flock
emoji: 🐝
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

flock controller ($(git -C "$ROOT" rev-parse --short HEAD)). Source:
https://github.com/huggingface/flock. The API needs the swarm bearer token;
\`/\` (status table) and \`/healthz\` are open.
MD
python3 - "$SPACE" "$STAGE" "$WHEEL_URL" "${JOB_LABELS:-experiment=flock-m1}" <<'PY'
import os, sys
from huggingface_hub import HfApi
space, stage, wheel, job_labels = sys.argv[1:]
api = HfApi()
api.create_repo(space, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
for key, env in (("FLOCK_TOKEN", "FLOCK_TOKEN"), ("HF_TOKEN", "HF_TOKEN"),
                 ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY")):
    api.add_space_secret(space, key, os.environ[env])
for key, value in {
    "FLOCK_WORKER_BOOTSTRAP": "1",
    "FLOCK_WHEEL_URL": wheel,
    "FLOCK_JOB_LABELS": job_labels,
}.items():
    api.add_space_variable(space, key, value)
info = api.upload_folder(repo_id=space, repo_type="space", folder_path=stage,
                         commit_message=f"deploy controller ({os.path.basename(os.path.dirname(wheel))})",
                         delete_patterns=["src/**"])
print(f"deployed {space}: {info.oid if hasattr(info, 'oid') else info}")
PY
