#!/usr/bin/env bash
# Deploy the controller to a public Docker Space and configure it.
#   scripts/deploy-space.sh <wheel-url> [space-id]     (default space cmpatino/swarm-harness)
# Needs HF_TOKEN, ANTHROPIC_KEY and SWARM_HARNESS_TOKEN in the environment; values are never
# printed. The Space is public: the API is protected by the bearer token (a private Space would
# need an HF bearer on every request, which collides with our Authorization header).
set -euo pipefail
WHEEL_URL=${1:?wheel url from scripts/publish-wheel.sh}
SPACE=${2:-cmpatino/swarm-harness}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp "$ROOT/Dockerfile.controller" "$STAGE/Dockerfile"
cp "$ROOT/pyproject.toml" "$ROOT/uv.lock" "$STAGE/"
cp -r "$ROOT/src" "$STAGE/src"
find "$STAGE/src" -name __pycache__ -prune -exec rm -rf {} +
cat > "$STAGE/README.md" <<MD
---
title: swarm-harness
emoji: 🐝
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

swarm-harness controller ($(git -C "$ROOT" rev-parse --short HEAD)). Source:
https://github.com/huggingface/swarm-harness. The API needs the swarm bearer token;
\`/\` (status table) and \`/healthz\` are open.
MD
python3 - "$SPACE" "$STAGE" "$WHEEL_URL" <<'PY'
import os, sys
from huggingface_hub import HfApi
space, stage, wheel = sys.argv[1:]
api = HfApi()
api.create_repo(space, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
for key, env in (("SWARM_HARNESS_TOKEN", "SWARM_HARNESS_TOKEN"), ("HF_TOKEN", "HF_TOKEN"),
                 ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY")):
    api.add_space_secret(space, key, os.environ[env])
for key, value in {
    "SWARM_HARNESS_WORKER_BOOTSTRAP": "1",
    "SWARM_HARNESS_WHEEL_URL": wheel,
    "SWARM_HARNESS_JOB_LABELS": "experiment=swarm-harness-m1",
}.items():
    api.add_space_variable(space, key, value)
info = api.upload_folder(repo_id=space, repo_type="space", folder_path=stage,
                         commit_message=f"deploy controller ({os.path.basename(os.path.dirname(wheel))})",
                         delete_patterns=["src/**"])
print(f"deployed {space}: {info.oid if hasattr(info, 'oid') else info}")
PY
