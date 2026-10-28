#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
revision=03742c79f48646e0130886e18e3e79798cf8edd7
repo="$PWD/.flock/deps/agent-escape"
mkdir -p .flock/deps
if [[ ! -d "$repo/.git" ]]; then
  git clone https://github.com/AGI-Eval-Official/Repo-of-AgentEscapeBench.git "$repo"
fi
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo 'Benchmark checkout is modified; refusing to overwrite it.' >&2
  exit 1
fi
git -C "$repo" checkout --detach "$revision"
uv venv --allow-existing .flock/deps/agent-escape-venv
uv pip install --python .flock/deps/agent-escape-venv/bin/python -r experiments/agent-escape/requirements.txt
