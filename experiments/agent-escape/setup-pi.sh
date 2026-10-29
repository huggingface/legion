#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
npm install --prefix .flock/pi-runtime @earendil-works/pi-coding-agent@1.2.0 --no-audit --no-fund
mkdir -p .flock/tools
node_path=$(command -v node)
node_path=$(readlink -f "$node_path")
printf '#!/bin/sh\nexec "%s" "%s" "$@"\n' "$node_path" "$PWD/.flock/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/cli.js" > .flock/tools/pi
chmod 700 .flock/tools/pi
uv pip install --python .venv/bin/python 'matplotlib==3.10.8'
.flock/tools/pi --version
