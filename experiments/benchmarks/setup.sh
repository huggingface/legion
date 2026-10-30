#!/bin/bash
# Install the isolated runtime and fetch only the pinned Terminal-Bench pilot task.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ ! -x .flock/deps/benchmarks-venv/bin/python ]]; then
  uv venv --python 3.12 .flock/deps/benchmarks-venv
fi
uv pip sync --python .flock/deps/benchmarks-venv/bin/python experiments/benchmarks/requirements.lock
.venv/bin/python - <<'PY'
import json
import subprocess
from pathlib import Path
source = json.loads(Path('experiments/benchmarks/terminal-source.json').read_text())
p = Path('.flock/deps/terminal-bench-2')
if not p.exists():
    p.mkdir(parents=True)
    for cmd in [['git', 'init'], ['git', 'remote', 'add', 'origin', source['git_url']],
                ['git', 'config', 'core.sparseCheckout', 'true']]:
        subprocess.run(cmd, cwd=p, check=True)
    (p / '.git/info/sparse-checkout').write_text('/' + source['path'] + '/\n')
    subprocess.run(['git', 'fetch', '--depth=1', 'origin', source['git_commit_id']], cwd=p, check=True)
    subprocess.run(['git', 'checkout', '--detach', 'FETCH_HEAD'], cwd=p, check=True)
from experiments.benchmarks.campaign import checkout_revision
checkout_revision(p, source['git_commit_id'])
print('Runtime and Terminal-Bench task prepared; no containers or inference launched.')
print('Agent Escape prerequisite: bash experiments/agent-escape/setup.sh')
PY
