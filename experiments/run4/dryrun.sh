#!/bin/bash
# Offline only: two Pi processes, fake model/board/Hub and local git; no job submissions.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
command -v pi >/dev/null || [ -x "${FLOCK_PI_CLI:-}" ] || {
  echo "Install Pi 1.2.0 or set FLOCK_PI_CLI before running this smoke." >&2
  exit 2
}
ruff check .
pytest -q -o "cache_dir=${TMPDIR:-/tmp}/run4-pytest-cache" \
  tests/test_run4_experiment.py tests/test_run4_cluster_review.py tests/test_isolation.py tests/test_run4_pi_smoke.py tests/test_run4_spawner.py \
  tests/test_chatbot4.py tests/test_comms_extended.py tests/test_worker_pi.py \
  tests/test_loop_worker.py tests/test_loop_controller.py tests/test_loop_notices.py
