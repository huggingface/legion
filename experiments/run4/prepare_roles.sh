#!/bin/bash
# Prepare code and environments only. No services, containers or jobs are started.
# Usage: bash prepare_roles.sh <full-commit-sha> [--project DIR] [--repo DIR] [--apply]
set -euo pipefail
commit=${1:?usage: prepare_roles.sh FULL_SHA [--project DIR] [--repo DIR] [--apply]}
shift
project=/fsx/carlos_patino/prj-eliza
source_repo=
apply=0
while [ "$#" -gt 0 ]; do
  case $1 in
    --project) project=$2; shift 2;;
    --repo) source_repo=$2; shift 2;;
    --apply) apply=1; shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
case $commit in *[!0-9a-f]*|'') echo "use a full 40-character commit SHA" >&2; exit 2;; esac
[ "${#commit}" = 40 ] || { echo "use a full 40-character commit SHA" >&2; exit 2; }
project=${project%/}
source_repo=${source_repo:-$project/flock}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/fsx/carlos_patino/uv/python}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/fsx/carlos_patino/uv/cache}
run() {
  printf '%q ' "$@"; printf '\n'
  if [ "$apply" = 1 ]; then "$@"; fi
}
# Inspect every existing role before changing any of them. This script never moves
# an existing checkout to a new revision or overwrites another operator's work.
if [ "$apply" = 1 ]; then
  git -C "$source_repo" cat-file -e "$commit^{commit}"
  for role in ctl world worker; do
    checkout=$project/flock-run4-$role
    if [ -e "$checkout" ]; then
      [ "$(git -C "$checkout" rev-parse --show-toplevel)" = "$checkout" ] \
        && [ "$(git -C "$checkout" rev-parse HEAD)" = "$commit" ] \
        && [ -z "$(git -C "$checkout" status --porcelain)" ] \
        || { echo "refusing to change existing checkout $checkout" >&2; exit 2; }
    fi
  done
fi
for role in ctl world worker; do
  checkout=$project/flock-run4-$role
  venv=$project/venv-run4-$role
  if [ ! -e "$checkout" ]; then
    run git -C "$source_repo" worktree add --detach "$checkout" "$commit"
  fi
  if [ ! -x "$venv/bin/python" ]; then
    run uv venv --managed-python --python 3.12 "$venv"
  fi
  run env "UV_PROJECT_ENVIRONMENT=$venv" uv sync --frozen --all-extras --project "$checkout"
  if [ "$apply" = 1 ]; then
    "$venv/bin/python" -c 'import flock, pytest, ruff'
    for binary in flock flock-controller flock-worker hf pytest ruff; do
      [ -x "$venv/bin/$binary" ] || { echo "missing $venv/bin/$binary" >&2; exit 2; }
    done
  fi
done
printf '\nRole settings for run.env (no secrets):\n'
for role in ctl world worker; do
  key=${role^^}
  printf '%s_REPO=%q\n%s_VENV=%q\n' "$key" "$project/flock-run4-$role" "$key" "$project/venv-run4-$role"
done
[ "$apply" = 1 ] || echo 'Plan only; --apply creates code checkouts and environments. Neither mode submits jobs.'
