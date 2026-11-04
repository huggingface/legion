#!/bin/bash
# common.sh: sourced by controller.sbatch, world.sbatch and worker.sbatch (never run on its own).
# `flock init` copies the four files to <run dir>/slurm/ (docs/slurm.md "Deploying"); each job
# sources <run dir>/slurm/common.sh, so a run keeps its own copy.
#
# The caller sets RUN_DIR (the run directory, an absolute path) before sourcing. This file sources
# the run's cluster profile (cluster.env, copied by `flock init --profile`), the cluster's secrets
# file when the profile names one (SECRETS_FILE, mode 600) and the run's config (run.env, mode 600;
# later wins), sets umask 077 and defines log(), die(), require_var(), http_status(),
# publish_host(), free_port() and the container helpers (CONTAINER_IMAGE,
# container_srun_args). It runs on the HOST: a worker then starts in a container when one is set.
#
# Nothing here is passed with `sbatch --export`: every value is read from files in the run
# directory inside the job. Secrets are never echoed. cluster.env names VENV (bin/flock,
# flock-controller, flock-worker, python), optionally REPO (a flock checkout), SECRETS_FILE,
# CONTAINER_IMAGE and CONTAINER_MOUNTS.

umask 077

log() {  # one timestamped line on the job's stdout (the sbatch -o file)
  printf '%s %s[%s]: %s\n' "$(date -u +%FT%TZ)" "${LOG_TAG:-$(basename "${BASH_SOURCE[-1]}")}" \
    "${SLURM_JOB_ID:-nojob}" "$*"
}
die() { log "FATAL: $*" >&2; exit 2; }

# require_var NAME...: fail fast, with the variable NAME only (never its value), when unset/empty.
require_var() {
  local n
  for n in "$@"; do
    [ -n "${!n:-}" ] || die "$n is empty or unset (run config: $RUN_ENV${SECRETS_FILE:+; secrets: $SECRETS_FILE})"
  done
}

# source_env_600 FILE: `set -a; . FILE; set +a` after checking it is ours and mode 600.
source_env_600() {
  local f=$1 mode owner
  [ -f "$f" ] || die "$f: no such file"
  mode=$(stat -c %a "$f") || die "$f: stat failed"
  owner=$(stat -c %U "$f") || die "$f: stat failed"
  [ "$owner" = "$(id -un)" ] || die "$f: owned by $owner, not $(id -un)"
  [ "$mode" = 600 ] || [ "$mode" = 400 ] || die "$f: mode $mode, must be 600 (chmod 600 '$f')"
  set -a
  # shellcheck disable=SC1090
  . "$f"
  set +a
}

# http_status URL [HEADER_NAME ENV_VAR]: prints the HTTP status code (000 when unreachable).
# A header value is taken from the environment variable ENV_VAR, so no secret reaches a
# command line (visible in /proc) or the log.
http_status() {
  "$VENV/bin/python" -I - "$@" <<'PY' 2>/dev/null || echo 000
import os, sys, httpx
url = sys.argv[1]
headers = {}
if len(sys.argv) >= 4:
    headers[sys.argv[2]] = os.environ.get(sys.argv[3], "")
try:
    print(httpx.get(url, headers=headers, timeout=5.0).status_code)
except Exception:
    print("000")
PY
}

# step_signal SIG: send SIG to the processes of this job's srun step(s) through slurmd
# (`scancel --signal=SIG <job>.<step>`), NOT through srun: srun turns a SIGTERM it receives into a
# SIGKILL of its task ("srun: forcing job termination"; a SIG_IGN set before srun does not stop
# it). USR1/USR2 can go to srun (`kill -USR1 $SRUN_PID`): it forwards them to the task as they are.
step_signal() {
  local sig=$1 steps s
  steps=$(timeout 5 squeue -h -s -j "$SLURM_JOB_ID" -o %i 2>/dev/null | grep -E "^${SLURM_JOB_ID}\.[0-9]+\$")
  [ -n "$steps" ] || { log "step_signal $sig: no running step found"; return 1; }
  for s in $steps; do
    timeout 10 scancel --signal="$sig" "$s" || log "scancel --signal=$sig $s failed"
  done
}

# ---------------------------------------------------------------- the run
case ${RUN_DIR:-} in /*) ;; *) die "RUN_DIR=${RUN_DIR:-}: pass the run directory as an absolute path";; esac
RUN_DIR=${RUN_DIR%/}
RUN=${RUN_DIR##*/}
case $RUN in *[!A-Za-z0-9._-]*|"") die "run directory name $RUN: use letters, digits, . _ - only";; esac
SLURM_DIR=$RUN_DIR/slurm   # these scripts (mounted read-only in a container)
RUN_ENV=$RUN_DIR/run.env
HOSTS_DIR=$RUN_DIR/hosts   # *.host: a directory, so a container can mount it ro and see updates
SPEC=$RUN_DIR/experiment/spec.yaml   # the experiment folder's copy (`flock init`)
export RUN RUN_DIR SLURM_DIR RUN_ENV HOSTS_DIR SPEC
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/launch" "$HOSTS_DIR" || die "cannot create $RUN_DIR"

# publish_host NAME PORT: <run>/hosts/NAME.host = `<this node's hostname>:PORT` (tmp + rename: a
# reader, or a directory mount, sees the old or the new file)
publish_host() {
  local f=$HOSTS_DIR/$1.host
  printf '%s\n' "$(hostname)${2:+:$2}" > "$f.tmp" && mv -f "$f.tmp" "$f" || die "cannot write $f"
}

# free_port: a TCP port free on this node now, so the controller and the world can share a node.
# Another process may take it before the server binds (rare): controller.sbatch retries once with a
# new port; world.sbatch's crash requeue (a fast failure) starts again with a new one.
free_port() {
  "$VENV/bin/python" -I -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])'
}

# ---------------------------------------------------------------- cluster profile, secrets and run config
# cluster.env (not secret): the cluster's values (docs/slurm.md "Cluster profile"). SLURM_PARTITION
# is not exported: srun reads it as its --partition, and a step must stay in its job's partition.
if [ -f "$RUN_DIR/cluster.env" ]; then
  # sourced as shell into jobs holding secrets: ours and not writable by anyone else
  _owner=$(stat -c %U "$RUN_DIR/cluster.env") && _mode=$(stat -c %a "$RUN_DIR/cluster.env") \
    || die "$RUN_DIR/cluster.env: stat failed"
  [ "$_owner" = "$(id -un)" ] || die "$RUN_DIR/cluster.env: owned by $_owner, not $(id -un)"
  (( 8#$_mode & 8#022 )) && die "$RUN_DIR/cluster.env: mode $_mode is group/world-writable (chmod go-w)"
  unset _owner _mode
  set -a
  # shellcheck disable=SC1091
  . "$RUN_DIR/cluster.env"
  set +a
  export -n SLURM_PARTITION 2>/dev/null
fi
[ -z "${SECRETS_FILE:-}" ] || source_env_600 "$SECRETS_FILE"
[ ! -f "$RUN_ENV" ] || source_env_600 "$RUN_ENV"   # `flock init`: the spec's ${VAR}s, its secrets, per-run tokens
require_var VENV

# A role's own venv / checkout (run.env: CTL_VENV, WORLD_VENV, WORKER_VENV, and *_REPO); default
# VENV / REPO. Read at run time, so a requeued job (Slurm replays the script stored at submit
# time) picks a change up too.
case ${LOG_TAG:-} in
  controller) VENV=${CTL_VENV:-$VENV} REPO=${CTL_REPO:-${REPO:-}};;
  world) VENV=${WORLD_VENV:-$VENV} REPO=${WORLD_REPO:-${REPO:-}};;
  worker) VENV=${WORKER_VENV:-$VENV} REPO=${WORKER_REPO:-${REPO:-}};;
esac
VENV=${VENV%/} REPO=${REPO:-} REPO=${REPO%/}
export VENV REPO

# The venv first on PATH: subprocesses started by flock (`flock` CLI inside tools,
# `python -I -m flock.tools.runner`) must find the same install. Scripts still call
# $VENV/bin/<cmd> directly.
export PATH="$VENV/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
# Never let huggingface_hub fall back to ~/.cache/huggingface (token, xet chunk cache): a
# job-private dir on the host. The controller sets its own; a container sets HF_HOME=/work/hf.
export HF_HOME=/tmp/hf-${SLURM_JOB_ID:-nojob}

# ---------------------------------------------------------------- containers (workers only)
# A worker's Python may run in an Enroot/Pyxis container: home not mounted, the shared disk
# visible only as the read-only mounts below, files written on the container root (/work, on the
# node's disk, deleted with the container). The controller and the world stay on the host.
# Do not use /tmp inside the container for data: it is a tmpfs counted against the job memory.
CONTAINER_IMAGE=${CONTAINER_IMAGE:-}   # cluster.env; run.env may override it
# select_worker_image TYPE (worker.sbatch, after the env file): the worker's image is
# CONTAINER_IMAGE_<TYPE> from run.env (TYPE with - and . as _, e.g. CONTAINER_IMAGE_agent_pi), else
# FLOCK_CONTAINER_IMAGE (the env file; flock's Slurm client writes it when
# compute.container_image / FLOCK_SLURM_CONTAINER_IMAGE is set), else CONTAINER_IMAGE above.
select_worker_image() {
  local t=${1//[-.]/_} v
  case $t in *[!A-Za-z0-9_]*|"") ;; *) v=CONTAINER_IMAGE_$t; [ -n "${!v:-}" ] && { CONTAINER_IMAGE=${!v}; return; } ;; esac
  CONTAINER_IMAGE=${FLOCK_CONTAINER_IMAGE:-$CONTAINER_IMAGE}
}
# container_mounts [DIR...]: the --container-mounts list: the venv, the flock repo, this slurm dir,
# each CONTAINER_MOUNTS entry (cluster.env, comma-separated: e.g. the uv Python dir the venv's
# bin/python links to; `flock init` checks it is mounted) and each DIR, read-only at their own path.
container_mounts() {
  local m="" d
  local -a profile
  IFS=, read -ra profile <<< "${CONTAINER_MOUNTS:-}"
  for d in "$VENV" "$REPO" "$SLURM_DIR" ${profile[@]+"${profile[@]}"} "$@"; do
    d=${d%%:*} d=${d%/}
    [ -n "$d" ] || continue
    case ",$m," in *",$d:$d:ro,"*) continue;; esac
    [ -e "$d" ] || die "container mount source $d does not exist"
    m+="${m:+,}$d:$d:ro"
  done
  printf '%s' "$m"
}

# container_srun_args [DIR...]: sets the array CTR_SRUN to the srun flags of the one container
# task. Use: srun "${CTR_SRUN[@]}" CMD... &
# The batch script's environment (secrets included) passes into the container: no --container-env.
# Pyxis prints a harmless "[Auto Resume] Failed to initialize the logger." line at start.
container_srun_args() {
  [ -f "$CONTAINER_IMAGE" ] || die "CONTAINER_IMAGE=$CONTAINER_IMAGE: no such file (absolute path to a .sqsh)"
  local mounts
  mounts=$(container_mounts "$@") || exit 2
  # shellcheck disable=SC2034  # read by worker.sbatch
  CTR_SRUN=(--ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --kill-on-bad-exit=1
            --container-image="$CONTAINER_IMAGE" --no-container-mount-home
            --container-mounts="$mounts" --container-workdir=/tmp)
}

# The venv must be complete before any job starts.
for _cmd in python flock flock-controller flock-worker; do
  [ -x "$VENV/bin/$_cmd" ] || die "$VENV/bin/$_cmd is missing: the venv is not ready"
done
unset _cmd

log "run=$RUN run_dir=$RUN_DIR node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}MB venv=$VENV repo=${REPO:-<none>}"
