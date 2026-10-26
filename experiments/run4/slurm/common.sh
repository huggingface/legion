#!/bin/bash
# common.sh: sourced by controller.sbatch, world.sbatch and worker.sbatch (never run on its own).
#
# It resolves RUN and the run directory, sources the secrets file (mode 600) and the run's
# non-secret config (run.env), sets umask 077 and defines log(), die(), require_var(),
# http_status() and the container helpers (CONTAINER_IMAGE, container_srun_args). The caller sets
# RUN before sourcing (from $1 or the environment). It runs on the HOST: the world and the worker
# then start their Python process in a container (srun --container-image), the controller does not.
#
# Nothing here is passed with `sbatch --export`: every value is read from files on /fsx inside
# the job (see README.md). Secrets are never echoed.

PRJ=/fsx/carlos_patino/prj-eliza
VENV=$PRJ/venv                       # built elsewhere: bin/flock, flock-controller, flock-worker, hf, python
REPO=$PRJ/flock                      # checkout of huggingface/flock, pinned run4 commit (read-only for the jobs)
RUNS=$PRJ/runs                       # one directory per run: $RUNS/<RUN>/{logs,launch,experiment,...}
SECRETS_FILE=$PRJ/secrets/secrets.env
SLURM_DIR=$PRJ/slurm
export PRJ VENV REPO RUNS SECRETS_FILE SLURM_DIR

umask 077

log() {  # one timestamped line on the job's stdout (the sbatch -o file on /fsx)
  printf '%s %s[%s]: %s\n' "$(date -u +%FT%TZ)" "${LOG_TAG:-$(basename "${BASH_SOURCE[-1]}")}" \
    "${SLURM_JOB_ID:-nojob}" "$*"
}
die() { log "FATAL: $*" >&2; exit 2; }

# require_var NAME...: fail fast, with the variable NAME only (never its value), when unset/empty.
require_var() {
  local n
  for n in "$@"; do
    [ -n "${!n:-}" ] || die "$n is empty or unset (secrets: $SECRETS_FILE; run config: ${RUN_ENV:-<run dir>/run.env})"
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
# SIGKILL of its task ("srun: forcing job termination"; a SIG_IGN set before srun does not stop it;
# tests/ctr-signal-test.slurm, job 255717). USR1/USR2 can go to srun (`kill -USR1 $SRUN_PID`): it
# forwards them to the task as they are (same test).
step_signal() {
  local sig=$1 steps s
  steps=$(timeout 5 squeue -h -s -j "$SLURM_JOB_ID" -o %i 2>/dev/null | grep -E "^${SLURM_JOB_ID}\.[0-9]+\$")
  [ -n "$steps" ] || { log "step_signal $sig: no running step found"; return 1; }
  for s in $steps; do
    timeout 10 scancel --signal="$sig" "$s" || log "scancel --signal=$sig $s failed"
  done
}

# ---------------------------------------------------------------- the run
[ -n "${RUN:-}" ] || die "RUN is not set: sbatch <script> <run>  (or RUN=<run> in the environment)"
case $RUN in *[!A-Za-z0-9._-]*|"") die "RUN=$RUN: use letters, digits, . _ - only";; esac
RUN_DIR=$RUNS/$RUN
RUN_ENV=$RUN_DIR/run.env
HOSTS_DIR=$RUN_DIR/hosts   # controller.host, world.host: a directory, so a container can mount it ro and see updates
export RUN RUN_DIR RUN_ENV HOSTS_DIR
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/launch" "$HOSTS_DIR" || die "cannot create $RUN_DIR"

# publish_host NAME: write this node's hostname to <run>/hosts/NAME.host (tmp + rename) and keep
# <run>/NAME.host as a symlink to it (the path people and older notes use). A directory mount sees
# a rename; a bind mount of the file itself would keep showing the old inode.
publish_host() {
  local f=$HOSTS_DIR/$1.host
  printf '%s\n' "$(hostname)" > "$f.tmp" && mv -f "$f.tmp" "$f" || die "cannot write $f"
  ln -sfn "hosts/$1.host" "$RUN_DIR/$1.host" || die "cannot link $RUN_DIR/$1.host"
}

# ---------------------------------------------------------------- secrets and run config
# secrets.env defines HF_TOKEN (via $(cat ~/.cache/huggingface/token)), OPENAI_API_KEY,
# CHATBOT_BOARD_INVITE, CHATBOT_WORLD_TOKEN, FLOCK_TOKEN. run.env (optional, not secret) holds
# the run's endpoint, board, bucket and model settings: see run.env.example.
source_env_600 "$SECRETS_FILE"
if [ -f "$RUN_ENV" ]; then
  source_env_600 "$RUN_ENV"
fi

# The world job only (LOG_TAG=world, set by world.sbatch before sourcing this file) may run its own
# checkout and venv: WORLD_REPO / WORLD_VENV in run.env (defaults: the shared $REPO / $VENV). Done
# HERE, at run time, so a requeue of an already-submitted world job (Slurm replays the batch script
# stored at submit time) still picks them up: everything after this point and in world.sbatch /
# world-inner.sh (PATH, http_status, WORLD_SRC, the world.yaml source, the container mounts, the
# flock/hf binaries) reads $VENV / $REPO. The controller and the workers keep the shared ones.
if [ "${LOG_TAG:-}" = world ]; then
  VENV=${WORLD_VENV:-$VENV}
  REPO=${WORLD_REPO:-$REPO}
  VENV=${VENV%/} REPO=${REPO%/}
  export VENV REPO
fi
# The same for the controller job only (LOG_TAG=controller): CTL_VENV / CTL_REPO in run.env.
# Workers are unaffected: worker.sbatch sources this file, which resets VENV/REPO to the shared
# ones above, and worker-inner.sh uses that $VENV; the FLOCK_VENV that controller.sbatch writes
# into the env files (FLOCK_SLURM_VENV=$VENV) is informational only.
if [ "${LOG_TAG:-}" = controller ]; then
  VENV=${CTL_VENV:-$VENV}
  REPO=${CTL_REPO:-$REPO}
  VENV=${VENV%/} REPO=${REPO%/}
  export VENV REPO
fi

# Pin workers separately from the controller and world.
if [ "${LOG_TAG:-}" = worker ]; then
  VENV=${WORKER_VENV:-$VENV}
  REPO=${WORKER_REPO:-$REPO}
  export VENV REPO
fi
# A dedicated dataset-write token is used only by the world process.
# Workers/controller retain the bucket-scoped HF_TOKEN for sync.
if [ "${LOG_TAG:-}" != world ]; then
  unset CHATBOT_REPO_TOKEN
fi
export CHATBOT_REPO=${CHATBOT_REPO:-cmpatino/chatbot-swarm}

# The venv first on PATH: subprocesses started by flock (`flock` CLI inside tools, `hf` from the
# world's sync loop, `python -I -m flock.tools.runner`) must find the same install. Scripts still
# call $VENV/bin/<cmd> directly.
export PATH="$VENV/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
# Never let huggingface_hub fall back to ~/.cache/huggingface (token, xet chunk cache): a
# job-private dir on the host (/tmp is per job here; the job id is in the path anyway). The
# controller sets its own; the containers set HF_HOME=/work/hf (world-inner.sh, worker-inner.sh).
export HF_HOME=/tmp/hf-${SLURM_JOB_ID:-nojob}

# ---------------------------------------------------------------- containers (world and worker only)
# The world's and the worker's Python run in an Enroot/Pyxis container: home not mounted, /fsx
# visible only as the read-only mounts below, files written on the container root (/work, on the
# node's disk, deleted with the container). The controller stays on the host: it needs sbatch,
# squeue, sacct and scancel, which do not work in the container (no Slurm client, munge socket).
# Do not use /tmp inside the container for data: it is a tmpfs counted against the job memory.
CONTAINER_IMAGE=${CONTAINER_IMAGE:-$PRJ/images/pi-worker-node22.sqsh}   # run.env may override it
# select_worker_image TYPE (worker.sbatch, after the env file): the worker's image is
# CONTAINER_IMAGE_<TYPE> from run.env (TYPE with - and . as _, e.g. CONTAINER_IMAGE_agent_pi), else
# FLOCK_CONTAINER_IMAGE (the env file; flock's Slurm client writes it, one image for all types, when
# compute.container_image / FLOCK_SLURM_CONTAINER_IMAGE is set), else CONTAINER_IMAGE above.
select_worker_image() {
  local t=${1//[-.]/_} v
  case $t in *[!A-Za-z0-9_]*|"") ;; *) v=CONTAINER_IMAGE_$t; [ -n "${!v:-}" ] && { CONTAINER_IMAGE=${!v}; return; } ;; esac
  CONTAINER_IMAGE=${FLOCK_CONTAINER_IMAGE:-$CONTAINER_IMAGE}
}
UV_PYTHON_DIR=/fsx/carlos_patino/uv/python   # venv/bin/python links here (uv-managed, not ~/.local/share/uv)

# container_mounts [DIR...]: the --container-mounts list. The venv, the uv Python, the flock repo
# and this slurm dir (world-inner.sh, worker-inner.sh), plus each DIR, all read-only at their own path.
container_mounts() {
  local m="" d
  for d in "$VENV" "$UV_PYTHON_DIR" "$REPO" "$SLURM_DIR" "$@"; do
    [ -e "$d" ] || die "container mount source $d does not exist"
    m+="${m:+,}$d:$d:ro"
  done
  printf '%s' "$m"
}

# container_srun_args [DIR...]: sets the array CTR_SRUN to the srun flags of the one container
# task (same flags as the probe, tests/enroot-probe.slurm). Use: srun "${CTR_SRUN[@]}" CMD... &
# The batch script's environment (secrets included) passes into the container: no --container-env.
# Pyxis prints a harmless "[Auto Resume] Failed to initialize the logger." line at start.
container_srun_args() {
  [ -f "$CONTAINER_IMAGE" ] || die "CONTAINER_IMAGE=$CONTAINER_IMAGE: no such file (absolute path to a .sqsh)"
  local mounts
  mounts=$(container_mounts "$@") || exit 2
  # shellcheck disable=SC2034  # read by world.sbatch / worker.sbatch
  CTR_SRUN=(--ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --kill-on-bad-exit=1
            --container-image="$CONTAINER_IMAGE" --no-container-mount-home
            --container-mounts="$mounts" --container-workdir=/tmp)
}

# The venv must be complete before any job starts (it is built in parallel to these scripts).
for _cmd in python flock flock-controller flock-worker hf; do
  [ -x "$VENV/bin/$_cmd" ] || die "$VENV/bin/$_cmd is missing: the venv is not ready"
done
unset _cmd

log "run=$RUN run_dir=$RUN_DIR node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}MB venv=$VENV repo=$REPO"
