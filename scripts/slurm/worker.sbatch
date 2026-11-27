#!/bin/bash
# worker.sbatch <env file>: one legion worker slot as a CPU Slurm job.
#
# Submitted BY THE CONTROLLER's Slurm jobs client (LEGION_COMPUTE=slurm; controller.sbatch points
# LEGION_SLURM_WORKER_SCRIPT at <run dir>/slurm/worker.sbatch), roughly as
#   sbatch --parsable [-p <partition>] -c <cpus> --mem=<mem> --time=<time> --no-requeue \
#          -J <worker id> --comment '<labels>' -o <run>/logs/<worker id>-%j.out \
#          <run>/slurm/worker.sbatch <run>/launch/<worker id>-<incarnation>.env
# The #SBATCH lines below are fallback defaults for a launch by hand (docs/slurm.md "The launch
# contract": the env file's format).
#
# - The run directory is the env file's <run>/launch/..; common.sh sources cluster.env, the
#   secrets file and run.env, then the env file (settings only, no secret values).
# - CTL_HOST is read from <run>/hosts/controller.host and exported BEFORE the env file is read, so
#   LEGION_URL='http://${CTL_HOST}:<port>' expands inside legion-worker (the file's host:port wins).
# - legion-worker gets the secrets the env file's comment names in a private 0600 file and never in
#   its exec environment, so /proc/<pid>/environ never shows them (docs/slurm.md "Worker secrets
#   file"). The file sits next to WORK_ROOT; legion-worker deletes it once read.
# - CONTAINER (an image from CONTAINER_IMAGE_<type> / LEGION_CONTAINER_IMAGE / CONTAINER_IMAGE):
#   legion-worker runs in an Enroot/Pyxis container through `srun ... worker.sbatch --inner`: home
#   not mounted, the shared disk read-only and limited to the venv, repo, slurm dir,
#   CONTAINER_MOUNTS and <run>/hosts. Agent files go to /work/legion/agents/work on the
#   container root (node disk, deleted with the container; not /tmp, a tmpfs counted against the
#   job memory). The paths carry no worker id, so a slot relaunched under a new id restores its
#   agents to the same absolute paths.
#   Without an image legion-worker runs here, in this job's shell (exec), with the env file's paths.
# - LEGION_HARNESS=harbor (the env file): each agent is a Harbor Trial in a rootless podman
#   container, which cannot run inside Enroot, so legion-worker runs here (never in CONTAINER) as
#   a child of this shell, with the standard rootless podman setup: the job's own
#   podman storage under /tmp, podman's health checks run by legion.harbor.podman_healthchecks,
#   the upstream git credential helper for huggingface.co (the token from a 0600 file there), and
#   at exit the job's containers stopped. Unverified on compute nodes. LEGION_RESULTS_DIR is
#   <run>/results: each Trial's result.json, config.json and verifier/, and the board export,
#   are copied there (shared disk), since WORK_ROOT goes with the node's /tmp.
# - Spot reclaim (container): the watcher sends SIGUSR1 to this shell, to srun AND to every process
#   of the step. legion-worker starts with USR1 ignored (SIG_IGN survives exec) until its own
#   handler (drain + sync) replaces it. The handler below forwards USR1 to the srun step, then
#   SIGTERM 3 s later if the step is still alive (through slurmd: common.sh step_signal), and
#   waits. NO requeue: the controller sees the job end and relaunches the slot. legion-worker may
#   see USR1 two or three times: its handler is idempotent.
# - A plain `scancel <id>` (the Slurm client's cancel) is fine: slurmd sends SIGTERM to
#   legion-worker directly. Never `kill -TERM $SRUN_PID`: srun answers that with a SIGKILL.
#
#SBATCH --job-name=legion-worker
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G              # 1792 MB per CPU (under a MaxMemPerCPU of 2000, which more would silently exceed)
#SBATCH --time=29:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@60    # 60 s before the time limit (DEADLINE + drain should end the worker long before)

set -uo pipefail  # no -e: the traps and the wait loop must keep running after a signal

# run_worker: make the agent dirs, write the secrets file, exec legion-worker without the secrets
# in its environment. Input: VENV WORK_ROOT SECRET_NAMES and the secrets.
run_worker() {
  local f v
  local -a unset_args=()
  umask 077
  mkdir -p "$WORK_ROOT" || { echo "worker.sbatch: cannot create $WORK_ROOT" >&2; exit 2; }
  f=$(mktemp "$(dirname "$WORK_ROOT")/legion-worker-secrets.XXXXXX") || exit 2
  for v in $SECRET_NAMES; do  # only the names that are set: an empty NAME= would shadow nothing useful
    [ -z "${!v:-}" ] || printf '%s=%s\n' "$v" "${!v}" >> "$f" || exit 2
    unset_args+=(-u "$v")
  done
  trap '' USR1  # until legion-worker's own handler
  # LEGION_WORLD_ADMIN_TOKEN (run.env): the world's /admin secret, never a worker's
  exec env "${unset_args[@]}" -u LEGION_WORLD_ADMIN_TOKEN LEGION_WORKER_SECRETS_FILE="$f" \
    LEGION_WORKER_SECRETS_FILE_UNLINK=1 "$VENV/bin/legion-worker"
}

# podman_cleanup (EXIT, harbor): stop the health checks and this job's containers (the wrapper's
# --root: never the user's storage), quickly: Slurm's KillWait (e.g. 30 s) runs from the
# scancel, the worker's drain included.
# shortcut: the storage stays for the job's own /tmp (e.g. a per-job tmpfs) to take;
# on a cluster with a shared /tmp, remove it here with `podman unshare rm -rf` (sub-uid files)
podman_cleanup() {
  local p=$PODMAN_LOCAL/bin/podman  # by path: a plain `podman` here would be the user's storage
  [ -z "${HEALTH_PID:-}" ] || kill "$HEALTH_PID" 2>/dev/null
  [ -x "$p" ] || return 0
  timeout 10 "$p" stop --all --time 5 >/dev/null 2>&1
  timeout 10 "$p" rm --all --force >/dev/null 2>&1
}

if [ "${1:-}" = --inner ]; then  # in the container (srun below): the batch shell's environment came along
  export HOME=/work/home HF_HOME=/work/hf TMPDIR=/work/tmp
  export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
  mkdir -p "$HOME" "$HF_HOME" "$TMPDIR" && cd /work || exit 2
  run_worker
fi

ENVFILE=${1:?usage: worker.sbatch <env file>  (<run dir>/launch/<worker id>-<incarnation>.env)}
[ -f "$ENVFILE" ] || { echo "worker.sbatch: $ENVFILE: no such file" >&2; exit 2; }
RUN_DIR=$(cd "$(dirname "$ENVFILE")/.." && pwd) || { echo "worker.sbatch: $ENVFILE: no run directory" >&2; exit 2; }
# shellcheck disable=SC2034  # LOG_TAG is read by common.sh
LOG_TAG=worker
# shellcheck source=common.sh
. "$RUN_DIR/slurm/common.sh" || exit 2

# ---------------------------------------------------------------- controller address, then the worker env
CTL_HOST_FILE=$HOSTS_DIR/controller.host
[ -s "$CTL_HOST_FILE" ] || die "$CTL_HOST_FILE is missing: is controller.sbatch running?"
CTL_HOST=$(tr -d '[:space:]' < "$CTL_HOST_FILE")
export CTL_HOST
source_env_600 "$ENVFILE"   # the controller's settings for this slot (no secret values)
# The secrets this worker expects: the env file's comment (slurm.py env_file_text)
SECRET_NAMES=$(sed -n 's/^# Secrets this worker expects[^:]*: //p' "$ENVFILE")
export SECRET_NAMES
# shellcheck disable=SC2086  # SECRET_NAMES: one name per word
require_var LEGION_URL WORKER_ID WORKER_INCARNATION SWARM TYPE DEADLINE DRAIN_WINDOW $SECRET_NAMES
case $WORKER_ID in *[!A-Za-z0-9._-]*|"") die "WORKER_ID=$WORKER_ID: unexpected characters";; esac
export SYNC_INTERVAL=${SYNC_INTERVAL:-180} HEARTBEAT_INTERVAL=${HEARTBEAT_INTERVAL:-30} LOG_LEVEL=${LOG_LEVEL:-INFO}
select_worker_image "$TYPE"        # per-type image (common.sh)
PODMAN_LOCAL=
[ "${LEGION_HARNESS:-}" != harbor ] || CONTAINER_IMAGE=   # podman cannot run inside Enroot
log "worker=$WORKER_ID incarnation=${WORKER_INCARNATION:0:8}... swarm=$SWARM type=$TYPE deadline=$DEADLINE ctl=$CTL_HOST container=${CONTAINER_IMAGE:-<none>} harness=${LEGION_HARNESS:-?}"
if [ "${LEGION_HARNESS:-}" = harbor ]; then
  # ---------------------------------------------------------------- harbor: rootless podman here
  export PODMAN_COMPOSE_PROVIDER=$VENV/bin/podman-compose
  export LEGION_RESULTS_DIR=$RUN_DIR/results  # Trial results and exports outlive WORK_ROOT
  [ -x "$PODMAN_COMPOSE_PROVIDER" ] || die "$PODMAN_COMPOSE_PROVIDER is missing: install legion in $VENV"
  PODMAN=$(command -v podman) || die "podman is not on PATH"
  PODMAN_LOCAL=$(mktemp -d /tmp/legion-podman.XXXXXX) || die "cannot create the podman storage dir"
  trap podman_cleanup EXIT
  # The controller may hand down its own node-local paths. Podman starts before the worker
  # process can recover, so give it a usable home and use this node's default temporary path.
  if [ ! -d "${HOME:-}" ] || [ ! -w "$HOME" ]; then
    export HOME=$PODMAN_LOCAL/home
    mkdir -p "$HOME" || die "cannot create this job's home"
  fi
  if [ -n "${TMPDIR:-}" ] && { [ ! -d "$TMPDIR" ] || [ ! -w "$TMPDIR" ]; }; then
    unset TMPDIR
  fi
  export XDG_RUNTIME_DIR=$PODMAN_LOCAL/runtime HF_XET_CACHE=$PODMAN_LOCAL/xet GIT_TERMINAL_PROMPT=0
  mkdir -p "$XDG_RUNTIME_DIR" "$PODMAN_LOCAL/bin" || die "cannot create $PODMAN_LOCAL"
  # `podman compose` prints a banner about its external provider on the stderr of every call,
  # `compose exec` included, where Harbor reads a command's output
  printf '[engine]\ncompose_warning_logs = false\n' > "$PODMAN_LOCAL/containers.conf" \
    || die "cannot write $PODMAN_LOCAL/containers.conf"
  export CONTAINERS_CONF_OVERRIDE=$PODMAN_LOCAL/containers.conf
  # Docker Hub through mirror.gcr.io (this run's slurm/registries.conf), unless the profile sets one
  export CONTAINERS_REGISTRIES_CONF=${CONTAINERS_REGISTRIES_CONF:-$SLURM_DIR/registries.conf}
  # Harbor clones task repos with git: HF_TOKEN for huggingface.co only (the upstream helper), read
  # from a 0600 file, as the worker's environment never holds it
  if [ -n "${HF_TOKEN:-}" ]; then
    printf '%s\n' "$HF_TOKEN" > "$PODMAN_LOCAL/hf-token" || die "cannot write $PODMAN_LOCAL/hf-token"
    git_slot=${GIT_CONFIG_COUNT:-0}
    export "GIT_CONFIG_KEY_${git_slot}=credential.https://huggingface.co.helper" \
      "GIT_CONFIG_VALUE_${git_slot}="
    git_slot=$((git_slot + 1))
    export "GIT_CONFIG_KEY_${git_slot}=credential.https://huggingface.co.helper" \
      "GIT_CONFIG_VALUE_${git_slot}=!f() { if [ \"\$1\" = get ]; then echo username=hf_user; printf password=; cat '$PODMAN_LOCAL/hf-token'; fi; }; f"
    export GIT_CONFIG_COUNT=$((git_slot + 1))
  fi
  # Explicit engine flags work on the cluster's Podman 4.9 and avoid user storage.
  cat > "$PODMAN_LOCAL/bin/podman" <<EOF
#!/bin/bash
exec "$PODMAN" --root "$PODMAN_LOCAL/graphroot" --runroot "$PODMAN_LOCAL/runroot" \
    --storage-driver overlay --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs \
    --cgroup-manager cgroupfs --events-backend file "\$@"
EOF
  # Netavark 1.4 tries systemd-run even when no user bus exists. Keep only its
  # DNS daemon in this Slurm job; delegate every other invocation unchanged.
  cat > "$PODMAN_LOCAL/bin/systemd-run" <<'EOF'
#!/bin/bash
if [[ "${1:-} ${2:-} ${3:-}" == "-q --scope --user" && "${4:-}" == /usr/lib/podman/aardvark-dns ]]; then
    shift 3
    exec "$@"
fi
exec /usr/bin/systemd-run "$@"
EOF
  chmod 700 "$PODMAN_LOCAL/bin/podman" "$PODMAN_LOCAL/bin/systemd-run" || exit 2
  export PATH="$PODMAN_LOCAL/bin:$PATH"
  # Prevent inherited remote-engine settings from bypassing this job's storage.
  unset CONTAINER_HOST CONTAINER_CONNECTION DOCKER_HOST
  [ "$(podman info --format '{{.Store.GraphRoot}}')" = "$PODMAN_LOCAL/graphroot" ] \
    || die "podman does not use this job's storage $PODMAN_LOCAL/graphroot"
elif [ -z "$CONTAINER_IMAGE" ]; then
  run_worker  # exec: legion-worker is this job's shell from here on (signals reach it directly)
else
  # ---------------------------------------------------------------- the container
  export WORK_ROOT=/work/legion/agents/work        # agents' workdirs WORK_ROOT/<agent>
  container_srun_args "$HOSTS_DIR"   # -> CTR_SRUN; <run>/hosts ro for LEGION_CTL_HOST_FILE
fi

SRUN_PID= ; STOPPING=0   # SRUN_PID: the srun step, or legion-worker itself (harbor)
# term_worker: srun turns a SIGTERM into a SIGKILL of its task, so a step gets it through slurmd
term_worker() {
  if [ -n "$PODMAN_LOCAL" ]; then kill -TERM "$SRUN_PID" 2>/dev/null; else step_signal TERM; fi
}
on_usr1() {  # Spot reclaim: the watcher cancels the job 30 s after this signal
  [ "$STOPPING" = usr1 ] && return 0
  STOPPING=usr1
  log "SIGUSR1 (Spot reclaim): forwarding to the legion-worker step (srun pid ${SRUN_PID:-<none>}), SIGTERM in 3 s; no requeue"
  [ -n "$SRUN_PID" ] || return 0
  kill -USR1 "$SRUN_PID" 2>/dev/null
  sleep 3
  if kill -0 "$SRUN_PID" 2>/dev/null; then
    log "legion-worker step still alive 3 s after USR1: SIGTERM"
    term_worker
  fi
}
on_term() {
  [ "$STOPPING" = 0 ] || return 0
  STOPPING=term
  log "SIGTERM (scancel by the controller, preemption or time limit): forwarding to legion-worker"
  [ -n "$SRUN_PID" ] && term_worker
}
trap on_usr1 USR1
trap on_term TERM INT

[ "$STOPPING" = 0 ] || { log "signal before legion-worker started: exit"; exit 0; }
trap '' USR1  # srun starts with USR1 ignored, like every child here
if [ -n "$PODMAN_LOCAL" ]; then
  run_worker &  # exec in a subshell: SRUN_PID is legion-worker
  SRUN_PID=$!
  # shellcheck disable=SC2086  # SECRET_NAMES: one name per word
  unset $SECRET_NAMES LEGION_WORLD_ADMIN_TOKEN  # legion-worker has them; the processes below never
  # shortcut: not restarted if it fails (a `podman ps` error), after which health-gated compose
  # services stay `starting`; restart it in a loop if that shows up
  "$VENV/bin/python" -m legion.harbor.podman_healthchecks &
  HEALTH_PID=$!
else
  srun "${CTR_SRUN[@]}" bash "$SLURM_DIR/worker.sbatch" --inner &
  SRUN_PID=$!
fi
trap on_usr1 USR1
while kill -0 "$SRUN_PID" 2>/dev/null; do
  wait "$SRUN_PID"
done
wait "$SRUN_PID"; rc=$?   # the real status even when the last wait was cut by a trap
log "legion-worker step exited with $rc"
exit "$rc"
