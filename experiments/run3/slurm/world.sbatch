#!/bin/bash
#SBATCH --job-name=run3-world
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --dependency=singleton
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
# Pin to a non-Spot node if possible (a world move costs its in-flight runs; agents keep their
# credential: the controller follows world.host and republishes the URL). For example:
##SBATCH --constraint=on-demand
#
# The run's world (worlds/chatbot3): one per run (singleton). world-run.sh does the work, on
# the node (bare venv, workspace in node-local /tmp) or, with SQSH set in run3.env, inside the
# container (workspace in the container's /work; its /tmp is RAM). The workspace is restored
# from <CHATBOT_BUCKET>/ws at start (fresh dir every start), pushed every WS_SYNC_S and once at
# stop. The knobs: a generated <RUN_DIR>/world.yaml (CHATBOT_CONFIG) with
# runs.parallel = WORLD_PARALLEL (default 14; 0 would mean the job's usable CPUs).
#   sbatch --output=$RUN_DIR/logs/%x-%j.out world.sbatch $RUN_DIR/run3.env
# Stop (TERM or USR1, once): stop the server, then ONE bounded push of the workspace
# (WS_STOP_S, default 15 s: the Spot budget is ~25 s after SIGUSR1). best/, probes, the log and
# verdicts are mirrored at each event anyway; the stop push only saves recent agents/ edits.
# shellcheck disable=SC1091
# (Slurm runs a spooled copy of this script: common.sh is found through RUN3_SLURM_DIR, else
# the submit directory; submit from experiments/run3/slurm or export RUN3_SLURM_DIR)
. "${RUN3_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$0")}}/common.sh" "$@"
: "${CHATBOT_WORLD_TOKEN:?}" "${HF_TOKEN:?}" "${CHATBOT_BUCKET:?}" "${REPO:?}" "${RUN_DIR:?}"
B=${CHATBOT_BUCKET%/}
export FLOCK_WORLD_TOKEN=$CHATBOT_WORLD_TOKEN CHATBOT_BUCKET=$B REPO
CONF=$RUN_DIR/world.yaml   # generated: the experiment's world.yaml with runs.parallel set
sed -E "s/^(  parallel:)[[:space:]]*[0-9]+/\1 ${WORLD_PARALLEL:-14}/" \
  "$REPO/experiments/run3/world.yaml" > "$CONF.tmp" && mv "$CONF.tmp" "$CONF"
grep -q "parallel: ${WORLD_PARALLEL:-14}" "$CONF" || { log "world.yaml: runs.parallel not set"; exit 1; }
export CHATBOT_CONFIG=$CONF
HERE=${RUN3_SLURM_DIR:-$REPO/experiments/run3/slurm}
if [ -n "${SQSH:-}" ]; then
  # in the container: /work is its own root filesystem (its /tmp is RAM: never used);
  # the repo and the run dir are mounted read-only; flock is the image's /opt/flock
  export CHATBOT_WS=/work/world/ws HOME=/work/world/home HF_HOME=/work/world/hf \
         TMPDIR=/work/world/tmp FLOCK_HOMELESS_ROOT=/work/world PY=/opt/flock/bin/python
  srun --ntasks=1 --container-image="$SQSH" --no-container-mount-home --container-writable \
       --container-mounts="$REPO:$REPO:ro,$RUN_DIR:$RUN_DIR:ro" \
       bash -c 'mkdir -p "$HOME" "$HF_HOME" "$TMPDIR" && exec "$0"' "$HERE/world-run.sh" &
else
  export CHATBOT_WS=/tmp/run3-world-${SLURM_JOB_ID:-0}/ws PY
  "$HERE/world-run.sh" &
fi
PID=$!
HOST=$(hostname)
n=0
until "$VENV/bin/python" -c 'import sys,urllib.request as u; u.urlopen(sys.argv[1], timeout=3)' \
      "http://127.0.0.1:7860/healthz" 2>/dev/null; do  # the host file once the world answers
  kill -0 "$PID" 2>/dev/null || { log "world exited before answering"; exit 1; }
  [ $((n % 12)) = 0 ] && log "waiting for the world to answer"; n=$((n + 1)); sleep 5
done
write_host "$FLOCK_WORLD_HOST_FILE" "$HOST"
log "world on $HOST:7860 (host file $FLOCK_WORLD_HOST_FILE)"
fwd() { kill -TERM "$PID" 2>/dev/null || true; }   # srun forwards it into the container
trap 'once fwd' TERM USR1 INT
rc=0; wait_for "$PID" || rc=$?
if [ -z "${SQSH:-}" ]; then rm -rf "/tmp/run3-world-${SLURM_JOB_ID:-0}"; fi
log "world exited $rc"
exit "$rc"
