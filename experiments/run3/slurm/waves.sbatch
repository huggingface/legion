#!/bin/bash
#SBATCH --job-name=run3-waves
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --dependency=singleton
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
# Renders the spec (render.py --variant $RUN3_VARIANT --compute slurm), applies it once the
# controller and the world answer (it keeps polling, with a line a minute; it never gives up),
# then runs the wave schedule: `waves.py run` (its clock starts at the controller's first
# answer and is kept in <spec>.waves.json, so a requeued runner resumes the same schedule).
#   sbatch --output=$RUN_DIR/logs/%x-%j.out waves.sbatch $RUN_DIR/run3.env
# A re-apply after a requeue keeps the live replica counts (`waves.py pin` first).
# shellcheck disable=SC1091
# (Slurm runs a spooled copy of this script: common.sh is found through RUN3_SLURM_DIR, else
# the submit directory; submit from experiments/run3/slurm or export RUN3_SLURM_DIR)
. "${RUN3_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$0")}}/common.sh" "$@"
: "${FLOCK_TOKEN:?}" "${REPO:?}" "${RUN_DIR:?}"
EXP=$REPO/experiments/run3
SPEC=$RUN_DIR/spec.slurm.yaml
WAVES=$EXP/${RUN3_WAVES:-waves.yaml}
ctl_url() { echo "http://$(head -1 "$FLOCK_SLURM_HOST_FILE"):7860"; }
up() { "$PY" -c 'import sys,urllib.request as u; u.urlopen(sys.argv[1] + "/healthz", timeout=5)' "$1" 2>/dev/null; }
n=0
until [ -s "$FLOCK_SLURM_HOST_FILE" ] && up "$(ctl_url)" \
      && [ -s "$FLOCK_WORLD_HOST_FILE" ] && up "http://$(head -1 "$FLOCK_WORLD_HOST_FILE"):7860"; do
  [ $((n % 12)) = 0 ] && log "waiting for the controller and the world (host files in $RUN_DIR)"
  n=$((n + 1)); sleep 5
done
export FLOCK_URL=$(ctl_url)
log "controller $FLOCK_URL and world up"
"$PY" "$EXP/render.py" --variant "${RUN3_VARIANT:-smoke}" --compute slurm --out "$SPEC" >/dev/null
cp "$SPEC" "$EXP/$(basename "$SPEC")"   # next to spec.yaml: prompts and tools resolve
SPEC=$EXP/$(basename "$SPEC")
SWARM=$("$PY" -c 'import sys; from flock.cli import load_spec; print(load_spec(sys.argv[1]).swarm)' "$SPEC")
if "$PY" -m flock status "$SWARM" >/dev/null 2>&1; then
  log "swarm $SWARM exists: updating it (live replica counts kept)"
  (cd "$EXP" && "$PY" -m flock update "$SPEC")
else
  (cd "$EXP" && "$PY" -m flock launch "$SPEC")
fi
log "applied $SWARM; waves from $(basename "$WAVES")"
cd "$EXP"
"$PY" waves.py run --spec "$SPEC" --waves "$WAVES" --log "$RUN_DIR/logs/waves.jsonl" &
PID=$!
trap 'once kill -TERM "$PID"' TERM USR1 INT
rc=0; wait_for "$PID" || rc=$?
log "waves exited $rc"
exit "$rc"
