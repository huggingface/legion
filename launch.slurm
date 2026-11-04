#!/bin/bash -l
#SBATCH --job-name=shepherd
#SBATCH --partition=hopper-cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=54G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=shepherd-%j.log
set -euo pipefail
umask 077

# sbatch copies this script; resolve paths from the submission directory.
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the Shepherd repository}"

if [[ $# != 1 || ! -f scripts/run.py || ! -x .venv/bin/python ]]; then
    echo 'Usage (from Shepherd): sbatch launch.slurm path/to/config.yaml' >&2
    exit 2
fi
if [[ ${SLURM_JOB_NUM_NODES:-1} != 1 || ${SLURM_NTASKS:-1} != 1 ]]; then
    echo 'This launcher supports one node and one task per experiment.' >&2
    exit 2
fi
CONFIG_PATH=$(realpath -- "$1")
PYTHON="$PWD/.venv/bin/python"
export PODMAN_COMPOSE_PROVIDER="$PWD/.venv/bin/podman-compose"
if [[ ! -x "$PODMAN_COMPOSE_PROVIDER" ]]; then
    echo "Missing podman-compose; run uv sync --locked before submitting." >&2
    exit 2
fi
"$PYTHON" - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path
from shepherd.config import Config
config = Config.load(Path(sys.argv[1]))
if config.environment.backend != 'podman':
    raise SystemExit('The Slurm launcher requires environment.backend: podman')
PY

# Harbor downloads tasks with Git. Scope HF_TOKEN authentication to the Hub,
# without writing a token or changing the user's Git configuration.
if [[ -n "${HF_TOKEN:-}" ]]; then
    git_slot=${GIT_CONFIG_COUNT:-0}
    export "GIT_CONFIG_KEY_${git_slot}=credential.https://huggingface.co.helper"
    export "GIT_CONFIG_VALUE_${git_slot}="
    git_slot=$((git_slot + 1))
    export "GIT_CONFIG_KEY_${git_slot}=credential.https://huggingface.co.helper"
    export "GIT_CONFIG_VALUE_${git_slot}="'!f() { if [ "$1" = get ]; then printf "username=hf_user\npassword=%s\n" "$HF_TOKEN"; fi; }; f'
    export GIT_CONFIG_COUNT=$((git_slot + 1))
fi
export GIT_TERMINAL_PROMPT=0

LOCAL=$(mktemp -d /tmp/shepherd.XXXXXX)
export XDG_RUNTIME_DIR="$LOCAL/runtime"
export HF_XET_CACHE="$LOCAL/xet"
mkdir -p "$XDG_RUNTIME_DIR" "$LOCAL/bin"
PODMAN=$(command -v podman)
# Explicit engine flags work on the cluster's Podman 4.9 and avoid user storage.
cat > "$LOCAL/bin/podman" <<EOF
#!/bin/bash
exec "$PODMAN" --root "$LOCAL/graphroot" --runroot "$LOCAL/runroot" \
    --storage-driver overlay --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs \
    --cgroup-manager cgroupfs --events-backend file "\$@"
EOF
chmod 700 "$LOCAL/bin/podman"
# Netavark 1.4 tries systemd-run even when no user bus exists. Keep only its
# DNS daemon in this Slurm job; delegate every other invocation unchanged.
cat > "$LOCAL/bin/systemd-run" <<'EOF'
#!/bin/bash
if [[ "${1:-} ${2:-} ${3:-}" == "-q --scope --user" && "${4:-}" == /usr/lib/podman/aardvark-dns ]]; then
    shift 3
    exec "$@"
fi
exec /usr/bin/systemd-run "$@"
EOF
chmod 700 "$LOCAL/bin/systemd-run"
export PATH="$LOCAL/bin:$PATH"
# Prevent inherited remote-engine settings from bypassing this job's storage.
unset CONTAINER_HOST CONTAINER_CONNECTION DOCKER_HOST
test "$(podman info --format '{{.Store.GraphRoot}}')" = "$LOCAL/graphroot"
child=
healthchecks=
cleanup() {
    status=$?
    trap - EXIT
    trap '' TERM INT
    if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" || true
    fi
    if [[ -n "$healthchecks" ]]; then
        kill -TERM "$healthchecks" 2>/dev/null || true
        wait "$healthchecks" 2>/dev/null || true
    fi
    # These commands address only the private storage configured above.
    timeout 15 podman stop --all --time 5 || true
    timeout 15 podman rm --all --force || true
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' TERM INT

"$PYTHON" scripts/podman_healthchecks.py &
healthchecks=$!
"$PYTHON" scripts/run.py --config "$CONFIG_PATH" &
child=$!
status=0
wait -n "$child" "$healthchecks" || status=$?
if ! kill -0 "$healthchecks" 2>/dev/null; then
    echo "Podman healthcheck scheduler exited unexpectedly" >&2
    status=1
fi
exit "$status"
