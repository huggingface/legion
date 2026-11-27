#!/bin/bash
# source scripts/slurm/podman-node.sh: rootless podman for this shell on a cluster node with no
# user systemd, for a `compute: local` run with `environment.backend: podman` there
# (docs/open-models.md "On the HF cluster"). The same setup worker.sbatch gives a Slurm worker: this
# shell's own podman storage under /tmp, explicit engine flags, netavark's DNS daemon kept in
# this job, Docker Hub through mirror.gcr.io (CONTAINERS_REGISTRIES_CONF: registries.conf next to
# this file, unless already set), and podman's health checks (legion.harbor.podman_healthchecks)
# in the background.
# `podman_node_cleanup` stops this shell's containers and the health checks; the storage stays
# for the job's own /tmp. Needs legion's venv active or VENV set (podman-compose, python).

VENV=${VENV:-${VIRTUAL_ENV:-$PWD/.venv}}
export PODMAN_COMPOSE_PROVIDER=$VENV/bin/podman-compose
[ -x "$PODMAN_COMPOSE_PROVIDER" ] || { echo "podman-node.sh: $PODMAN_COMPOSE_PROVIDER is missing" >&2; return 2; }
PODMAN_BIN=$(command -v podman) || { echo "podman-node.sh: podman is not on PATH" >&2; return 2; }
PODMAN_LOCAL=$(mktemp -d /tmp/legion-podman.XXXXXX) || return 2
export XDG_RUNTIME_DIR=$PODMAN_LOCAL/runtime
mkdir -p "$XDG_RUNTIME_DIR" "$PODMAN_LOCAL/bin" || return 2
# `podman compose` prints a banner about its external provider on the stderr of every call,
# `compose exec` included, where Harbor reads a command's output
printf '[engine]\ncompose_warning_logs = false\n' > "$PODMAN_LOCAL/containers.conf"
export CONTAINERS_CONF_OVERRIDE=$PODMAN_LOCAL/containers.conf
CONTAINERS_REGISTRIES_CONF=${CONTAINERS_REGISTRIES_CONF:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/registries.conf}
export CONTAINERS_REGISTRIES_CONF
cat > "$PODMAN_LOCAL/bin/podman" <<EOF
#!/bin/bash
exec "$PODMAN_BIN" --root "$PODMAN_LOCAL/graphroot" --runroot "$PODMAN_LOCAL/runroot" \
    --storage-driver overlay --storage-opt overlay.mount_program=/usr/bin/fuse-overlayfs \
    --cgroup-manager cgroupfs --events-backend file "\$@"
EOF
cat > "$PODMAN_LOCAL/bin/systemd-run" <<'EOF'
#!/bin/bash
if [[ "${1:-} ${2:-} ${3:-}" == "-q --scope --user" && "${4:-}" == /usr/lib/podman/aardvark-dns ]]; then
    shift 3
    exec "$@"
fi
exec /usr/bin/systemd-run "$@"
EOF
chmod 700 "$PODMAN_LOCAL/bin/podman" "$PODMAN_LOCAL/bin/systemd-run"
export PATH="$PODMAN_LOCAL/bin:$PATH"
unset CONTAINER_HOST CONTAINER_CONNECTION DOCKER_HOST
"$VENV/bin/python" -m legion.harbor.podman_healthchecks &
PODMAN_HEALTH_PID=$!

podman_node_cleanup() {
  kill "$PODMAN_HEALTH_PID" 2>/dev/null
  timeout 10 "$PODMAN_LOCAL/bin/podman" stop --all --time 5 >/dev/null 2>&1
  timeout 10 "$PODMAN_LOCAL/bin/podman" rm --all --force >/dev/null 2>&1
}
