"""Apply validated cluster compatibility settings to a fresh run snapshot, not Legion.

Run after legion init: python prepare_slurm.py RUN_DIR NETWORK_HELPERS
"""

import shlex
import sys
from pathlib import Path


def prepare(root, helpers):
    script = root / "slurm/worker.sbatch"
    text = script.read_text()
    if "# AgentEscape cluster compatibility" in text:
        return
    marker = "  # Explicit engine flags work on the cluster's Podman 4.9 and avoid user storage."
    if marker not in text:
        raise ValueError("Worker script changed; inspect before applying compatibility settings")
    setup = """  # AgentEscape cluster compatibility (run-local).
  for setting in 'process=git-lfs filter-process' 'clean=git-lfs clean -- %f' 'smudge=git-lfs smudge -- %f' 'required=true'; do
    slot=${GIT_CONFIG_COUNT:-0}
    export "GIT_CONFIG_KEY_${slot}=filter.lfs.${setting%%=*}" "GIT_CONFIG_VALUE_${slot}=${setting#*=}"
    export GIT_CONFIG_COUNT=$((slot + 1))
  done
  helpers=__NETWORK_HELPERS__
  mkdir -p "$PODMAN_LOCAL/network-helpers"
  cp "$helpers/netavark" "$helpers/aardvark-dns" "$PODMAN_LOCAL/network-helpers/" || exit 2
  export CONTAINERS_CONF_OVERRIDE="$PODMAN_LOCAL/network-stack.conf"
  printf '[engine]\\nhelper_binaries_dir = ["%s", "/usr/lib/podman"]\\n' "$PODMAN_LOCAL/network-helpers" > "$CONTAINERS_CONF_OVERRIDE"
  export CONTAINERS_REGISTRIES_CONF="$PODMAN_LOCAL/registries.conf"
  printf 'unqualified-search-registries = ["docker.io"]\\n' > "$CONTAINERS_REGISTRIES_CONF"
"""
    setup += """  # Harbor uses Docker Compose spelling; podman-compose expects -T.
  export VENV
  cat > "$PODMAN_LOCAL/bin/podman-compose" <<'COMPOSE'
#!/bin/bash
args=("$@")
for ((i=0; i+1<${#args[@]}; i++)); do
  if [[ "${args[i]}" == exec && "${args[i+1]}" == --no-TTY ]]; then
    args[i+1]=-T
  fi
done
exec "$VENV/bin/podman-compose" "${args[@]}"
COMPOSE
  chmod 700 "$PODMAN_LOCAL/bin/podman-compose"
  cp __COMPOSE_COMPAT__ "$PODMAN_LOCAL/bin/podman-compose"
  export PODMAN_COMPOSE_PROVIDER="$PODMAN_LOCAL/bin/podman-compose"
"""
    setup = setup.replace("__NETWORK_HELPERS__", shlex.quote(helpers)).replace("__COMPOSE_COMPAT__", shlex.quote(str(Path(__file__).with_name("compose_compat.py"))))
    text = text.replace(marker, setup + marker)
    marker = '  chmod 700 "$PODMAN_LOCAL/bin/podman" "$PODMAN_LOCAL/bin/systemd-run" || exit 2'
    if marker not in text:
        raise ValueError("DNS shim changed; inspect before applying compatibility settings")
    text = text.replace(
        marker,
        """  # An explicit path survives Legion's removal of inherited LEGION_* variables.
  sed -i "s|/usr/lib/podman/aardvark-dns|$PODMAN_LOCAL/network-helpers/aardvark-dns|g" "$PODMAN_LOCAL/bin/systemd-run"
"""
        + marker,
    )
    script.write_text(text)
    (root / "cluster.env").chmod(0o600)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]).resolve(), sys.argv[2])
