"""Apply validated cluster compatibility settings to a fresh run snapshot, not Flock.

Run after flock init: python prepare_slurm.py RUN_DIR
"""
import sys
from pathlib import Path


def prepare(root):
    script = root / "slurm/worker.sbatch"
    text = script.read_text()
    if '# ProgramBench cluster compatibility' in text:
        return
    marker = '  # Explicit engine flags work on the cluster\'s Podman 4.9 and avoid user storage.'
    if marker not in text:
        raise ValueError('Worker script changed; inspect before applying compatibility settings')
    setup = '''  # ProgramBench cluster compatibility (run-local).
  export PATH="/fsx/edward/work/swarms/harbor-ports/.work/tools/git-lfs/usr/bin:$PATH"
  for setting in 'process=git-lfs filter-process' 'clean=git-lfs clean -- %f' 'smudge=git-lfs smudge -- %f' 'required=true'; do
    slot=${GIT_CONFIG_COUNT:-0}
    export "GIT_CONFIG_KEY_${slot}=filter.lfs.${setting%%=*}" "GIT_CONFIG_VALUE_${slot}=${setting#*=}"
    export GIT_CONFIG_COUNT=$((slot + 1))
  done
  helpers=/fsx/edward/.local/share/flock/network/v1.11.0
  mkdir -p "$PODMAN_LOCAL/network-helpers"
  cp "$helpers/netavark" "$helpers/aardvark-dns" "$PODMAN_LOCAL/network-helpers/" || exit 2
  export CONTAINERS_CONF_OVERRIDE="$PODMAN_LOCAL/network-stack.conf"
  printf '[engine]\\nhelper_binaries_dir = ["%s", "/usr/lib/podman"]\\n' "$PODMAN_LOCAL/network-helpers" > "$CONTAINERS_CONF_OVERRIDE"
  export CONTAINERS_REGISTRIES_CONF="$PODMAN_LOCAL/registries.conf"
  printf 'unqualified-search-registries = ["docker.io"]\\n' > "$CONTAINERS_REGISTRIES_CONF"
'''
    text = text.replace(marker, setup + marker)
    marker = '  chmod 700 "$PODMAN_LOCAL/bin/podman" "$PODMAN_LOCAL/bin/systemd-run" || exit 2'
    if marker not in text:
        raise ValueError('DNS shim changed; inspect before applying compatibility settings')
    text = text.replace(marker, '''  # An explicit path survives Flock's removal of inherited FLOCK_* variables.
  sed -i "s|/usr/lib/podman/aardvark-dns|$PODMAN_LOCAL/network-helpers/aardvark-dns|g" "$PODMAN_LOCAL/bin/systemd-run"
''' + marker)
    script.write_text(text)
    (root / 'cluster.env').chmod(0o600)


if __name__ == '__main__':
    prepare(Path(sys.argv[1]).resolve())
