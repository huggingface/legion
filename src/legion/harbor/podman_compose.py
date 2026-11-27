"""podman-compose 1.5 (legion's pinned provider), run by legion's Podman environment in place of
`podman compose` / `podman-compose` when Harbor picked that dialect, with one fix.

legion.harbor.envs passes a secret to `compose exec` as a bare `--env NAME`, its value in this
process's environment, so it stays out of the argv. podman-compose 1.5 also merges the exec's
`--env` list into the map it interpolates the compose file with, and a bare NAME goes in as
None, over the environment's value: a task's `${HF_TOKEN:?required}` failed and `${HF_TOKEN}`
was empty. Here a bare name interpolates as the environment's value; the exec still passes
`--env NAME` bare. Run as `python -I -m legion.harbor.podman_compose ARGS...`."""

import os

import podman_compose

_rec_subs = podman_compose.rec_subs


def rec_subs(value, subs_dict):
    """podman-compose's, with each None in the map (a bare `--env NAME`) read from the
    environment."""
    if any(v is None and k in os.environ for k, v in subs_dict.items()):
        subs_dict = {k: os.environ.get(k) if v is None else v for k, v in subs_dict.items()}
    return _rec_subs(value, subs_dict)


def main():
    podman_compose.rec_subs = rec_subs  # its own recursive calls find this one too
    podman_compose.main()


if __name__ == "__main__":
    main()
