"""Worker secrets from a file instead of the exec environment (docs/slurm.md, "Worker
secrets file").

A process's /proc/<pid>/environ always shows the environment it was exec'd with; unsetting a
variable later does not change it. So a launcher that wants the worker's credentials out of
there passes `FLOCK_WORKER_SECRETS_FILE=<path>` instead: an env-format file, mode 0600 (no
group or other bits), read once at start. Its values take precedence over the environment, and
every name it defines is removed from `os.environ`, so child processes that inherit the
environment never see it. `secret_env()` is the view the worker reads secrets through: the
file's values over the live `os.environ`. Without the variable nothing changes: secrets come
from the environment as before. (`FLOCK_SECRETS_FILE` is the cluster's shell-sourced file and
is never read by flock.)

Format: one `NAME=value` per line; blank lines and lines starting with `#` are ignored; an
optional leading `export `; a value wrapped in matching single or double quotes is unquoted.
No variable expansion, escapes or multi-line values. `FLOCK_WORKER_SECRETS_FILE_UNLINK=1`
deletes the file after reading it.
"""
from __future__ import annotations

import os
import re
import stat
from collections import ChainMap
from collections.abc import Mapping, MutableMapping
from pathlib import Path

SECRETS_FILE_ENV = "FLOCK_WORKER_SECRETS_FILE"
UNLINK_ENV = "FLOCK_WORKER_SECRETS_FILE_UNLINK"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_secrets: dict[str, str] = {}


class SecretsFileError(RuntimeError):
    """The secrets file is unreadable, too open or malformed (the message never has a value)."""


def parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or not _NAME.fullmatch(name):
            raise SecretsFileError(f"secrets file line {n}: expected NAME=value")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[name] = value
    return out


def read_file(path: str | Path) -> dict[str, str]:
    """Read and check a secrets file: a regular file owned by this user, no group/other bits."""
    path = Path(path)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as e:
        raise SecretsFileError(f"cannot open the secrets file {path}: {e.strerror}") from None
    with os.fdopen(fd, encoding="utf-8") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise SecretsFileError(f"the secrets file {path} is not a regular file")
        if st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise SecretsFileError(f"the secrets file {path} must be owned by this user, mode 0600")
        return parse(f.read())


def load(env: MutableMapping[str, str] = os.environ) -> list[str]:
    """At process start: if `FLOCK_WORKER_SECRETS_FILE` is set, read it, keep its values for
    `secret_env()` and remove those names from `env`. Returns the names loaded (never values)."""
    path = env.get(SECRETS_FILE_ENV)
    if not path:
        return []
    values = read_file(path)
    if env.get(UNLINK_ENV) == "1":
        Path(path).unlink(missing_ok=True)
    _secrets.clear()
    _secrets.update(values)
    for name in values:
        env.pop(name, None)
    return sorted(values)


def secret_env() -> Mapping[str, str]:
    """The file's secrets over the live process environment."""
    return ChainMap(_secrets, os.environ)  # type: ignore[arg-type]
