"""Spec rendering at read time (docs/live.md, "Env files, overlays and --ref"): overlays
deep-merged over the base spec, then `${VAR}` / `${VAR:-default}` substituted in string
values from `--env` files and the process environment; `--ref` reads the folder at a git ref."""
from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from flock.models import load_spec_yaml
from flock.world.protocol import WORLD_HOST_VAR

# Only UPPERCASE names are variables, so inline prompt placeholders (`${spend}`) pass through;
# `$${` is a literal `${` (a `$$` elsewhere stays as is). `${WORLD_HOST}` is always left for the
# controller, which expands it at run time (world.url, docs/slurm.md).
_RUNTIME = {WORLD_HOST_VAR[2:-1]}
_VAR = re.compile(r"\$\$(?=\{)|\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def read_env_file(path: str | Path) -> dict[str, str]:
    """NAME=value lines (`export ` prefix, `#` comments and surrounding quotes allowed).
    shortcut: no shell expansion inside values; use a real shell `. run.env` if one needs it."""
    env = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.removeprefix("export ").partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"{path}:{n}: expected NAME=value")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        env[name] = value
    return env


def merge(base: Any, over: Any) -> Any:
    """Overlay semantics: mappings merge key by key, `null` deletes a key, anything else
    (lists, scalars) replaces."""
    if not (isinstance(base, dict) and isinstance(over, dict)):
        return over
    out = dict(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = merge(out[k], v) if k in out else v
    return out


def substitute(data: Any, env: dict[str, str]) -> Any:
    """Expand placeholders in every string value (keys are left alone) and return a fresh
    tree (no shared sub-objects, so `<<: *anchor` copies come out independent). A string that
    is one placeholder takes the value's YAML type (`"${N:-20}"` -> 20). Raises ValueError
    naming every unset variable without a default."""
    missing: set[str] = set()

    def rep(m: re.Match) -> str:
        name, default = m.groups()
        if name is None:
            return "$"
        if name in _RUNTIME:
            return m.group(0)
        value = env.get(name)
        if value or (value is not None and default is None):  # `:-` as in sh: empty = unset
            return value
        if default is not None:
            return default
        missing.add(name)
        return m.group(0)

    def one(s: str) -> Any:
        out = _VAR.sub(rep, s)
        whole = _VAR.fullmatch(s)
        if whole and whole.group(1) and out:
            try:
                typed = load_spec_yaml(out)
            except yaml.YAMLError:
                return out
            if isinstance(typed, (bool, int, float)):
                return typed
        return out

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return one(v)
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v

    result = walk(data)
    if missing:
        raise ValueError("unset variables without a default: " + ", ".join(sorted(missing))
                         + " (set them in an --env file or the environment)")
    return result


def render(spec_file: str | Path, env_files=(), overlays=()) -> Any:
    """The spec's data as `flock apply` validates it: base, overlays in order, then
    substitution. Env sources, later winning: each --env file in order, then the process
    environment."""
    env: dict[str, str] = {}
    for f in env_files:
        env.update(read_env_file(f))
    env.update(os.environ)
    data = load_spec_yaml(Path(spec_file).read_text(encoding="utf-8"))
    for o in overlays:
        data = merge(data, load_spec_yaml(Path(o).read_text(encoding="utf-8")) or {})
    return substitute(data, env)


def at_ref(target: str | Path, ref: str, overlays=()) -> tuple[Path, list[str], dict[str, str]]:
    """`--ref REF[:PATH]`: the repo holding `target` exported at REF (`git archive`) into a
    temp dir removed at exit. Returns the spec file there (PATH, repo-relative, else `target`'s
    own path in the repo; a folder means its spec.yaml), the overlays (those inside the repo are
    read at REF too) and the `X-Flock-Source` header. Raises ValueError."""
    ref, _, rel = ref.partition(":")
    target = Path(target).resolve()
    where = target if target.is_dir() else target.parent

    def git(*args: str) -> str:
        r = subprocess.run(["git", "-C", str(where), *args], check=False,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise ValueError(f"--ref {ref}: {r.stderr.strip() or 'not a commit'}")
        return r.stdout.strip()

    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    sha = git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    tmp = Path(tempfile.mkdtemp(prefix="flock-ref-"))
    atexit.register(shutil.rmtree, tmp, True)
    git("-C", str(root), "archive", "--format=tar", f"--output={tmp / 'ref.tar'}", sha)
    subprocess.run(["tar", "-xf", "ref.tar"], cwd=tmp, check=True)
    spec = tmp / (rel or target.relative_to(root))
    if spec.is_dir():
        spec = spec / "spec.yaml"
    if not spec.is_file():
        raise ValueError(f"--ref {ref}: {spec.relative_to(tmp)} not found at {sha[:8]}")

    def moved(o: str) -> str:
        p = Path(o).resolve()
        return str(tmp / p.relative_to(root)) if p.is_relative_to(root) else o

    src = {"git_sha": sha, "ref": ref, "dirty": False, "by": os.environ.get("USER")}
    return spec, [moved(o) for o in overlays], {"X-Flock-Source": json.dumps(src)}
