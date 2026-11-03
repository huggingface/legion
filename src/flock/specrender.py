"""Spec rendering at read time (docs/live.md, "Env files and --ref"): `${VAR}` /
`${VAR:-default}` substituted in string values from `--env` files and the process
environment, with where each value came from (`update --check`); `--ref` reads the folder at a
git ref. Overlays (deep-merged over the spec) remain for library callers only (no CLI option)."""
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

from flock.models import _SpecLoader, load_spec_yaml
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


def substitute(data: Any, env: dict[str, str],
               used: dict[str, list[tuple[str, bool]]] | None = None) -> Any:
    """Expand placeholders in every string value (keys are left alone) and return a fresh
    tree (no shared sub-objects, so `<<: *anchor` copies come out independent). A string that
    is one placeholder takes the value's YAML type (`"${N:-20}"` -> 20). Raises ValueError
    naming every unset variable without a default. `used` gets, per dotted path, each
    variable with whether its value came from `env` (else its default)."""
    missing: set[str] = set()
    here: list[tuple[str, bool]] = []

    def rep(m: re.Match) -> str:
        name, default = m.groups()
        if name is None:
            return "$"
        if name in _RUNTIME:
            return m.group(0)
        value = env.get(name)
        if value or (value is not None and default is None):  # `:-` as in sh: empty = unset
            here.append((name, True))
            return value
        if default is not None:
            here.append((name, False))
            return default
        missing.add(name)
        return m.group(0)

    def one(s: str, path: str) -> Any:
        here.clear()
        out = _VAR.sub(rep, s)
        if here and used is not None:
            used[path] = list(here)
        whole = _VAR.fullmatch(s)
        if whole and whole.group(1) and out:
            try:
                typed = load_spec_yaml(out)
            except yaml.YAMLError:
                return out
            if isinstance(typed, (bool, int, float)):
                return typed
        return out

    def walk(v: Any, path: str) -> Any:
        if isinstance(v, str):
            return one(v, path)
        if isinstance(v, dict):
            return {k: walk(x, f"{path}.{k}" if path else str(k)) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x, path) for x in v]
        return v

    result = walk(data, "")
    if missing:
        raise ValueError("unset variables without a default: " + ", ".join(sorted(missing))
                         + " (set them in an --env file or the environment)")
    return result


def yaml_lines(text: str) -> dict[str, int]:
    """Dotted path -> the line (1-based) of each value in spec YAML, `<<` merge keys
    resolved (a merged value keeps its anchor's line)."""
    loader = _SpecLoader(text)
    out: dict[str, int] = {}

    def walk(node: yaml.Node, path: str) -> None:
        out[path] = node.start_mark.line + 1
        if isinstance(node, yaml.MappingNode):
            loader.flatten_mapping(node)
            for k, v in node.value:
                walk(v, f"{path}.{k.value}" if path else str(k.value))

    try:
        node = loader.get_single_node()
    finally:
        loader.dispose()
    if node is not None:
        walk(node, "")
    return out


def render(spec_file: str | Path, env_files=(), overlays=(),
           sources: dict[str, str] | None = None) -> Any:
    """The spec's data as `flock apply` validates it: base, overlays in order, then
    substitution. Env sources, later winning: the process environment, then each --env file
    in order. `sources` gets where each value came from, per dotted path:
    `spec.yaml:12`, `run.env WORKER_MEM`, `env WORKER_MEM`, `spec.yaml:12 default of X`."""
    env: dict[str, str] = dict(os.environ)
    origin = dict.fromkeys(env, "env")
    for f in env_files:
        vals = read_env_file(f)
        env.update(vals)
        origin.update(dict.fromkeys(vals, Path(f).name))
    text = Path(spec_file).read_text(encoding="utf-8")
    data = load_spec_yaml(text)
    for o in overlays:
        data = merge(data, load_spec_yaml(Path(o).read_text(encoding="utf-8")) or {})
    used: dict[str, list[tuple[str, bool]]] = {}
    out = substitute(data, env, used)
    if sources is not None:
        name = Path(spec_file).name
        sources.update({p: f"{name}:{n}" for p, n in yaml_lines(text).items()})
        for p, names in used.items():
            sources[p] = ", ".join(f"{origin[v]} {v}" if from_env else
                                   f"{sources.get(p, name)} default of {v}"
                                   for v, from_env in names)
    return out


def source_of(sources: dict[str, str], path: str) -> str:
    """Where the value at a dotted path came from: its own entry or its nearest parent's
    (a list item, a key of an inline mapping), else a model default."""
    path = path.replace("while_", "while")  # LoopSpec.while_ dumps by its field name
    while path:
        if path in sources:
            return sources[path]
        path = path.rpartition(".")[0]
    return "default"


def at_ref(target: str | Path, ref: str) -> tuple[Path, dict[str, str]]:
    """`--ref REF[:PATH]`: the repo holding `target` exported at REF (`git archive`) into a
    temp dir removed at exit. Returns the spec file there (PATH, repo-relative, else `target`'s
    own path in the repo; a folder means its spec.yaml) and the `X-Flock-Source` header.
    Raises ValueError."""
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

    src = {"git_sha": sha, "ref": ref, "dirty": False, "by": os.environ.get("USER")}
    return spec, {"X-Flock-Source": json.dumps(src)}
