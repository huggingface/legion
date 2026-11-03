"""Swarm operations shared by the `flock` CLI and Python callers: read an experiment folder
(`read_spec`), check it, `launch` it (create, adopt or resume) and `update` the running swarm,
all through `flock.client.Client`. Nothing here prints or asks: results are returned,
refusals raise SpecError, failed controller calls raise ClientError.

    spec = read_spec("exp/spec.yaml", env_files)
    done = launch(Client(), spec, "exp/spec.yaml")
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from flock.client import Client, ClientError
from flock.models import SwarmSpec, comms_tools, loop_refusals
from flock.specrender import render, source_of


class SpecError(Exception):
    """A spec that cannot be applied: the lines say why (printed to stderr by the CLI).
    `done`: what `launch` did or found before refusing (the keys it returns), to show first."""

    def __init__(self, lines: list[str], done: dict[str, Any] | None = None) -> None:
        super().__init__("; ".join(lines))
        self.lines, self.done = lines, done or {}


def _inline(base: Path, holder, field: str, target: str, where: str) -> None:
    """Read `holder.<field>` (a path relative to the spec) into `holder.<target>`."""
    name = getattr(holder, field)
    if not name or getattr(holder, target) is not None:
        return
    src = Path(name)
    src = src if src.is_absolute() else base / src
    if not src.is_file():
        raise SpecError([f"{where}: {field} {src} not found"])
    setattr(holder, target, src.read_text(encoding="utf-8"))


def read_spec(spec_file: str, env_files=(), overlays=(),
              sources: dict[str, str] | None = None) -> SwarmSpec:
    """Read a spec (overlay files merged: library callers only; then ${VAR} from --env files
    and the environment substituted: flock.specrender, `sources` filled) and inline each type's
    system_prompt_file (plus system_prompt_append_file) and loop prompt files, and the comms
    kickoff and protocol files (relative to the spec file). Raises SpecError."""
    path = Path(spec_file)
    try:
        spec = SwarmSpec.model_validate(render(path, env_files, overlays, sources))
    except ValidationError as e:
        lines = []
        for err in e.errors():
            where = ".".join(str(x) for x in err["loc"])
            lines.append(f"{spec_file}: {where + ': ' if where else ''}{err['msg']}")
        raise SpecError(lines) from None
    except ValueError as e:  # an unset ${VAR}, a bad --env line
        raise SpecError([f"{spec_file}: {e}"]) from None
    except (OSError, yaml.YAMLError) as e:  # unreadable file, broken YAML (a save in progress)
        raise SpecError([f"{spec_file}: {' '.join(str(e).split())}"]) from None
    for name, ts in spec.types.items():
        if ts.system_prompt_file and ts.system_prompt is None:
            src = Path(ts.system_prompt_file)
            src = src if src.is_absolute() else path.parent / src
            if not src.is_file():
                raise SpecError([f"type {name}: system_prompt_file {src} not found"])
            ts.system_prompt = src.read_text(encoding="utf-8")
        if ts.system_prompt_append_file:
            src = Path(ts.system_prompt_append_file)
            src = src if src.is_absolute() else path.parent / src
            if not src.is_file():
                raise SpecError([f"type {name}: system_prompt_append_file {src} not found"])
            ts.system_prompt = ((ts.system_prompt or "").rstrip("\n") + "\n\n"
                                + src.read_text(encoding="utf-8")).lstrip("\n")
        if ts.loop is not None:
            _inline(path.parent, ts.loop, "prompt_file", "prompt", f"type {name}: loop")
            if ts.loop.final is not None:
                _inline(path.parent, ts.loop.final, "prompt_file", "prompt", f"type {name}: loop.final")
    c = spec.comms
    if c is not None:  # kickoff and protocol files, next to the spec (docs/comms.md)
        def read(rel: str, what: str) -> str:
            src = Path(rel)
            src = src if src.is_absolute() else path.parent / src
            if not src.is_file():
                raise SpecError([f"comms.{what}: {src} not found"])
            return src.read_text(encoding="utf-8")

        if c.kickoff_file:
            c.kickoff = read(c.kickoff_file, "kickoff_file")
        c.protocols_text = [read(f, "protocols") for f in c.protocols]
    w = spec.world
    if w is not None and w.command is not None:  # the world runs in its dir (docs/world.md)
        d = Path(w.dir or ".")
        w.dir = str((d if d.is_absolute() else path.parent / d).resolve())
        if not Path(w.dir).is_dir():
            raise SpecError([f"world.dir: {w.dir} not found"])
    return spec


# `loop.on` sources the workers' comms events feed (docs/comms.md) -> their `comms.wake` kind.
COMMS_WAKES = {"comms.mention": "mention", "comms.thread": "thread"}


def _unfed(spec: SwarmSpec, ts, src: str) -> str | None:
    """Why `src` in a type's `loop.on` can never wake its loop; None when comms feeds it."""
    kind = COMMS_WAKES.get(src)
    if kind is None:
        return "no feeder in this version (comms.mention and comms.thread are fed)"
    if spec.comms is None or not comms_tools(ts, spec.comms):
        return ("no comms feeder: the type is not on the board (set `comms: true` and the "
                "swarm-level `comms:` block)")
    if kind not in spec.comms.wake:
        return f"no comms feeder: comms.wake does not list {kind}"
    return None


def loop_problems(spec: SwarmSpec) -> tuple[list[str], list[str]]:
    """(refusals, warnings): loop settings this version cannot run, and `on:` sources that
    have no feeder."""
    refused = [f"refused: {r}" for r in loop_refusals(spec)]
    warnings = [f"warning: type {name}: loop.on {src}: {why}"
                for name, ts in spec.types.items() for src in (ts.loop.on if ts.loop else [])
                if (why := _unfed(spec, ts, src))]
    return refused, warnings


def check_spec(spec: SwarmSpec, spec_file: str):
    """(the folder's packed tools bundle or None, warnings), or SpecError with the loop
    settings this version cannot run or a bad tools folder (`done["warnings"]` then)."""
    from flock.tools.loader import ToolsError
    from flock.tools.spec import prepare

    refused, warnings = loop_problems(spec)
    if refused:
        raise SpecError(refused)
    warnings += extra_warnings(spec)
    try:
        return prepare(spec, spec_file), warnings
    except ToolsError as e:
        raise SpecError([f"{spec_file}: tools: {e}"], {"warnings": warnings}) from None


def provider_backed(spec: SwarmSpec) -> bool:
    return any(ts.resolved_model is not None for ts in spec.types.values())


# Harnesses that cannot add request-body fields, so a model's `extra` is kept but not sent.
# The `llm` harness (and any harness not listed) sends it.
EXTRA_NOT_SENT = frozenset({"claude", "codex", "pi"})


def extra_warnings(spec: SwarmSpec) -> list[str]:
    from flock.worker.adapters.pi import env_dropped

    return [f"warning: types.{name}.env.{k}: dropped by pi isolation (use `secrets`)"
            for name, ts in spec.types.items() if ts.harness.value == "pi"
            for k in env_dropped(ts.env, ts.resolved_model and ts.resolved_model.provider.api_key_env)] + [f"warning: type {name}: `extra` is kept in the spec but not sent by the "
            f"{ts.harness.value} harness (docs/open-models.md)"
            for name, ts in spec.types.items()
            if ts.harness.value in EXTRA_NOT_SENT and ts.resolved_model is not None
            and ts.resolved_model.extra]


def unprobed(spec: SwarmSpec) -> list[str]:
    return sorted({ts.resolved_model.qualified for ts in spec.types.values()
                   if ts.resolved_model is not None})


def source_header(spec_file: str | Path | None) -> dict[str, str]:
    """`X-Flock-Source` for an apply (spec history): who applied it and, when the spec's
    folder is in a git repo, its commit, branch and whether the folder has local changes."""
    src: dict[str, Any] = {"by": os.environ.get("USER")}
    if spec_file is not None:
        d = str(Path(spec_file).resolve().parent)
        try:
            git = lambda *a: subprocess.run(["git", "-C", d, *a], check=False,
                                            capture_output=True, text=True, timeout=10)
            head = git("rev-parse", "HEAD")
            if head.returncode == 0:
                src["git_sha"] = head.stdout.strip()
                src["ref"] = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
                src["dirty"] = bool(git("status", "--porcelain", "--", ".").stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass  # no git: `by` only
    return {"X-Flock-Source": json.dumps(src)}


def resume_diff(old: SwarmSpec, spec: SwarmSpec) -> list[str]:
    """What `launch` on a stored swarm refuses: the folder's spec against the stored one,
    without the live replicas (the ramp rewrites them) and `world.dir` (absolute: the same
    folder copied elsewhere). Changes `spec` in place."""
    from flock.live import diff_specs, keep_replicas

    keep_replicas(old, spec)
    if old.world is not None and spec.world is not None:
        spec.world.dir = old.world.dir
    return diff_specs(old, spec)


def launch(api: Client, spec: SwarmSpec, spec_file: str, *, source: dict[str, str] | None = None,
           types: list[str] | None = None, agents: list[str] | None = None, adopt: bool = False,
           checkpoint: str | None = None, no_checkpoint: bool = False,
           preflight: bool | None = None, strict: bool = False) -> dict[str, Any]:
    """`flock launch` (docs/live.md "Lifecycle") on a spec read with `read_spec`. Missing on
    the controller: created (with `adopt`, first restored from a controller checkpoint, its
    own or `checkpoint`); refused when its bucket already holds agents of that name, unless
    `adopt`. Present: its holds (all, or `types`/`agents`) and release are lifted; refused
    when the spec differs from the stored one (replicas aside). Preflight runs by default
    when the spec names a provider-backed model. Returns, in this order, what applies:
    "warnings"; "adopted" (the adopt reply) or "adopt_skipped" (why, with `no_checkpoint`);
    "resumed" (the holds left) or "preflight" (the report) / "unverified" (models not probed),
    "tools" (the upload reply, plus "bytes") and "status" (the new swarm's). Refusals raise
    SpecError with these keys so far in `done`."""
    bundle, warnings = check_spec(spec, spec_file)
    done: dict[str, Any] = {"warnings": warnings}

    def refuse(*lines: str):
        raise SpecError(list(lines), done)

    def stored() -> SwarmSpec | None:
        try:
            return SwarmSpec.model_validate(api.swarm(spec.swarm)["spec"])
        except ClientError as e:
            if e.status == 404:
                return None
            if e.status is None:
                raise
            refuse(f"error {e.status}: {e.detail}")

    old = stored()
    if old is None and adopt:
        try:
            done["adopted"] = api.adopt(spec, checkpoint)
        except ClientError as e:
            if e.status != 404:
                refuse(f"error {e.status}: {e.detail}")
            if not no_checkpoint:
                refuse(f"refused: {e.detail}",
                       "pass --no-checkpoint to rebuild it from the bucket alone: agents restore "
                       "by name, but costs, loops, ramp steps and spawn links start over")
            done["adopt_skipped"] = e.detail
        else:
            old = stored()
    if old is not None:
        from flock.tools.bundle import version_of

        spec.tools_version = version_of(bundle.data) if bundle is not None else None
        if lines := resume_diff(old, spec):
            refuse(*lines, f"the folder's spec differs from swarm {spec.swarm}'s: run "
                   "`flock update` first, then `flock launch`")
        done["resumed"] = api.resume(spec.swarm, types=types, agents=agents)
        return done
    if preflight is None:
        preflight = provider_backed(spec)
    if preflight:
        rep = done["preflight"] = api.preflight(spec)
        if not rep["ok"] or (strict and int(rep.get("unverified") or 0)):
            refuse("preflight failed: not launching (fix the provider or pass --skip-preflight)")
    elif models := unprobed(spec):
        done["unverified"] = models
        if strict:
            refuse("--strict: not launching unverified models")
    spec.tools_version = None
    if bundle is not None:
        up = done["tools"] = {**api.upload_tools(spec.swarm, bundle.data), "bytes": len(bundle.data)}
        spec.tools_version = up["version"]
    done["status"] = api.create(spec, adopt=adopt, source=source or source_header(spec_file))
    return done


def update(api: Client, spec: SwarmSpec, spec_file: str, *, sources: dict[str, str] | None = None,
           preflight: bool | None = None, strict: bool = False, replicas: str = "keep",
           check: bool = False, source: dict[str, str] | None = None) -> list[str]:
    """`flock update`: apply a spec read with `read_spec` (and its folder's tools) to the
    running swarm (docs/live.md). Returns the
    lines to show (warnings, then what changed, or "no changes"); raises SpecError with the
    refusal when the folder cannot be applied, and then nothing was changed. Preflight runs
    by default only when the spec names a provider-backed model the running one does not.
    `replicas="keep"` keeps the live counts of existing types; `check` runs the controller's
    validation (dry run), stores nothing and names where each changed value came from.
    `sources` (from `read_spec`) names the origins for `check`; `source` replaces the
    `X-Flock-Source` header (--ref)."""
    from flock.live import diff_specs, keep_replicas, world_tool_changes

    if replicas not in ("keep", "spec"):
        raise SpecError([f"--replicas {replicas!r}: use keep or spec"])
    bundle, warnings = check_spec(spec, spec_file)
    try:
        old = SwarmSpec.model_validate(api.swarm(spec.swarm)["spec"])
    except ClientError as e:
        if e.status is None:
            raise
        raise SpecError([(f"no swarm {spec.swarm!r} is running on this controller: start it "
                          "with `flock launch`") if e.status == 404
                         else f"error {e.status}: {e.detail}"]) from None
    if preflight is None:  # --check probes every model; an update only the ones it adds
        preflight = bool(set(unprobed(spec)) - set(unprobed(old))) or (check and bool(unprobed(spec)))
    probed: list[str] = []
    if preflight:
        try:
            rep = api.preflight(spec)
        except ClientError as e:
            raise SpecError([f"preflight error {e.status}: {e.detail}"]) from None
        probed = [f"preflight {x['model']}: " + ("unverified" if x.get("unverified") else "ok"
                                                 if x["ok"] else "FAIL") for x in rep["results"]]
        bad = [f"preflight {x['model']}: {'unverified' if x.get('unverified') else 'FAIL'}"
               + (f" ({x['error']})" if x.get("error") else "")
               for x in rep["results"] if not x["ok"] and (strict or not x.get("unverified"))]
        if bad:
            raise SpecError([*bad, ("preflight failed: not updating (fix the provider or pass "
                                    "--skip-preflight)")])
    spec.tools_version = None
    names: tuple[list[str], list[str]] | None = None
    if bundle is not None:
        try:
            up = api.upload_tools(spec.swarm, bundle.data)
        except ClientError as e:
            raise SpecError([f"refused: tools upload ({e.status}): {e.detail}"]) from None
        spec.tools_version = up["version"]
        if spec.tools_version != old.tools_version:
            listed = api.tool_bundles(spec.swarm)
            before = next((b["tools"] for b in listed if b["version"] == old.tools_version), [])
            names = (before, up["tools"])
    kept = keep_replicas(old, spec, ramps_only=replicas == "spec")
    lines = diff_specs(old, spec, names)
    if check:  # `path: old -> new (spec.yaml:41 | run.env WORKER_MEM | default)`
        lines = [f"{x} ({source_of(sources or {}, x.split(':', 1)[0])})" if ": " in x
                 and not x.startswith(("tools:", "prompts changed:")) else x for x in lines]
    if not lines and not check:
        return [*warnings, *kept, "no changes"]
    try:
        out = api.update(spec, check=check, source=source or source_header(spec_file))
    except ClientError as e:
        raise SpecError([f"refused: {e.detail}"]) from None
    if check:  # the stored spec is untouched: the new manifest is in the reply
        new = SwarmSpec.model_validate(out["spec"])
        lines += out.get("deletes", [])
    else:
        new = SwarmSpec.model_validate(api.swarm(spec.swarm)["spec"])
    wt = world_tool_changes(old, new)
    return [*warnings, *(probed if check else []), *kept, *lines, *([wt] if wt else []),
            *(["check ok: nothing was applied"] if check else [])]
