"""Swarm operations shared by the `flock` CLI and Python callers: read an experiment folder
(`read_spec`), check it, `launch` it (create or resume) and `update` the running swarm,
all through `flock.client.Client`. Nothing here prints or asks: results are returned,
refusals raise SpecError, failed controller calls raise ClientError.

    spec = read_spec("exp/spec.yaml", env_files)
    done = launch(Client(), spec, "exp/spec.yaml")
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from flock.client import Client, ClientError, local_run
from flock.models import SwarmSpec, comms_tools, loop_refusals
from flock.registry import HARNESSES
from flock.specrender import render, source_of


class SpecError(Exception):
    """A spec that cannot be applied: the lines say why (printed to stderr by the CLI).
    `done`: what `launch` did or found before refusing (the keys it returns), to show first."""

    def __init__(self, lines: list[str], done: dict[str, Any] | None = None) -> None:
        super().__init__("; ".join(lines))
        self.lines, self.done = lines, done or {}


def _read_rel(base: Path, rel: str, where: str) -> str:
    """The text of `rel` (relative to the spec's folder `base`); SpecError when missing."""
    src = base / rel  # an absolute `rel` replaces `base`
    if not src.is_file():
        raise SpecError([f"{where} {src} not found"])
    return src.read_text(encoding="utf-8")


def read_spec(spec_file: str, env_files=(), sources: dict[str, str] | None = None) -> SwarmSpec:
    """Read a spec (${VAR} from --env files and the environment substituted:
    flock.specrender, `sources` filled) and inline each type's
    system_prompt_file (plus system_prompt_append_file) and loop prompt files, and the comms
    kickoff and protocol files (relative to the spec file). Raises SpecError."""
    path = Path(spec_file)
    try:
        spec = SwarmSpec.model_validate(render(path, env_files, sources))
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
    base = path.parent
    for name, ts in spec.types.items():
        if ts.system_prompt_file and ts.system_prompt is None:
            ts.system_prompt = _read_rel(base, ts.system_prompt_file,
                                         f"type {name}: system_prompt_file")
        if ts.system_prompt_append_file:
            ts.system_prompt = ((ts.system_prompt or "").rstrip("\n") + "\n\n" + _read_rel(
                base, ts.system_prompt_append_file, f"type {name}: system_prompt_append_file")
                                ).lstrip("\n")
        for lp, where in ((ts.loop, "loop"), (ts.loop and ts.loop.final, "loop.final")):
            if lp is not None and lp.prompt_file and lp.prompt is None:
                lp.prompt = _read_rel(base, lp.prompt_file, f"type {name}: {where}: prompt_file")
    c = spec.comms
    if c is not None:  # kickoff and protocol files, next to the spec (docs/comms.md)
        if c.kickoff_file:
            c.kickoff = _read_rel(base, c.kickoff_file, "comms.kickoff_file:")
        c.protocols_text = [_read_rel(base, f, "comms.protocols:") for f in c.protocols]
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
EXTRA_NOT_SENT = frozenset(name for name, h in HARNESSES.items() if not h.sends_extra)


def extra_warnings(spec: SwarmSpec) -> list[str]:
    from flock.worker.adapters.pi import env_dropped

    return [f"warning: types.{name}.env.{k}: dropped by pi isolation (use `secrets`)"
            for name, ts in spec.types.items() if ts.harness.value == "pi"
            for k in env_dropped(ts.env, ts.resolved_model and ts.resolved_model.provider.api_key_env)] + [f"warning: type {name}: `extra` is kept in the spec but not sent by the "
            f"{ts.harness.value} harness (docs/open-models.md)"
            for name, ts in spec.types.items()
            if ts.harness.value in EXTRA_NOT_SENT and ts.resolved_model is not None
            and ts.resolved_model.extra]


def preflight_lines(results: list[dict]) -> list[str]:
    """One line per probed model: `MODEL: ok|unverified|FAIL[ N ms][ (error)]`."""
    return [f"{r['model']}: " + ("unverified" if r.get("unverified") else "ok" if r["ok"] else "FAIL")
            + (f" {r['latency_ms']} ms" if r.get("latency_ms") is not None else "")
            + (f" ({r['error']})" if r.get("error") else "") for r in results]


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


def is_swarm_name(target: str) -> bool:
    """`flock launch TARGET`: no file, no path and no .yaml name: a stored swarm to resume."""
    return not Path(target).exists() and "/" not in target \
        and not target.endswith((".yaml", ".yml"))


def launch(api: Client, spec: SwarmSpec, spec_file: str, *, env_files=(),
           source: dict[str, str] | None = None, types: list[str] | None = None, agents: list[str] | None = None,
           preflight: bool | None = None, strict: bool = False,
           checked: tuple[Any, list[str]] | None = None) -> dict[str, Any]:
    """`flock launch` (docs/live.md "Lifecycle") on a spec read with `read_spec`. Missing on
    the controller: created; refused when its bucket already holds agents of that name.
    Present: its holds (all, or `types`/`agents`) and release are lifted; refused
    when the spec differs from the stored one (replicas aside). Preflight runs by default
    when the spec names a provider-backed model. Returns, in this order, what applies:
    "warnings";
    "resumed" (the holds left) or "preflight" (the report) / "unverified" (models not probed),
    "tools" (the upload reply, plus "bytes") and "status" (the new swarm's). Refusals raise
    SpecError with these keys so far in `done`, as do failed controller calls. `checked`:
    `check_spec`'s result when the caller already has it. `compute: local`: first
    `start_local` (with `env_files`), "local" its run; ended again when the launch fails."""
    bundle, warnings = checked or check_spec(spec, spec_file)
    done: dict[str, Any] = {"warnings": warnings}
    if spec.compute.kind == "local" and (run := start_local(spec, spec_file, env_files, api.env)):
        done["local"] = run
    try:
        try:
            return _launch(api, spec, spec_file, bundle, done, source, types, agents,
                           preflight, strict)
        except ClientError as e:
            raise SpecError([str(e)], done) from None
    except BaseException:  # also Ctrl-C during the preflight
        if "local" in done:  # nothing created: do not leave a controller behind
            end_local(done["local"], api.env)
        raise


def _launch(api, spec, spec_file, bundle, done, source, types, agents, preflight, strict) -> dict[str, Any]:

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
    done["status"] = api.create(spec, source=source or source_header(spec_file))
    return done


def update(api: Client, spec: SwarmSpec, spec_file: str, *, sources: dict[str, str] | None = None,
           preflight: bool | None = None, strict: bool = False, replicas: str = "keep",
           check: bool = False, source: dict[str, str] | None = None) -> dict[str, Any]:
    """`flock update`: apply a spec read with `read_spec` (and its folder's tools) to the
    running swarm (docs/live.md). Returns {"warnings", "probed" (preflight lines, with
    `check`), "kept" (live replica counts kept), "diff" (what changed), "deletes" (what the
    controller would drop, with `check`), "world_tools" (a line or None), "applied"}; raises
    SpecError with the refusal when the folder cannot be applied, and then nothing was changed. Preflight runs
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
        probed = [f"preflight {x}" for x in preflight_lines(rep["results"])]
        bad = [line for line, x in zip(probed, rep["results"], strict=True)
               if not x["ok"] and (strict or not x.get("unverified"))]
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
    out: dict[str, Any] = {"warnings": warnings, "probed": probed if check else [], "kept": kept,
                           "diff": lines, "deletes": [], "world_tools": None, "applied": False}
    if not lines and not check:
        return out
    try:
        reply = api.update(spec, check=check, source=source or source_header(spec_file))
    except ClientError as e:
        raise SpecError([f"refused: {e.detail}"]) from None
    if check:  # the stored spec is untouched: the new manifest is in the reply
        new = SwarmSpec.model_validate(reply["spec"])
        out["deletes"] = reply.get("deletes", [])
    else:
        new = SwarmSpec.model_validate(api.swarm(spec.swarm)["spec"])
    out["world_tools"], out["applied"] = world_tool_changes(old, new), not check
    return out


def stop(api: Client, swarm: str, *, types: list[str] | None = None,
         agents: list[str] | None = None, interrupt: bool = False, reason: str = "",
         release: bool = False, final: bool = False, delete: bool = False,
         confirm: Callable[[], Any] | None = None,
         progress: Callable[[int], None] | None = None) -> dict[str, Any]:
    """`flock stop` (docs/live.md "Lifecycle"). Default: hold the swarm (or `types`/`agents`;
    `interrupt` also cuts running turns) -> {"held": `api.stop`'s reply}; `release` also
    drains every worker and waits (`release_wait`) -> plus {"unsynced": name -> error}.
    `final`: stop the loops of `types` (default every type with one) after a final round ->
    {"loops": replies}. `delete` (after `confirm()`): drop the swarm from the controller ->
    {"deleted": reply}, or end this directory's local run of it -> {"ended": its record}.
    Flags that do not combine raise SpecError."""
    if delete:
        if final or release or interrupt or types or agents:
            raise SpecError(["--delete takes only --yes"])
        if confirm is not None:
            confirm()
        if (run := local_run(api.env)) is not None and run.get("swarm") == swarm:
            end_local(run, api.env)  # no DELETE first: the bundle is written from the rows it would drop
            return {"ended": run}
        return {"deleted": api.delete(swarm)}
    if final:
        if release or interrupt or agents:
            raise SpecError(["--final takes only --type"])
        spec = api.swarm(swarm)["spec"]
        return {"loops": [api.stop_loop(swarm, t) for t in types
                          or [n for n, ts in spec["types"].items() if ts.get("loop")]]}
    if release and (types or agents):
        raise SpecError(["--release frees the whole swarm: no --type or --agent"])
    done = {"held": api.stop(swarm, types=types, agents=agents, interrupt=interrupt,
                             reason=reason, release=release)}
    if release:
        done["unsynced"] = release_wait(api, swarm, progress=progress)
    return done


RELEASE_POLL = 5.0  # s between worker checks while `release_wait` waits
_sleep = time.sleep  # tests replace it


def release_wait(api: Client, swarm: str, *,
                 progress: Callable[[int], None] | None = None) -> dict[str, str]:
    """After `api.stop(swarm, release=True)`: wait until no worker of the swarm runs
    (`progress` gets the count still running at each check), then return the agents whose
    final sync failed (name -> error; empty when every agent synced)."""
    while live := [w for w in api.workers(swarm) if w["job_id"]]:
        if progress is not None:
            progress(len(live))
        _sleep(RELEASE_POLL)
    return (api.swarm(swarm)["status"].get("released") or {}).get("unsynced") or {}


LOCAL_START_S = 120.0  # s `start_local` waits for a local controller to come up
LOCAL_DIR = Path(".flock")  # the local run's data dir and launch.log, in the working directory


def start_local(spec: SwarmSpec, spec_file: str, env_files=(),
                env: Mapping[str, str] = os.environ) -> dict | None:
    """`compute: local`: start this machine's controller and workers detached
    (flock.controller.localrun, output in .flock/launch.log) unless a live one in this directory
    already serves the swarm; wait for its `.flock/current`. Returns the run started (its
    `.flock/current` record), or None; SpecError when another swarm's local run is up or the
    controller does not start (the log's tail in the lines)."""
    if env.get("FLOCK_URL"):
        return None  # pointed at a controller explicitly
    if (run := local_run(env)) is not None:
        if run.get("swarm") != spec.swarm:
            raise SpecError([(f"a local run of {run.get('swarm')} is up in this directory: "
                              f"`flock stop {run.get('swarm')} --release` first")])
        return None
    LOCAL_DIR.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "flock.controller.localrun", spec_file, "--no-apply",
           "--data-dir", str(LOCAL_DIR), *(a for f in env_files for a in ("--env", f))]
    with (LOCAL_DIR / "launch.log").open("w") as out:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                                start_new_session=True, env=dict(env))
    end = time.monotonic() + LOCAL_START_S
    while time.monotonic() < end:
        if (run := local_run(env)) is not None and run.get("pid") == proc.pid:
            return run
        if proc.poll() is not None:
            break
        _sleep(0.2)
    else:
        proc.terminate()
    log = (LOCAL_DIR / "launch.log").read_text(errors="replace").strip().splitlines()[-20:]
    raise SpecError([*log, "the local controller did not start"])


def end_local(run: dict, env: Mapping[str, str] = os.environ) -> None:
    """SIGTERM a background local run and wait (up to LOCAL_START_S) until it is gone: it
    drains what is left, writes the run bundle (<run_dir>/run.json) and exits (docs/local.md)."""
    with contextlib.suppress(ProcessLookupError):
        os.kill(run["pid"], signal.SIGTERM)
    end = time.monotonic() + LOCAL_START_S
    while local_run(env) is not None and time.monotonic() < end:
        _sleep(0.2)
