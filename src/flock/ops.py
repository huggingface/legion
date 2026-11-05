"""Swarm operations shared by the `flock` CLI and Python callers: read an experiment folder
(`read_spec`), check it, `launch` it (create or resume; `up` first starts the run's controller
and world when none answers), `update` the running swarm and `stop` it (`down` ends the
controller and world after a release), all through `flock.client.Client`. Nothing here prints
or asks: results are returned, refusals raise SpecError, failed controller calls raise
ClientError.

    spec = read_spec("exp/spec.yaml", env_files)
    done = launch(Client(), spec, "exp/spec.yaml")
"""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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
COMMS_WAKES = {f"comms.{k}": k for k in ("mention", "thread", "dm", "channel")}  # = controller.api.COMMS_WAKE_EVENTS values


def _unfed(spec: SwarmSpec, ts, src: str) -> str | None:
    """Why `src` in a type's `loop.on` can never wake its loop; None when comms feeds it."""
    kind = COMMS_WAKES.get(src)
    if kind is None:
        return "no feeder in this version (comms.mention, comms.thread, comms.dm and comms.channel are fed)"
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
           checked: tuple[Any, list[str]] | None = None,
           say: Callable[[str], None] = lambda _: None, hold: bool = False) -> dict[str, Any]:
    """`flock launch` (docs/live.md "Lifecycle") on a spec read with `read_spec`. First, when
    no controller answers (always with `compute: local`): `up` starts the run's controller
    (and world), which restores its checkpoint. Missing on the controller: created; refused
    when its bucket already holds agents of that name. Present: its holds (all, or
    `types`/`agents`) and release are lifted; refused when the spec differs from the stored
    one (replicas aside). Preflight runs by default when the spec names a provider-backed
    model. Returns, in this order, what applies: "warnings"; "local" (the local run `up`
    started) or "up" (the jobs it started); "resumed" (the holds left) or "preflight" (the
    report) / "unverified" (models not probed), "tools" (the upload reply, plus "bytes") and
    "status" (the new swarm's). Refusals raise SpecError with these keys so far in `done`, as
    do failed controller calls. `checked`: `check_spec`'s result when the caller already has
    it. `env_files` also go to `up`; `say` gets its waiting lines. A local run is ended again
    when the launch fails; jobs are not (a refused resume needs the controller for
    `flock update`). `hold` (`flock launch --hold`, Slurm): only start the run's controller
    and world, the controller restoring every swarm held and released (as `flock stop
    --release`), so `flock update` can change the spec before any worker starts; nothing is
    created or resumed. "held" is the swarm's holds, or None when the controller has no such
    swarm. Refused when a controller already answers."""
    bundle, warnings = checked or check_spec(spec, spec_file)
    done: dict[str, Any] = {"warnings": warnings}
    try:
        try:
            if hold and spec.compute.kind != "slurm":
                raise SpecError([("--hold: Slurm runs only (their controller job restores a "
                                 "checkpoint)")], done)
            if hold and api.answers():
                raise SpecError([(f"--hold: a controller already serves this run: `flock stop "
                                 f"{spec.swarm} --release` first")], done)
            if spec.compute.kind == "local" or not api.answers():
                try:
                    started = up(api, spec, spec_file, env_files, say=say, hold=hold)
                except SpecError as e:
                    raise SpecError(e.lines, done) from None
                if started:
                    done["local" if spec.compute.kind == "local" else "up"] = started
            if hold:
                try:
                    status = api.swarm(spec.swarm)["status"]
                    done["held"] = {"swarm": spec.swarm, "held": status["held"]}
                except ClientError as e:
                    if e.status != 404:
                        raise
                    done["held"] = None
                return done
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
    `release`, once no worker runs: `down` -> plus {"down": its reply}; its SpecError (a failed
    final checkpoint: nothing ended) carries the keys so far. Flags that do not combine raise
    SpecError."""
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
        try:
            done["down"] = down(api, swarm)
        except SpecError as e:
            raise SpecError(e.lines, done) from None
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


# ---------------------------------------------------------------- the run's controller and world

UP_WAIT_S = 1800.0  # s `up` waits for the controller (a Slurm job may pend ~15 min for a node)
STOP_WAIT_S = 660.0  # s `down` waits for the world to save and end after its SIGTERM
JOB_POLL_S = 5.0
SLURM_ROLES = {"world": "WORLD", "controller": "CTL"}  # run.env <P>_PARTITION/_QOS/_CPUS/_MEM


def _run_dir(api: Client) -> Path | None:
    d = api.env.get("FLOCK_RUN_DIR")
    return Path(d).resolve() if d else None


def _values(api: Client, env_files=()) -> dict[str, str]:
    """The environment, then the env files (later wins), as `read_spec` reads them."""
    from flock.specrender import read_env_file

    return {**api.env, **{k: v for f in env_files for k, v in read_env_file(f).items()}}


def _secret_names(spec: SwarmSpec) -> list[str]:
    """The secrets the spec names: provider keys, the board invite, the world token, types'."""
    names = [*(p.api_key_env for p in spec.providers.values()),
             spec.comms.invite_env if spec.comms else None,
             spec.world.token_env if spec.world else None,
             *(s for ts in spec.types.values() for s in ts.secrets)]
    return list(dict.fromkeys(n for n in names if n))


def _job_name(role: str, run_dir: Path) -> str:
    """The run's Slurm job of `role` (one per run: the scripts' --dependency=singleton)."""
    return f"flock-{'ctl' if role == 'controller' else role}-{run_dir.name}"


def _slurm(*argv: str) -> subprocess.CompletedProcess:
    """sbatch/squeue/scancel with the controller's scrubbed environment (no token reaches
    Slurm's accounting); SpecError when the command fails."""
    from flock.controller.slurm import CMD_TIMEOUT, scrubbed_env

    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=CMD_TIMEOUT,
                           check=False, env=scrubbed_env())
    except (OSError, subprocess.SubprocessError) as e:
        raise SpecError([f"{argv[0]}: {type(e).__name__}: {e}"]) from None
    if r.returncode:
        raise SpecError([f"{' '.join(argv)}: exit {r.returncode}: {r.stderr.strip()[:300]}"])
    return r


def _state(name: str) -> str:
    """The state of this user's job `name` while it is pending or running ("RUNNING" when one
    runs), else "": squeue keeps listing ended jobs for a few minutes."""
    from flock.controller.slurm import RUNNING, SCHEDULING

    out = _slurm("squeue", "-h", "-u", getpass.getuser(), "-n", name, "-o", "%T").stdout.split()
    live = [s for s in out if s in SCHEDULING | RUNNING]
    return "RUNNING" if "RUNNING" in live else (live[0] if live else "")


def _scancel(name: str, *signal_only: str) -> None:
    _slurm("scancel", *signal_only, "-u", getpass.getuser(), "-n", name)


def _hf_api(env: Mapping[str, str]):  # tests replace it
    from huggingface_hub import HfApi

    return HfApi(token=env.get("HF_TOKEN") or None)


def up(api: Client, spec: SwarmSpec, spec_file: str, env_files=(), *,
       say: Callable[[str], None] = lambda _: None, hold: bool = False) -> dict | None:
    """Start what serves `spec` (`flock launch` when no controller answers). `compute: local`:
    `start_local` (the controller starts the world); its run, or None. `slurm`: sbatch the
    run directory's slurm/world.sbatch (when the spec has a world command) and
    slurm/controller.sbatch as jobs flock-world-<run> / flock-ctl-<run>, each unless that job
    is still queued (never a second one). `hf-jobs`: the controller job
    (flock.controller.jobdeploy; FLOCK_WHEEL_URL, network group FLOCK_NETWORK_GROUP or
    `flock`), unless the group has one. Then wait (UP_WAIT_S; `say` gets a line per check)
    until the controller answers, and the world its /manifest (hosts/world.host), and point
    `api` at the controller. It restores its checkpoint on start. Returns {"jobs": role -> job,
    "url"}. SpecError when it cannot start them or they do not come up. `hold` (Slurm): the
    controller job restores every swarm held (<run dir>/controller.hold; refused when that job
    is already queued: it would start without it)."""
    if spec.compute.kind == "local":
        return start_local(spec, spec_file, env_files, api.env)
    vals = _values(api, env_files)
    if spec.compute.kind == "slurm":
        return _up_slurm(api, spec, vals, say, hold)
    return _up_hf(api, spec, vals, say)


def _up_slurm(api: Client, spec: SwarmSpec, vals: Mapping[str, str], say, hold=False) -> dict:
    from flock.rundir import run_context

    d = _run_dir(api)
    if d is None:
        raise SpecError(["no controller answers and no run directory to start one from",
                         "`flock init RUN_DIR --from FOLDER`, then `flock --run RUN_DIR launch ...`"])
    roles = [*(["world"] if spec.world and spec.world.command else []), "controller"]
    jobs = {r: _job_name(r, d) for r in roles}
    if hold:  # controller.sbatch reads it at start and removes it once held
        if _state(jobs["controller"]):
            raise SpecError([(f"--hold: {jobs['controller']} is already queued and would start "
                              "without the hold: `scancel -n` it first")])
        (d / "controller.hold").write_text("flock launch --hold\n")
    for role, name in jobs.items():
        if _state(name):
            continue  # still pending or starting: wait for it
        script = d / "slurm" / f"{role}.sbatch"
        if not script.is_file():
            raise SpecError([f"{script} is missing: run `flock init {d}` (it copies the job scripts)"])
        (d / "hosts" / f"{role}.host").unlink(missing_ok=True)  # an ended job's
        p = SLURM_ROLES[role]
        opts = {"-p": vals.get(f"{p}_PARTITION") or vals.get("SLURM_PARTITION"),
                "--qos": vals.get(f"{p}_QOS"), "-c": vals.get(f"{p}_CPUS"),
                "--mem": vals.get(f"{p}_MEM")}
        _slurm("sbatch", "-J", name, "-o", str(d / "logs" / "%x-%j.out"),
               *(a for k, v in opts.items() if v for a in (k, v)), str(script), str(d))
        say(f"submitted {name}")
    end = time.monotonic() + UP_WAIT_S
    while True:
        api.env = {**api.env, **run_context(d)[0]}  # the controller's host:port once written
        waiting = [r for r in jobs if not (d / "hosts" / f"{r}.host").is_file()
                   or (r == "controller" and not api.answers())]
        if not waiting:
            return {"jobs": jobs, "url": api.env["FLOCK_URL"]}
        if dead := [jobs[r] for r in waiting if not _state(jobs[r])]:
            raise SpecError([f"job {n} ended before it came up: see {d}/logs/{n}-*.out"
                             for n in dead])
        if time.monotonic() >= end:
            raise SpecError([f"{' and '.join(waiting)} not up after {UP_WAIT_S:g} s",
                             f"jobs {', '.join(jobs.values())} still queued: see `squeue`"])
        say(f"waiting for the {' and '.join(waiting)} ({', '.join(jobs.values())})")
        _sleep(JOB_POLL_S)


def _up_hf(api: Client, spec: SwarmSpec, vals: Mapping[str, str], say) -> dict:
    from flock.controller import jobdeploy as jd

    wheel = vals.get("FLOCK_WHEEL_URL")
    if not wheel:
        raise SpecError(["no controller answers, and no FLOCK_WHEEL_URL to start one with",
                         "publish a wheel (scripts/publish-wheel.sh) or deploy it (docs/deploy-job.md)"])
    if "HF_TOKEN" in _secret_names(spec) and not vals.get("FLOCK_WORKER_HF_TOKEN"):
        raise SpecError(["the spec sends HF_TOKEN to workers: the controller's job-launching token",
                         "set FLOCK_WORKER_HF_TOKEN (to $HF_TOKEN to opt in) or name another key"])
    try:
        hf = _hf_api(vals)
        ns = spec.namespace or hf.whoami()["name"]
        group = vals.get("FLOCK_NETWORK_GROUP") or jd.DEFAULT_GROUP
        if running := jd._running_controllers(hf, ns, group):
            job = running[0].id  # still starting: wait for it, never two in one group
        else:
            secrets = jd.collect_secrets(vals, [n for n in _secret_names(spec) if vals.get(n)])
            job = hf.run_job(**jd.run_job_args(
                wheel_url=wheel, group=group, namespace=ns, flavor="cpu-upgrade", timeout="48h",
                state_url=vals.get("FLOCK_STATE_URL") or jd.default_state_url(wheel, group),
                job_labels=vals.get("JOB_LABELS", ""), secrets=secrets)).id
            say(f"started controller job {job} (group {group})")
    except SystemExit as e:  # collect_secrets: a missing FLOCK_TOKEN or HF_TOKEN
        raise SpecError([str(e)]) from None
    except Exception as e:  # noqa: BLE001 - a Hub error or a bad wheel URL: the launch's answer
        raise SpecError([f"controller job: {type(e).__name__}: {e}"]) from None
    url = f"https://{job}--{jd.API_PORT}.hf.jobs"
    api.env = {**api.env, "FLOCK_URL": url}
    if (d := _run_dir(api)) is not None:  # `flock --run` finds it
        (d / "hosts" / "controller.host").write_text(url + "\n")
    end = time.monotonic() + UP_WAIT_S
    while not api.answers():
        if time.monotonic() >= end:
            raise SpecError([f"controller job {job} does not answer at {url} after {UP_WAIT_S:g} s"])
        say(f"waiting for controller job {job} (it installs the wheel first)")
        _sleep(JOB_POLL_S)
    return {"jobs": {"controller": job}, "url": url}


def down(api: Client, swarm: str) -> dict[str, Any]:
    """End what serves `swarm` once it is released (`flock stop --release`), in this order:
    1. a final controller checkpoint (POST /checkpoint). When it fails, SpecError and nothing
       is ended: the controller holds the only copy of its state;
    2. the world's clean stop (Slurm: a running flock-world-<run> gets `scancel --signal=TERM
       --batch` and is waited for: the world saves on SIGTERM);
    3. the jobs cancelled (Slurm: by name; HF Jobs: the controller job of the network group,
       jobdeploy's cancel and wait), the host files removed so `launch` waits for new ones.
    Returns {"checkpoint": files pushed, "ended": [what]}, or {"kept": why}: a local run (its
    agents' workdirs live only in its run directory), a controller this run has no job for
    (Slurm flock-ctl-<run>, an HF Job URL), or one serving swarms that are not released."""
    spec = api.swarm(swarm)["spec"]
    kind, d = spec["compute"]["kind"], _run_dir(api)
    if kind == "local":
        return {"kept": "a local run keeps its controller (`flock stop --delete` ends it)"}
    if kind == "hf-jobs":
        ours = (urlsplit(api.env.get("FLOCK_URL") or "").hostname or "").endswith(".hf.jobs")
    else:
        ours = d is not None and bool(_state(_job_name("controller", d)))
    if not ours:
        return {"kept": "no job of this run serves it (Slurm flock-ctl-<run>, an HF Job URL)"}
    if others := [s["swarm"] for s in api.swarms() if s["swarm"] != swarm and not s.get("released")]:
        return {"kept": f"it also serves {', '.join(others)}"}
    try:
        files = api.checkpoint()["files"]
    except ClientError as e:
        raise SpecError([f"final checkpoint failed: {e}",
                         "nothing was ended: the controller keeps the only copy of its state",
                         f"fix it, then `flock stop {swarm} --release` again"]) from None
    ended: list[str] = []
    if kind == "slurm":
        world, ctl, note = _job_name("world", d), _job_name("controller", d), ""
        if (queued := _state(world)) == "RUNNING":  # Slurm signals only a running job
            _scancel(world, "--signal=TERM", "--batch")  # it saves, exits, the job ends
            if not _ends(lambda: _state(world)):
                note = f" (cancelled: running {STOP_WAIT_S:g} s after SIGTERM)"
        for name in (world, ctl):  # a pending world is just cancelled
            if _state(name):
                _scancel(name)
        ended += [*([world + note] if queued else []), ctl]
        for role in SLURM_ROLES:
            (d / "hosts" / f"{role}.host").unlink(missing_ok=True)
    else:
        from flock.controller import jobdeploy as jd

        vals = _values(api, api.env_files)
        group = vals.get("FLOCK_NETWORK_GROUP") or jd.DEFAULT_GROUP
        try:
            hf = _hf_api(vals)
            gone = jd._cancel_and_wait(hf, spec.get("namespace") or hf.whoami()["name"], group,
                                       say=lambda _: None)  # SIGTERM: one more final push
        except Exception as e:  # noqa: BLE001 - a Hub error: say so, the checkpoint is pushed
            raise SpecError([f"cancel group {group}'s controller: {type(e).__name__}: {e}"]) from None
        ended.append(f"group {group}'s controller job" + ("" if gone else " (still running)"))
        if d is not None:
            (d / "hosts" / "controller.host").unlink(missing_ok=True)
    return {"checkpoint": files, "ended": ended}


def _ends(alive: Callable[[], Any]) -> bool:
    """Poll `alive` until it is falsy (True) or STOP_WAIT_S have passed (False)."""
    end = time.monotonic() + STOP_WAIT_S
    while alive():
        if time.monotonic() >= end:
            return False
        _sleep(JOB_POLL_S)
    return True
