"""`flock` CLI: argument parsing, spec and file loading, local-run processes and output over
`flock.client.Client` (connection config: see flock/client.py)."""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from pydantic import ValidationError
from typer.core import TyperGroup

from flock.client import Client, ClientError, local_run
from flock.models import SwarmSpec, comms_tools, loop_refusals
from flock.specrender import at_ref, render, source_of


class _Group(TyperGroup):
    """Every command: a refused or failed controller call prints its line and exits 1."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except ClientError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1) from None


app = typer.Typer(cls=_Group, help="Host coding-harness agent swarms on HF Jobs.",
                  no_args_is_help=True)
_sleep = time.sleep  # tests replace it


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


def _table(rows: list[list[Any]], header: list[str]) -> None:
    cells = [header] + [[("" if v is None else str(v)) for v in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    for r in cells:
        typer.echo("  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)).rstrip())


def _states(d: dict[str, int]) -> str:
    return " ".join(f"{k}:{v}" for k, v in sorted(d.items())) or "-"


def _print_status(st: dict) -> None:
    typer.echo(
        f"{st['swarm']}  agents={st['agents_total']} workers={st['workers_total']} "
        f"pending={st['pending_prompts']} cost=${st['cost_usd']:.2f}"
        + (f" gen {st['generation']}" if st.get("generation") else "")
    )
    rows = [
        [
            t["name"],
            t["health"],
            t["replicas"],
            _states(t["agents_by_state"]),
            _states(t["workers_by_state"]),
            f"${t['cost_usd']:.2f}",
        ]
        for t in st["types"].values()
    ]
    _table(rows, ["type", "health", "replicas", "agents", "workers", "cost"])
    held = st.get("held") or {}
    for kind in ("types", "agents"):
        for name, h in (held.get(kind) or {}).items():
            typer.echo(f"HELD {kind[:-1]} {name}: {h['reason'] or '-'} (by {h['by']}"
                       f"{', auto' if h.get('auto') else ''}, since {h['since'][:19]}Z)")
    if rel := st.get("released"):
        typer.echo(f"RELEASED since {rel['since'][:19]}Z: workers drained, nothing launches "
                   "until `flock launch`")
        for name, err in (rel.get("unsynced") or {}).items():
            typer.echo(f"NOT synced: {name}: {err}")
    for t in st["types"].values():
        if t.get("loop"):
            typer.echo(_loop_line(t["loop"]))
    if st.get("budget_stop"):
        what = ("final reserve reached: loops stopped, final rounds"
                if st["budget_stop"] == "final_reserve" else "every agent stopped, prompts refused")
        typer.echo(f"warning: limits.budget_usd: {what}")
    for t in st["types"].values():
        if t.get("launch_error"):
            what = ("launch refused, not retrying until the next apply"
                    if t["health"] == "launch_refused" else "last launch error, retrying")
            typer.echo(f"warning: type {t['name']}: {what}: {t['launch_error']}")
    for pv in (st.get("providers") or {}).values():
        cap = f"/{pv['max_working']}" if pv.get("max_working") else ""
        kind = pv["type"] + (" self-hosted" if pv["self_hosted"] else "")
        typer.echo(f"provider {pv['name']} ({kind}): working {pv['working']}{cap}, "
                   f"types {','.join(pv['types']) or '-'}")
        if pv.get("zero_usage_agents"):
            typer.echo(f"warning: provider {pv['name']}: {pv['zero_usage_agents']} agents finished "
                       "turns with zero reported tokens; their cost is unknown (shown as $0)")


def _loop_line(lp: dict) -> str:
    rounds = (f"rounds {lp['rounds_min']}-{lp['rounds_max']}" if lp["rounds_min"] != lp["rounds_max"]
              else f"round {lp['rounds_max']}")
    line = (f"loop {lp['type']}: {lp['state']} ({lp['mode']}, every {lp['every']}), {rounds}, "
            f"{lp['agents_done']}/{lp['agents']} agents done")
    if lp.get("reason"):
        line += f", stopped: {lp['reason']}"
    if lp["final_sent"] or lp["state"] == "final":
        line += f", final {lp['final_done']}/{lp['final_sent']} done"
    return line


class SpecError(Exception):
    """A spec that cannot be applied: the lines say why (printed to stderr by the CLI)."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__("; ".join(lines))
        self.lines = lines


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


def load_spec(spec_file: str, env_files=()) -> SwarmSpec:
    """`read_spec`, exiting 1 with the reasons on stderr."""
    try:
        return read_spec(spec_file, env_files)
    except SpecError as e:
        for line in e.lines:
            typer.echo(line, err=True)
        raise typer.Exit(1) from None


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
            typer.echo(f"world.dir: {w.dir} not found", err=True)
            raise typer.Exit(1)
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


def _check_loops(spec: SwarmSpec) -> None:
    """Refuse loop settings this version cannot run; say which `on:` sources have no feeder."""
    refused, warnings = loop_problems(spec)
    for line in refused or warnings:
        typer.echo(line, err=True)
    if refused:
        raise typer.Exit(1)


def _provider_backed(spec: SwarmSpec) -> bool:
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


def _warn_extra(spec: SwarmSpec) -> None:
    for line in extra_warnings(spec):
        typer.echo(line, err=True)


def _warn_unverified(models: list[str], why: str) -> None:
    for m in models:
        typer.echo(f"warning: {m}: unverified ({why})", err=True)


def _unprobed(spec: SwarmSpec) -> list[str]:
    return sorted({ts.resolved_model.qualified for ts in spec.types.values()
                   if ts.resolved_model is not None})


def _preflight(spec: SwarmSpec, strict: bool) -> bool:
    """POST /swarms/{swarm}/preflight with the spec; print one row per model. True when no probe
    failed and, with `strict`, every model was actually probed. Never silent about a model that
    was not probed."""
    rep = Client().preflight(spec)
    rows = [
        [r["model"], ",".join(r["types"]), r["api"],
         "unverified" if r.get("unverified") else "ok" if r["ok"] else "FAIL",
         "yes" if r["tool_call_seen"] else "no", "yes" if r.get("follow_up_ok") else "no",
         "" if r["latency_ms"] is None else r["latency_ms"], r["error"] or ""]
        for r in rep["results"]
    ]
    if rows:
        _table(rows, ["model", "types", "api", "result", "tool_call", "follow_up", "latency_ms",
                      "error"])
        for r in rep["results"]:
            if r.get("warning"):
                typer.echo(f"warning: {r['model']}: {r['warning']}", err=True)
    else:
        typer.echo("preflight: no provider-backed models in this spec")
    unverified = int(rep.get("unverified") or 0)
    if unverified:
        typer.echo(f"warning: {unverified} model(s) unverified"
                   + (" (--strict: counted as failure)" if strict else ""), err=True)
    return bool(rep["ok"]) and not (strict and unverified)


ENV_OPT = typer.Option(None, "--env", help="NAME=value file for ${VAR} in the spec (repeatable; "
                       "later files win, and every file wins over the process environment).")
REF_OPT = typer.Option(None, "--ref", metavar="GIT_REF[:PATH]",
                       help="Apply the folder as committed at GIT_REF (git archive); PATH is "
                            "repo-relative, default the given folder's own path.")


def _at_ref(spec_file: str, ref: str | None):
    """(spec_file, X-Flock-Source header or None) for --ref; exit 1 on a bad ref."""
    if ref is None:
        return spec_file, None
    try:
        spec, header = at_ref(spec_file, ref)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    return str(spec), header


def _spec_file(target: str) -> str:
    path = Path(target)
    return str(path / "spec.yaml" if path.is_dir() else path)


def _fail(lines: list[str]) -> None:
    for line in lines:
        typer.echo(line, err=True)
    raise typer.Exit(1)


@app.command()
def launch(
    target: str = typer.Argument(..., metavar="FOLDER_OR_SPEC_OR_SWARM"),
    type_: Annotated[list[str] | None, typer.Option(
        "--type", help="resume: lift only this type's hold (repeatable)")] = None,
    agent: Annotated[list[str] | None, typer.Option(
        "--agent", help="resume: lift only this agent's hold (repeatable)")] = None,
    adopt: bool = typer.Option(
        False, "--adopt", help="Take over a swarm this controller does not know: restore it "
                               "from the controller checkpoint in its bucket."),
    checkpoint: str = typer.Option(
        None, "--checkpoint", metavar="STATE_URL",
        help="With --adopt: the checkpoint to read (an earlier run's FLOCK_STATE_URL, e.g. "
             "hf://buckets/ns/b/controller/<RUN>); default the controller's own, else "
             "<bucket>/controller/<swarm>."),
    no_checkpoint: bool = typer.Option(
        False, "--no-checkpoint", help="With --adopt and no checkpoint: create the swarm from "
                                       "the spec; agents restore from the bucket by name."),
    preflight: bool = typer.Option(
        None, "--preflight/--skip-preflight",
        help="Probe provider-backed models before creating (default: on when the spec has any)."),
    strict: bool = typer.Option(
        False, "--strict", help="Treat models that were not probed (unverified) as failures."),
    env_files: list[str] | None = ENV_OPT,
    ref: str | None = REF_OPT,
) -> None:
    """Start a swarm from an experiment folder (or spec file), or resume a stopped one
    (docs/live.md "Lifecycle"). Missing: creates it (refused when its bucket already holds
    agents of that name, unless --adopt). Present: clears `flock stop`'s holds and release;
    refused when the folder's spec differs from the stored one (replicas aside): run
    `flock update` first. Given a swarm name instead of a folder: resumes it."""
    if ref is None and not Path(target).exists() and "/" not in target \
            and not target.endswith((".yaml", ".yml")):
        _resume(target, type_, agent)
        return
    spec_file, ref_header = _at_ref(_spec_file(target), ref)
    source = ref_header or source_header(spec_file)  # before any write lands in the folder
    spec = load_spec(spec_file, env_files or ())
    _check_loops(spec)
    _warn_extra(spec)
    bundle = _tools_bundle(spec, spec_file)
    started = _start_local(spec, spec_file, env_files or ()) if spec.compute.kind == "local" \
        else None
    try:
        _launch(spec, spec_file, bundle, source, type_, agent, adopt, checkpoint, no_checkpoint,
                preflight, strict)
    except BaseException:  # also Ctrl-C during the preflight
        if started is not None:  # nothing created: do not leave a controller behind
            _end_local(started)
        raise


def _stored(api: Client, swarm: str) -> SwarmSpec | None:
    """The controller's spec of `swarm`, None when it has none; exit 1 when it refuses."""
    try:
        return SwarmSpec.model_validate(api.swarm(swarm)["spec"])
    except ClientError as e:
        if e.status == 404:
            return None
        if e.status is None:
            raise
        _fail([f"error {e.status}: {e.detail}"])


def _launch(spec, spec_file, bundle, source, type_, agent, adopt, checkpoint, no_checkpoint,
            preflight, strict) -> None:
    api = Client()
    old = _stored(api, spec.swarm)
    if old is None and adopt:
        try:
            a = api.adopt(spec, checkpoint)
        except ClientError as e:
            if e.status != 404:
                _fail([f"error {e.status}: {e.detail}"])
            elif not no_checkpoint:
                _fail([f"refused: {e.detail}",
                       ("pass --no-checkpoint to rebuild it from the bucket alone: agents restore "
                       "by name, but costs, loops, ramp steps and spawn links start over")])
            typer.echo(f"{e.detail}: creating from the spec (agents restore from the bucket)")
        else:
            typer.echo(f"adopted {spec.swarm}: {a['agents']} agents from {a['checkpoint']}")
            old = _stored(api, spec.swarm)
    if old is not None:
        from flock.tools.bundle import version_of

        spec.tools_version = version_of(bundle.data) if bundle is not None else None
        if lines := resume_diff(old, spec):
            _fail([*lines, (f"the folder's spec differs from swarm {spec.swarm}'s: run "
                           "`flock update` first, then `flock launch`")])
        _resume(spec.swarm, type_, agent)
        return
    if preflight is None:
        preflight = _provider_backed(spec)
    if preflight:
        if not _preflight(spec, strict):
            _fail(["preflight failed: not launching (fix the provider or pass --skip-preflight)"])
    elif unprobed := _unprobed(spec):
        _warn_unverified(unprobed, "preflight skipped")
        if strict:
            _fail(["--strict: not launching unverified models"])
    spec.tools_version = None
    if bundle is not None:
        up = api.upload_tools(spec.swarm, bundle.data)
        spec.tools_version = up["version"]
        names = ", ".join(up["tools"]) or "none"
        typer.echo(f"tools: version {up['version']} ({len(bundle.data)} bytes): {names}")
    _print_status(api.create(spec, adopt=adopt, source=source))


LOCAL_START_S = 120.0  # s `launch` waits for a local controller to come up


def _start_local(spec: SwarmSpec, spec_file: str, env_files) -> dict | None:
    """`compute: local`: start this machine's controller and workers detached
    (flock.controller.localrun, output in .flock/launch.log) unless a live one in this directory
    already serves the swarm; wait for its `.flock/current`. Returns the run started, or None."""
    if os.environ.get("FLOCK_URL"):
        return None  # pointed at a controller explicitly
    if (run := local_run()) is not None:
        if run.get("swarm") != spec.swarm:
            _fail([(f"a local run of {run.get('swarm')} is up in this directory: "
                    f"`flock stop {run.get('swarm')} --release` first")])
        return None
    data_dir = Path(".flock")
    data_dir.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "flock.controller.localrun", spec_file, "--no-apply",
           "--data-dir", str(data_dir), *(a for f in env_files for a in ("--env", f))]
    with (data_dir / "launch.log").open("w") as out:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                                start_new_session=True)
    end = time.monotonic() + LOCAL_START_S
    while time.monotonic() < end:
        if (run := local_run()) is not None and run.get("pid") == proc.pid:
            typer.echo(f"local run {run['run_dir']} at {run['url']} (log: {data_dir}/launch.log)")
            return run
        if proc.poll() is not None:
            break
        _sleep(0.2)
    else:
        proc.terminate()
    log = (data_dir / "launch.log").read_text(errors="replace").strip().splitlines()[-20:]
    _fail([*log, "the local controller did not start"])
    return None


def _end_local(run: dict) -> None:
    """SIGTERM the background local run: it drains what is left, writes the run bundle and
    exits (docs/local.md)."""
    with contextlib.suppress(ProcessLookupError):
        os.kill(run["pid"], signal.SIGTERM)
    end = time.monotonic() + LOCAL_START_S
    while local_run() is not None and time.monotonic() < end:
        _sleep(0.2)
    typer.echo(f"local run ended; bundle: {run['run_dir']}/run.json")


def resume_diff(old: SwarmSpec, spec: SwarmSpec) -> list[str]:
    """What `launch` on a stored swarm refuses: the folder's spec against the stored one,
    without the live replicas (the ramp rewrites them) and `world.dir` (absolute: the same
    folder copied elsewhere). Changes `spec` in place."""
    from flock.live import diff_specs, keep_replicas

    keep_replicas(old, spec)
    if old.world is not None and spec.world is not None:
        spec.world.dir = old.world.dir
    return diff_specs(old, spec)


def _resume(swarm: str, types: list[str] | None, agents: list[str] | None) -> None:
    """Lift holds (all, or the named ones) and the release: workers relaunch."""
    _show_held(Client().resume(swarm, types=types, agents=agents))


def _show_held(r: dict) -> None:
    h = r["held"]
    held = [f"type {n}" for n in h["types"]] + [f"agent {n}" for n in h["agents"]]
    typer.echo(f"{r['swarm']}: held: {', '.join(held) or 'nothing'}")


RELEASE_POLL = 5.0  # s between worker checks while `stop --release` waits


@app.command()
def stop(
    swarm: str,
    type_: Annotated[list[str] | None, typer.Option(
        "--type", help="hold only this type (repeatable); with --final: its loop")] = None,
    agent: Annotated[list[str] | None, typer.Option(
        "--agent", help="hold only this agent (repeatable)")] = None,
    interrupt: bool = typer.Option(False, "--interrupt", help="also cut the running turns"),
    release: bool = typer.Option(
        False, "--release", help="drain every worker (each agent synced to the bucket), then "
                                 "free the compute; waits and reports agents not synced"),
    final: bool = typer.Option(
        False, "--final", help="stop the loops after their final round (default: every type "
                               "with a loop)"),
    reason: str = typer.Option("", "--reason", help="why (shown in status and events)"),
    delete: bool = typer.Option(
        False, "--delete", help="drain and DELETE the swarm from the controller (asks first)"),
    yes: bool = typer.Option(False, "--yes", help="with --delete: do not ask"),
) -> None:
    """Stop a swarm, keeping everything (docs/live.md "Lifecycle"). Default: hold, no new
    prompts or loop rounds, running turns finish, workers stay. `flock launch SWARM` resumes.
    A local run (`compute: local`) ends its background controller (and writes its bundle) on
    --delete only."""
    api = Client()
    if delete:
        if final or release or interrupt or type_ or agent:
            _fail(["--delete takes only --yes"])
        if not yes:
            typer.confirm(f"delete swarm {swarm} from the controller (agents drained, state "
                          "gone; the bucket keeps its copies)?", abort=True)
        if (run := local_run()) is not None and run.get("swarm") == swarm:
            _end_local(run)  # no DELETE first: the bundle is written from the rows it would drop
        else:
            typer.echo(json.dumps(api.delete(swarm)))
        return
    if final:
        if release or interrupt or agent:
            _fail(["--final takes only --type"])
        spec = api.swarm(swarm)["spec"]
        for t in type_ or [n for n, ts in spec["types"].items() if ts.get("loop")]:
            typer.echo(_loop_line(api.stop_loop(swarm, t)))
        return
    if release and (type_ or agent):
        _fail(["--release frees the whole swarm: no --type or --agent"])
    _show_held(api.stop(swarm, types=type_, agents=agent, interrupt=interrupt, reason=reason,
                        release=release))
    if not release:
        return
    typer.echo("releasing: running turns finish, every agent syncs, workers exit "
               "(Ctrl-C stops waiting, not the release)")
    while live := [w for w in api.workers(swarm) if w["job_id"]]:
        typer.echo(f"  {len(live)} workers still running")
        _sleep(RELEASE_POLL)
    unsynced = (api.swarm(swarm)["status"].get("released") or {}).get("unsynced")
    if unsynced:
        _fail([*(f"NOT synced: {n}: {e}" for n, e in sorted(unsynced.items())),
               (f"released, but {len(unsynced)} agents' final sync failed: their bucket copies "
               "are older than their last turns")])
    typer.echo(f"released: every agent synced, compute freed; `flock launch {swarm}` resumes")


def update_swarm(api: Client, spec_file: str, *, preflight: bool | None = None,
                 strict: bool = False, replicas: str = "keep", check: bool = False,
                 env_files=(), source: dict[str, str] | None = None) -> list[str]:
    """Re-read the experiment folder and apply it to the running swarm (docs/live.md): the
    spec, its prompt files and the tools folder, as `flock launch` reads them. Returns the
    lines to show (warnings, then what changed, or "no changes"); raises SpecError with the
    refusal when the folder cannot be applied, and then nothing was changed. Preflight runs
    by default only when the spec names a provider-backed model the running one does not.
    `replicas="keep"` keeps the live counts of existing types; `check` runs the controller's
    validation (dry run), stores nothing and names where each changed value came from.
    `env_files` as in `read_spec`; `source` replaces the `X-Flock-Source` header (--ref)."""
    from flock.live import diff_specs, keep_replicas, world_tool_changes
    from flock.tools.loader import ToolsError
    from flock.tools.spec import prepare

    sources: dict[str, str] = {}
    spec = read_spec(spec_file, env_files, sources=sources)
    refused, warnings = loop_problems(spec)
    if refused:
        raise SpecError(refused)
    warnings += extra_warnings(spec)
    try:
        bundle = prepare(spec, spec_file)
    except ToolsError as e:
        raise SpecError([f"{spec_file}: tools: {e}"]) from None
    try:
        old = SwarmSpec.model_validate(api.swarm(spec.swarm)["spec"])
    except ClientError as e:
        if e.status is None:
            raise
        raise SpecError([(f"no swarm {spec.swarm!r} is running on this controller: start it "
                          "with `flock launch`") if e.status == 404
                         else f"error {e.status}: {e.detail}"]) from None
    if preflight is None:  # --check probes every model; an update only the ones it adds
        preflight = bool(set(_unprobed(spec)) - set(_unprobed(old))) or (check and bool(_unprobed(spec)))
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
        lines = [f"{x} ({source_of(sources, x.split(':', 1)[0])})" if ": " in x
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


@app.command()
def update(
    target: str = typer.Argument(..., metavar="FOLDER_OR_SPEC"),
    preflight: bool = typer.Option(
        None, "--preflight/--skip-preflight",
        help="Probe provider-backed models first (default: only when the update adds one)."),
    strict: bool = typer.Option(False, "--strict", help="Unverified models count as failures."),
    replicas: str = typer.Option(
        "keep", "--replicas", help="`keep` (default): the running replica counts; `spec`: the file's."),
    check: bool = typer.Option(
        False, "--check", help="Run every check and print the diff; apply nothing (exit 1 if refused)."),
    env_files: list[str] | None = ENV_OPT,
    ref: str | None = REF_OPT,
) -> None:
    """Re-read an experiment folder (spec.yaml, prompts, tools/) and apply what changed to the
    running swarm (docs/live.md): new tools version, prompts, loop and comms settings, limits,
    replicas. Running agents take tools and prompts at their next turn boundary.
    Prints what changed; a refused change applies nothing."""
    path = Path(target)
    spec_file, ref_header = _at_ref(str(path / "spec.yaml" if path.is_dir() else path), ref)
    try:
        if replicas not in ("keep", "spec"):
            raise SpecError([f"--replicas {replicas!r}: use keep or spec"])
        lines = update_swarm(Client(), spec_file, preflight=preflight, strict=strict,
                             replicas=replicas, check=check, env_files=env_files or (),
                             source=ref_header)
    except SpecError as e:
        for line in e.lines:
            typer.echo(line, err=True)
        raise typer.Exit(1) from None
    for line in lines:
        typer.echo(line, err=line.startswith("warning:"))


def _tools_bundle(spec: SwarmSpec, spec_file: str):
    """The spec's tools folder packed and checked (None: the spec uses none); exit 1 on a bad
    module or an unknown tool name."""
    from flock.tools.loader import ToolsError
    from flock.tools.spec import prepare

    try:
        return prepare(spec, spec_file)
    except ToolsError as e:
        typer.echo(f"{spec_file}: tools: {e}", err=True)
        raise typer.Exit(1) from None


@app.command()
def status(
    swarm: str = typer.Argument(None),
    watch: Annotated[bool, typer.Option("--watch", help="Health view, refreshed every 10 s "
                                        "(docs/health.md).")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Status and newest health snapshot "
                                          "as JSON.")] = False,
) -> None:
    """Swarm status: counts by type and state. Given an agent name (or inside an agent), that
    agent's status as JSON."""
    api = Client()
    if not swarm and api.as_agent:
        swarm = os.environ.get("FLOCK_AGENT")
    if swarm and (api.as_agent or not api.is_swarm(swarm)):
        typer.echo(json.dumps(api.agent(swarm), indent=2))
        return
    if watch or as_json:
        _health_view(api, swarm, as_json=as_json, watch=watch)
        return
    if swarm:
        _print_status(api.swarm(swarm)["status"])
        return
    swarms = api.swarms()
    if not swarms:
        typer.echo("no swarms")
    for st in swarms:
        _print_status(st)
        typer.echo("")


def _health_view(api: Client, swarm: str | None, *, as_json: bool, watch: bool) -> None:
    while True:
        sts = [api.swarm(swarm)["status"]] if swarm else api.swarms()
        docs = [{"status": st, "health": api.health(st["swarm"])} for st in sts]
        if as_json:
            typer.echo(json.dumps(docs if not swarm else docs[0], indent=2))
        else:
            if watch:
                typer.echo("\x1b[H\x1b[J", nl=False)
            for d in docs:
                _print_health(d["status"], d["health"])
                typer.echo("")
        if not watch:
            return
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            return


def _num(v: Any, fmt: str = "{:.0f}") -> str:
    return "-" if v is None else fmt.format(v)


def _print_health(st: dict, h: dict | None) -> None:
    """`flock status --watch`: world, generation, alerts; workers by time to memory limit;
    restores; tool rates; provider queues; world metrics; stuck agents; recent events."""
    if h is None:
        _print_status(st)
        typer.echo("health: no snapshot yet")
        return
    b = h["body"]
    w = b.get("world")
    world = "no world" if not w else (
        f"world {'up' if w['up'] else 'DOWN'} {_num(w.get('manifest_ms'))} ms"
        + (f" slurm {w['slurm_state']}" if w.get("slurm_state") else ""))
    age = time.time() - h["ts"]
    typer.echo(f"{st['swarm']}  gen {h.get('generation')}  {world}  agents={st['agents_total']} "
               f"workers={st['workers_total']} cost=${st['cost_usd']:.2f}  "
               f"alerts {len(b['alerts'])}  (snapshot {age:.0f} s old)")
    for a in b["alerts"]:
        typer.echo(f"  ! {a['msg']}")
    for name, t in b["types"].items():
        gens = " ".join(f"{r[:8]}:{n}" for r, n in sorted(t["gens"].items(), key=lambda x: -x[1]))
        typer.echo(f"type {name}: {_states(t['by_state'])}  revs {gens or '-'}")
    workers = sorted(b["workers"].items(), key=lambda x: (x[1].get("eta_limit_s") is None,
                                                          x[1].get("eta_limit_s") or 0))
    if workers:
        typer.echo("workers (by time to memory limit):")
        _table([[wid, f"{_num(wk['mem'].get('current_mb'))}/{_num(wk['mem'].get('limit_mb'))}",
                 _num(wk.get("slope_mb_h")), _num(wk.get("eta_limit_s"), "{:.0f}s"),
                 wk.get("relaunches", 0)] for wid, wk in workers[:15]],
               ["worker", "mem MB", "MB/h", "to limit", "relaunches"])
    restores = [(wid, wk["restore"]) for wid, wk in workers if wk.get("restore")]
    if restores:
        typer.echo("restores:")
        _table([[wid, r.get("agents"), f"{r.get('files_done')}/{r.get('files_total')}",
                 _num(r.get("eta_s"), "{:.0f}s")] for wid, r in restores],
               ["worker", "agents", "files", "eta"])
    if b["tools"]:
        typer.echo("world tools:")
        _table([[n, _num(t["rate_per_min"], "{:.1f}"), _num(t["err_pct"], "{:.1f}"),
                 _num(t["refused_pct"], "{:.1f}"), _num(t["max_s"], "{:.1f}")]
                for n, t in sorted(b["tools"].items())],
               ["tool", "/min", "err %", "refused %", "max s"])
    m = (w or {}).get("metrics") or {}
    if m:
        bits = [f"loop lag {_num(m.get('loop_lag_s'), '{:.3f}')} s",
                f"429s {_num(m.get('overloaded_429_total'))}"]
        bits += [f"{p} queue {_num(q.get('queue_wait_oldest_s'), '{:.1f}')} s"
                 for p, q in (m.get("pools") or {}).items()]
        wm = {k: v for k, v in (m.get("world") or {}).items() if isinstance(v, (int, float))}
        if "lane_comment_outbox_depth" in wm:
            bits.append(f"comment outbox {wm.pop('lane_comment_outbox_depth'):g} (oldest "
                        f"{_num(wm.pop('lane_comment_outbox_oldest_age_s', None))} s)")
        lanes: dict[str, dict[str, float]] = {}
        for k in list(wm):
            if mt := re.match(r"lane_(.+)_(tripped|paused|effective_slots|in_flight)$", k):
                lanes.setdefault(mt[1], {})[mt[2]] = wm.pop(k)
        bits += [f"{k} {v:g}" for k, v in sorted(wm.items())]  # any world's own numbers
        typer.echo("world metrics: " + ", ".join(bits))
        for lane, lv in sorted(lanes.items()):
            st = "TRIPPED" if lv.get("tripped") else "paused" if lv.get("paused") else "ok"
            typer.echo(f"  lane {lane}: {st} slots {_num(lv.get('effective_slots'))} "
                       f"in flight {_num(lv.get('in_flight'))}")
    for p, q in b["providers"].items():
        typer.echo(f"provider {p}: waiting {_num(q.get('waiting'))} running "
                   f"{_num(q.get('running'))} kv {_num(q.get('kv'), '{:.0%}')}")
    stuck = sorted(((s, t) for t, ty in b["types"].items() for s in ty["stuck"]),
                   key=lambda x: -x[0]["silent_s"])[:10]
    if stuck:
        typer.echo("stuck or silent (top 10):")
        _table([[s["agent"], t, s["state"], f"{s['silent_s'] / 60:.0f} min"] for s, t in stuck],
               ["agent", "type", "state", "silent"])
    if h.get("recent_events"):
        typer.echo("control-plane events (last 10):")
        _table([[e["ts"][:19].replace("T", " "), e["type"],
                 e["data"].get("target") or e["data"].get("generation") or "",
                 str(e["data"].get("detail") or e["data"].get("error") or "")[:80]]
                for e in h["recent_events"]], ["time (UTC)", "event", "target", "detail"])


@app.command()
def agents(
    swarm: str,
    type: str = typer.Option(None, "--type"),
    state: str = typer.Option(None, "--state"),
    parent: str = typer.Option(None, "--parent", help="only the children of this agent"),
) -> None:
    """List a swarm's agents."""
    rows = [[a["name"], a["state"], a.get("provider") or "-", a["worker"], a["turns"],
             a["pending"], f"${a['cost_usd']:.4f}",  # a short turn costs a fraction of a cent
             a["last_heartbeat"]]
            for a in Client().agents(swarm, type=type, state=state, parent=parent)]
    _table(rows, ["name", "state", "provider", "worker", "turns", "pending", "cost",
                  "last_heartbeat"])


@app.command()
def agent(name: str) -> None:
    """One agent's status as JSON."""
    typer.echo(json.dumps(Client().agent(name), indent=2))


@app.command()
def prompt(
    args: Annotated[list[str], typer.Argument(metavar="[NAME] TEXT")],
    timeout: str = typer.Option(None, "--timeout"),
    reply_to: str = typer.Option(None, "--reply-to"),
    swarm: str = typer.Option(None, "--swarm", help="with --type: the swarm"),
    type: str = typer.Option(None, "--type", help="prompt every agent of this type (one request)"),
    state: str = typer.Option(None, "--state", help="with --type: only agents in this state, e.g. idle"),
    skip_pending: bool = typer.Option(
        None, "--skip-pending/--stack",
        help="with --type/--children: skip agents that already have an undelivered prompt "
             "(default: on with --state idle, off otherwise)"),
    children: bool = typer.Option(
        False, "--children", help="prompt the direct children of --parent (one request)"),
    parent: str = typer.Option(None, "--parent", help="with --children; default: $FLOCK_AGENT"),
) -> None:
    """Send a prompt to an agent (`prompt NAME TEXT`), to every agent of a type
    (`prompt --swarm S --type T [--state idle] TEXT`) or to your children
    (`prompt --children [--type T] [--state idle] TEXT`); prints the ack (counts for the bulk
    forms). With `--state idle` an agent that already has a pending prompt is skipped, so
    repeating the call never stacks prompts (`skipped_pending` in the counts)."""
    api, opts = Client(), {"timeout": timeout, "reply_to": reply_to}
    bulk = {"state": state, "skip_pending": skip_pending, **opts}
    if children:
        parent = parent or os.environ.get("FLOCK_AGENT")
        if not parent or swarm is not None or len(args) != 1:
            typer.echo("usage: flock prompt --children [--parent P] [--type T] [--state STATE] "
                       "TEXT (parent defaults to $FLOCK_AGENT)", err=True)
            raise typer.Exit(2)
        typer.echo(json.dumps(api.prompt_children(parent, args[0], type=type, **bulk)))
        return
    if type is not None:
        if swarm is None or len(args) != 1:
            typer.echo("usage: flock prompt --swarm S --type T [--state STATE] TEXT", err=True)
            raise typer.Exit(2)
        typer.echo(json.dumps(api.prompt_type(swarm, type, args[0], **bulk)))
        return
    if swarm is not None or state is not None or len(args) != 2:
        typer.echo("usage: flock prompt NAME TEXT (or --swarm S --type T TEXT)", err=True)
        raise typer.Exit(2)
    typer.echo(json.dumps(api.prompt(args[0], args[1], **opts)))


def _short(text: Any, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _blocks_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "toolCall":
            parts.append(f"[{b.get('name')} {_short(json.dumps(b.get('arguments')), 80)}]")
    return " ".join(parts)


def pretty_event(ev: dict) -> str:
    """One line per event: time, turn, kind, then the text or a summary (`events --pretty`)."""
    ts = str(ev.get("ts", ""))[11:19]
    turn = ev.get("turn_id") or "-"
    data = ev.get("data") or {}
    kind = ev.get("type", "?")
    if kind == "message":
        m = data.get("message") or {}
        role = m.get("role", "?")
        if role == "toolResult":
            return f"{ts} {turn:>5} tool     {m.get('toolName', '')}: {_short(_blocks_text(m.get('content')), 120)}"
        return f"{ts} {turn:>5} {role:<8} {_short(_blocks_text(m.get('content')))}"
    detail = data.get("data", data) if isinstance(data.get("data"), dict) else data
    if kind == "turn_ended":
        toks = detail.get("tokens") or {}
        used = " ".join(f"{model} in={u.get('input', 0)} out={u.get('output', 0)}"
                        for model, u in toks.items())
        return f"{ts} {turn:>5} {kind:<8} {detail.get('status')} {used}".rstrip()
    keep = {k: v for k, v in detail.items() if v is not None}
    summary = " ".join(f"{k}={_short(v, 60)}" for k, v in keep.items())
    return f"{ts} {turn:>5} {kind:<8} {summary}".rstrip()


@app.command()
def events(
    name: str,
    after: int = typer.Option(0, "--after"),
    follow: bool = typer.Option(False, "--follow", "-f"),
    wait: bool = typer.Option(False, "--wait", help="block until an event after --after"),
    timeout: float = typer.Option(600, "--timeout", help="seconds --wait blocks at most"),
    pretty: bool = typer.Option(False, "--pretty", help="one readable line per event"),
) -> None:
    """Print an agent's event stream (JSONL); --follow keeps long-polling; --wait returns once
    something new arrived (the last line on stderr is the cursor for the next --after);
    --pretty prints time, turn, kind and the text or a summary instead of JSON. Given a swarm,
    its swarm-level stream (loop_started, loop_tick, loop_stopped, final_round, budget_stop)."""
    for page in Client().events(name, after=after, follow=follow or wait,
                                timeout=timeout if wait else None):
        for ev in page["events"]:
            typer.echo(pretty_event(ev) if pretty else json.dumps(ev))
        after = page["next"]
        if wait and page["events"]:
            break
    if wait:
        typer.echo(f"next: {after}", err=True)


@app.command()
def workers(swarm: str) -> None:
    """List a swarm's workers."""
    found = Client().workers(swarm)

    rows = [
        [
            w["id"],
            w["type"],
            w["state"],
            w["job_id"],
            len(w["agents"]),
            w["crashes"],
            w["deadline"],
            w["last_heartbeat"],
            w.get("knobs_rev") or "",
            w.get("last_error") or "",
        ]
        for w in found
    ]
    _table(rows, ["id", "type", "state", "job", "agents", "crashes", "deadline", "last_heartbeat",
                  "knobs", "last_error"])
    if any(str(w["job_id"] or "").startswith("local-") for w in found):  # compute local
        typer.echo("full worker logs: <run dir>/logs/<worker id>.log", err=True)
    elif any(str(w["job_id"] or "").isdigit() for w in found):  # compute slurm
        typer.echo("full job logs: <compute log_dir>/<worker id>-<job>.out", err=True)
    else:
        typer.echo("full job logs: hf jobs logs <job>", err=True)


agent_app = typer.Typer(cls=_Group, help="Inside an agent (FLOCK_AGENT_TOKEN): spawn and remove "
                        "child agents (docs/spawn.md).", no_args_is_help=True)


@agent_app.command()
def spawn(
    type: str,
    prompt: str = typer.Option(None, "--prompt", help="first prompt for the child"),
    key: str = typer.Option(None, "--key", help="idempotency key: a repeat returns the same child"),
    parent: str = typer.Option(None, "--parent", help="default: $FLOCK_AGENT (yourself)"),
    as_json: bool = typer.Option(False, "--json", help="print the whole ack"),
) -> None:
    """Spawn a child agent of TYPE; prints its name."""
    parent = parent or os.environ.get("FLOCK_AGENT")
    if not parent:
        typer.echo("no parent: pass --parent or set FLOCK_AGENT", err=True)
        raise typer.Exit(1)
    ack = Client().spawn(parent, type, prompt=prompt, key=key)
    typer.echo(json.dumps(ack) if as_json else ack["name"])


@agent_app.command()
def remove(name: str) -> None:
    """Remove a spawned agent and everything it spawned."""
    typer.echo(json.dumps(Client().remove(name)))


# Hidden aliases: agents of runs started before flock-agent and stored specs still call
# `flock spawn|remove` and `{python} -m flock.cli world serve`.
app.command("spawn", hidden=True)(spawn)
app.command("remove", hidden=True)(remove)
from flock.world.__main__ import world_app

app.add_typer(world_app, name="world", hidden=True)

if __name__ == "__main__":
    app()
