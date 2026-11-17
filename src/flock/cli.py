"""`flock` CLI: argument parsing, prompts and output over `flock.ops` and
`flock.client.Client` (connection config: see flock/client.py)."""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from typer.core import TyperGroup

from flock import ops, rundir
from flock import status as flock_status
from flock.client import Client, ClientError
from flock.models import SwarmSpec
from flock.ops import SpecError, check_spec, read_spec, source_header
from flock.specrender import at_ref
from flock.world.__main__ import world_app


class _Group(TyperGroup):
    """Every command: a refused or failed controller call prints its line and exits 1."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except ClientError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1) from None


app = typer.Typer(cls=_Group, help="Host coding-harness agent swarms on HF Jobs.", rich_markup_mode=None,
                  no_args_is_help=True)
_API: list[Client] = []  # set by `_main`: --run / FLOCK_RUN_DIR (`Client.for_run`)


def _api() -> Client:
    return _API[0] if _API else Client()


@app.callback()
def _main(ctx: typer.Context, run: str = typer.Option(
        None, "--run", metavar="RUN_DIR",
        help="Run every command against RUN_DIR (`flock init`): its cluster.env and run.env as "
             "the first --env files, its controller (hosts/controller.host) and token. Also FLOCK_RUN_DIR, which "
             "an exported FLOCK_URL beats; --run beats FLOCK_URL.")) -> None:
    _API[:] = [] if ctx.invoked_subcommand in ("init", "world") else [
        Client.for_run(run, need_host=ctx.invoked_subcommand != "launch")]


def _table(rows: list[list[Any]], header: list[str]) -> None:
    cells = [header] + [[("" if v is None else str(v)) for v in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    for r in cells:
        typer.echo("  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)).rstrip())


def _states(d: dict[str, int]) -> str:
    return " ".join(f"{k}:{v}" for k, v in sorted(d.items())) or "-"


def _usd(v: float) -> str:
    return f"${v:.4f}" if v < 0.01 else f"${v:.2f}"  # a short turn costs a fraction of a cent


def _print_status(st: dict) -> None:
    typer.echo(
        f"{st['swarm']}  agents={st['agents_total']} workers={st['workers_total']} "
        f"pending={st['pending_prompts']} cost={_usd(st['cost_usd'])}"
        + (f" gen {st['generation']}" if st.get("generation") else "")
    )
    rows = [
        [
            t["name"],
            t["health"],
            t["replicas"],
            _states(t["agents_by_state"]),
            _states(t["workers_by_state"]),
            _usd(t["cost_usd"]),
        ]
        for t in st["types"].values()
    ]
    _table(rows, ["type", "health", "replicas", "agents", "workers", "cost"])
    if success := st.get("success"):
        typer.echo(f"STOPPED ON SUCCESS: {success['agent']} at {success['accepted_at']}; "
                   f"verification: {success.get('verification', 'pending')}")
    held = st.get("held") or {}
    for kind in ("types", "agents"):
        for name, h in (held.get(kind) or {}).items():
            typer.echo(f"HELD {kind[:-1]} {name}: {h['reason'] or '-'} (by {h['by']}"
                       f"{', auto' if h.get('auto') else ''}, since {h['since'][:19]}Z)")
    if rel := st.get("released"):
        next_run = "; launch a new run to repeat" if st.get("success") else " until `flock launch`"
        typer.echo(f"RELEASED since {rel['since'][:19]}Z: workers drained, nothing launches"
                   f"{next_run}")
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
        models = "".join(f"{m} context {w or '?'}, " for m, w in (pv.get("models") or {}).items())
        typer.echo(f"{pv['name']} ({pv['endpoint']}): {models}working {pv['working']}{cap}, "
                   f"types {','.join(pv['types']) or '-'}")
        if pv.get("zero_usage_agents"):
            typer.echo(f"warning: {pv['name']}: {pv['zero_usage_agents']} agents finished "
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


def load_spec(spec_file: str, env_files=(), sources: dict[str, str] | None = None) -> SwarmSpec:
    """`read_spec`, exiting 1 with the reasons on stderr."""
    try:
        return read_spec(spec_file, env_files, sources)
    except SpecError as e:
        _fail(e.lines)


def _warn_unverified(models: list[str], why: str) -> None:
    for m in models:
        typer.echo(f"warning: {m}: unverified ({why})", err=True)


def _checked(spec: SwarmSpec, spec_file: str):
    """`check_spec`: its warnings on stderr; exit 1 with the refusal. Returns its result."""
    try:
        checked = check_spec(spec, spec_file)
    except SpecError as e:
        _fail([*e.done.get("warnings", []), *e.lines])
    for line in checked[1]:
        typer.echo(line, err=True)
    return checked


def _print_preflight(rep: dict, strict: bool) -> None:
    """One row per model; never silent about a model that was not probed."""
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
def init(
    run_dir: str = typer.Argument(..., metavar="RUN_DIR"),
    from_: str = typer.Option(None, "--from", metavar="EXPERIMENT_FOLDER",
                              help="The experiment folder to copy (needed while RUN_DIR has none)."),
    secrets: str = typer.Option(None, "--secrets", metavar="FILE",
                                help="NAME=value file (mode 600) filling run.env's missing keys."),
    profile: str = typer.Option(None, "--profile", metavar="FILE",
                                help="Cluster profile (env file) to copy to RUN_DIR/cluster.env, "
                                     "read before run.env (docs/slurm.md)."),
) -> None:
    """Prepare RUN_DIR for `flock launch` (docs/slurm.md "Deploying"): create what is missing
    (logs/, hosts/, a copy of the experiment folder, cluster.env, run.env with per-run tokens), then check
    it without a controller. Never overwrites a run.env key or an edited copy. Run it again as
    the doctor; exit 1 if a check fails."""
    try:
        rep = rundir.init(run_dir, from_, secrets_file=secrets, profile=profile)
    except ValueError as e:
        _fail([str(e)])
    for what in rep.created:
        typer.echo(f"created {what}")
    for note in rep.notes:
        typer.echo(f"note: {note}")
    for c in rep.checks:
        word = "warn" if c.warn else "ok" if c.ok else "skip" if c.ok is None else "FAIL"
        typer.echo(f"{word:<5}{c.name}: {c.detail}")
        for line in c.lines:
            typer.echo(f"       {line}")
    if rep.failed:
        raise typer.Exit(1)


@app.command()
def launch(
    target: str = typer.Argument(..., metavar="FOLDER_OR_SPEC_OR_SWARM"),
    type_: Annotated[list[str] | None, typer.Option(
        "--type", help="resume: lift only this type's hold (repeatable)")] = None,
    agent: Annotated[list[str] | None, typer.Option(
        "--agent", help="resume: lift only this agent's hold (repeatable)")] = None,
    preflight: bool = typer.Option(
        None, "--preflight/--skip-preflight",
        help="Probe provider-backed models before creating (default: on when the spec has any)."),
    strict: bool = typer.Option(
        False, "--strict", help="Treat models that were not probed (unverified) as failures."),
    env_files: list[str] | None = ENV_OPT,
    ref: str | None = REF_OPT,
    hold: bool = typer.Option(
        False, "--hold", help="Slurm, no controller up: start the run's controller and world "
                              "with every swarm held and released; `flock update`, then "
                              "`flock launch` resumes it"),
) -> None:
    """Start a swarm from an experiment folder (or spec file), or resume a stopped one
    (docs/live.md "Lifecycle"). When no controller answers, first starts the run's controller
    (and world): a local run, Slurm jobs (`flock init` run dir) or an HF Job (FLOCK_WHEEL_URL),
    and waits for it; it restores its checkpoint. Missing: creates it (refused when its bucket
    already holds agents of that name). Present: clears `flock stop`'s holds and release;
    refused when the folder's spec differs from the stored one (replicas aside): run
    `flock update` first. Given a swarm name instead of a folder: resumes it on the running
    controller."""
    api = _api()
    if ref is None and ops.is_swarm_name(target):
        if hold:
            _fail(["--hold takes the experiment folder (it starts the run's controller)"])
        _show_held(api.resume(target, types=type_, agents=agent))
        return
    spec_file, ref_header = _at_ref(_spec_file(target), ref)
    source = ref_header or source_header(spec_file)  # before any write lands in the folder
    env_files = [*api.env_files, *(env_files or ())]
    spec = load_spec(spec_file, env_files)
    checked = _checked(spec, spec_file)
    try:
        done = ops.launch(api, spec, spec_file, env_files=env_files, source=source, types=type_,
                          agents=agent, preflight=preflight, strict=strict, checked=checked,
                          say=lambda line: typer.echo(f"  {line}"), hold=hold)
    except SpecError as e:
        _show_launch(spec.swarm, e.done, strict)
        if run := e.done.get("local"):
            typer.echo(f"local run ended; bundle: {run['run_dir']}/run.json")
        _fail(e.lines)
    _show_launch(spec.swarm, done, strict)


def _show_launch(swarm: str, done: dict, strict: bool) -> None:
    """What `ops.launch` did, in order (its warnings were shown by `_checked`)."""
    if run := done.get("local"):
        typer.echo(f"local run {run['run_dir']} at {run['url']} (log: {ops.LOCAL_DIR}/launch.log)")
    if started := done.get("up"):
        typer.echo(f"up: {', '.join(started['jobs'].values())}; controller at {started['url']}")
        if "FLOCK_RUN_DIR" not in _api().env:  # no run dir records it
            typer.echo(f"export FLOCK_URL={started['url']}")
    if "resumed" in done:
        _show_held(done["resumed"])
    if "held" in done:  # --hold
        if done["held"] is None:
            typer.echo(f"{swarm}: not on the controller (no checkpoint of it was restored)")
        else:
            _show_held(done["held"])
            typer.echo(f"released; `flock update` the spec, then `flock launch` resumes {swarm}")
    if "preflight" in done:
        _print_preflight(done["preflight"], strict)
    _warn_unverified(done.get("unverified", []), "preflight skipped")
    if up := done.get("tools"):
        names = ", ".join(up["tools"]) or "none"
        typer.echo(f"tools: version {up['version']} ({up['bytes']} bytes): {names}")
    if "status" in done:
        _print_status(done["status"])


def _show_held(r: dict) -> None:
    h = r["held"]
    held = [f"type {n}" for n in h["types"]] + [f"agent {n}" for n in h["agents"]]
    typer.echo(f"{r['swarm']}: held: {', '.join(held) or 'nothing'}")


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
                                 "checkpoint and end the run's controller and world jobs (a "
                                 "local run keeps its controller); reports agents not synced"),
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
    --delete only; --release ends a run's Slurm or HF Jobs controller and world after a
    checkpoint (`flock launch FOLDER` starts them again)."""
    try:
        done = ops.stop(_api(), swarm, types=type_, agents=agent, interrupt=interrupt,
                        reason=reason, release=release, final=final, delete=delete,
                        confirm=None if yes else lambda: typer.confirm(
                            f"delete swarm {swarm} from the controller (agents drained, state "
                            "gone; the bucket keeps its copies)?", abort=True),
                        progress=lambda n: typer.echo(
                            f"  releasing: {n} workers still running (every agent syncs; "
                            "Ctrl-C stops waiting, not the release)"))
    except SpecError as e:
        for name, err in sorted((e.done.get("unsynced") or {}).items()):
            typer.echo(f"NOT synced: {name}: {err}", err=True)
        _fail(e.lines)
    if run := done.get("ended"):
        typer.echo(f"local run ended; bundle: {run['run_dir']}/run.json")
    if "deleted" in done:
        typer.echo(json.dumps(done["deleted"]))
    for lp in done.get("loops", []):
        typer.echo(_loop_line(lp))
    if "held" in done:
        _show_held(done["held"])
    down = done.get("down") or {}
    if "kept" in down:
        typer.echo(f"controller kept: {down['kept']}")
    for what in down.get("ended", []):
        typer.echo(f"ended {what}")
    if unsynced := done.get("unsynced"):
        _fail([*(f"NOT synced: {n}: {e}" for n, e in sorted(unsynced.items())),
               (f"released, but {len(unsynced)} agents' final sync failed: their bucket copies "
               "are older than their last turns")])
    if down.get("ended"):
        typer.echo("released: every agent synced, controller checkpointed, nothing runs; "
                   + ("this swarm succeeded; launch a new run" if done.get("success") else
                      "`flock launch FOLDER` starts it again and resumes"))
    elif "unsynced" in done:
        typer.echo("released: every agent synced, compute freed; "
                   + ("this swarm succeeded; launch a new run" if done.get("success") else
                      f"`flock launch {swarm}` resumes"))


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
    spec_file, ref_header = _at_ref(_spec_file(target), ref)
    sources: dict[str, str] = {}
    spec = load_spec(spec_file, [*_api().env_files, *(env_files or ())], sources)
    try:
        done = ops.update(_api(), spec, spec_file, sources=sources, preflight=preflight,
                          strict=strict, replicas=replicas, check=check, source=ref_header)
    except SpecError as e:
        _fail(e.lines)
    for line in update_lines(done, check):
        typer.echo(line, err=line.startswith("warning:"))


def update_lines(done: dict, check: bool = False) -> list[str]:
    """`ops.update`'s result as `flock update` prints it: warnings, then what changed (or
    "no changes")."""
    if not done["diff"] and not check:
        return [*done["warnings"], *done["kept"], "no changes"]
    wt = done["world_tools"]
    return [*done["warnings"], *done["probed"], *done["kept"], *done["diff"], *done["deletes"],
            *([wt] if wt else []), *(["check ok: nothing was applied"] if check else [])]


@app.command()
def status(
    swarm: str = typer.Argument(None, help="A swarm (default: all), or an agent's name."),
    agent: str = typer.Option(None, "--agent", help="That agent's status (JSON)."),
    type_: str = typer.Option(None, "--type", help="List that type's agents, with its "
                              "per-agent loop rows."),
    state: str = typer.Option(None, "--state", help="List the agents in this state (idle, "
                              "working, stuck, ...; idle as `prompt --state idle` means it)."),
    parent: str = typer.Option(None, "--parent", help="List the children of this agent."),
    events: bool = typer.Option(False, "--events", help="Print the event stream of SWARM (or "
                                "--agent) instead, one line per event (docs/health.md)."),
    only: str = typer.Option("", "--only", metavar="PREFIX",
                             help="With --events: only names starting with PREFIX, e.g. agent."),
    after: int = typer.Option(0, "--after", help="With --events: start after this cursor."),
    follow: bool = typer.Option(False, "--follow", "-f", help="With --events: keep printing."),
    wait: bool = typer.Option(False, "--wait", help="With --events: return once something new "
                              "arrived (600 s at most); the last line on stderr is the next "
                              "--after."),
    watch: bool = typer.Option(False, "--watch", help="Reprint every 10 s."),
    as_json: bool = typer.Option(False, "--json", help="Everything, as JSON."),
) -> None:
    """The operator view of a swarm (docs/health.md): a verdict line first (ok, degraded,
    failing and why; exit 1 on failing), then counts, holds, world, model servers, workers,
    loops, ramps and alerts. A filter lists agents; --events prints the event stream. Inside
    an agent: your own status, or with --parent your children."""
    api = _api()
    if events:
        _print_events(api, agent or swarm or os.environ.get("FLOCK_AGENT", ""), only, after,
                      follow, wait, as_json)
        return
    if api.as_agent:
        me = agent or swarm or os.environ.get("FLOCK_AGENT")
        if not (type_ or state or parent):
            typer.echo(json.dumps(api.agent(me), indent=2))
            return
        sw = swarm or api.agent(os.environ.get("FLOCK_AGENT", ""))["swarm"]
        _print_agents(api.agents(sw, type=type_, state=state, parent=parent))
        return
    if agent or (swarm and not api.is_swarm(swarm)):
        typer.echo(json.dumps(api.agent(agent or swarm), indent=2))
        return
    while True:
        names = [swarm] if swarm else [s["swarm"] for s in api.swarms()]
        views = [flock_status.view(api, n, type=type_, state=state, parent=parent) for n in names]
        if as_json:
            typer.echo(json.dumps(views[0] if swarm else views, indent=2))
        else:
            if watch:
                typer.echo("\x1b[H\x1b[J", nl=False)
            for v in views:
                _print_view(v)
                typer.echo("")
            if not views:
                typer.echo("no swarms")
        if not watch:
            break
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            return
    if any(v["verdict"]["level"] == "failing" for v in views):
        raise typer.Exit(1)


def _print_events(api: Client, name: str, only: str, after: int, follow: bool, wait: bool,
                  as_json: bool, timeout: float = 600, raw: bool = False) -> None:
    if not name:
        _fail(["--events needs a swarm or --agent"])
    for page in flock_status.events(api, name, after=after, follow=follow or wait, prefix=only,
                                    timeout=timeout if wait else None, raw=raw):
        for ev in page["events"]:
            typer.echo(json.dumps(ev) if as_json else pretty_event(ev))
        after = page["next"]
        if wait and page["events"]:
            break
    if wait:
        typer.echo(f"next: {after}", err=True)


def _print_agents(rows: list[dict]) -> None:
    """Turns and cost as of each agent's last heartbeat: its age next to them."""
    _table([[a["name"], a["state"], a["type"], a["worker"], a["turns"], a["pending"],
             _usd(a["cost_usd"]), _ago(a["last_heartbeat"]),
             (a.get("endpoint") or "-") + (" DEAD" if a.get("on_dead_server") else "")]
            for a in rows],
           ["name", "state", "type", "worker", "turns", "pending", "cost", "heartbeat",
            "server"])


def _ago(iso: str | None) -> str:
    return "-" if not iso else f"{time.time() - datetime.fromisoformat(iso).timestamp():.0f} s ago"


def _num(v: Any, fmt: str = "{:.0f}") -> str:
    return "-" if v is None else fmt.format(v)


def _hhmm(epoch: float | None) -> str:
    return "-" if epoch is None else time.strftime("%H:%M", time.gmtime(epoch)) + "Z"


def _print_view(v: dict) -> None:
    """`flock status SWARM`: `flock_status.view` as text, the verdict first."""
    vd, st, h = v["verdict"], v["status"], v["health"]
    typer.echo(vd["level"].upper() + (": " + "; ".join(vd["reasons"]) if vd["reasons"] else ""))
    _print_status(st)
    _print_health(h, v["workers"])
    if "agents" in v:
        _print_agents(v["agents"])
    if per := (v.get("loop") or {}).get("per_agent"):
        _table([[n, r["phase"], r["rounds"], r.get("done_reason") or "", r["final"] or "",
                 r["errors"], r["wakes"]] for n, r in per.items()],
               ["agent", "phase", "rounds", "done", "final", "errors", "wakes"])


def _print_health(h: dict | None, workers: list[dict]) -> None:
    """The snapshot's sections: world, model servers, workers (pending reasons, memory,
    restores, problems), tool rates, world metrics, ramps, alerts, stuck agents, recent
    control-plane events."""
    pend = Counter(w["pending_reason"] for w in workers if w.get("pending_reason"))
    if workers:
        typer.echo("workers: " + _states(Counter(w["state"] for w in workers))
                   + "".join(f"; {n} {why}" for why, n in sorted(pend.items())))
    bad = [w for w in workers if w["crashes"] or w.get("last_error")]
    if bad:
        _table([[w["id"], w["state"], w["job_id"], w["crashes"], (w.get("last_error") or "")[:100]]
                for w in bad], ["worker", "state", "job", "crashes", "last_error"])
        typer.echo("  full logs: " + (
            "<run dir>/logs/<worker id>.log" if any(str(w["job_id"] or "").startswith("local-")
                                                   for w in bad)
            else "<compute log_dir>/<worker id>-<job>.out" if any(
                str(w["job_id"] or "").isdigit() for w in bad) else "hf jobs logs <job>"))
    if h is None:
        typer.echo("health: no snapshot yet")
        return
    b = h["body"]
    w = b.get("world")
    age = time.time() - h["ts"]
    typer.echo(f"snapshot {age:.0f} s old, generation {h.get('generation')}; "
               + ("no world" if not w else f"world {'up' if w['up'] else 'DOWN'} "
                  f"{_num(w.get('manifest_ms'))} ms"
                  + (f" slurm {w['slurm_state']}" if w.get("slurm_state") else "")))
    for p, servers in flock_status.servers(h).items():
        for host, s in sorted(servers.items()):
            typer.echo(f"vllm {p} {host}: " + (
                f"waiting {_num(s.get('waiting'))} running {_num(s.get('running'))} "
                f"kv {_num(s.get('kv'), '{:.0%}')}" if s["up"]
                else f"{'DOWN' if s.get('dead') else 'no answer'} x{s.get('fails', 1)} "
                     f"({s.get('error')})") + f", {s.get('agents', 0)} agents")
    for t, r in (b.get("ramps") or {}).items():
        nxt = f", next +{r['next_add']} at {_hhmm(r['next_at'])}" if r["next_add"] else ""
        typer.echo(f"ramp {t}: step {r['step']}/{r['steps']} ({r['replicas']} of max "
                   f"{r['max']}){nxt}" + (f", held {r['holds']}x" if r["holds"] else ""))
    for a in b["alerts"]:
        typer.echo(f"  ! {a['msg']}")
    if h.get("before_restart") and b["types"]:
        typer.echo("type counts: not shown, the snapshot predates the controller's restart")
    for name, t in ({} if h.get("before_restart") else b["types"]).items():
        gens = " ".join(f"{r[:8]}:{n}" for r, n in sorted(t["gens"].items(), key=lambda x: -x[1]))
        typer.echo(f"type {name}: {_states(t['by_state'])}  revs {gens or '-'}  "
                   f"(health {age:.0f} s ago)")
    ws = sorted(((wid, wk) for wid, wk in b["workers"].items() if isinstance(wk, dict)),
                key=lambda x: (x[1].get("eta_limit_s") is None, x[1].get("eta_limit_s") or 0))
    if ws:
        typer.echo("workers (by time to memory limit):")
        _table([[wid, f"{_num(wk['mem'].get('current_mb'))}/{_num(wk['mem'].get('limit_mb'))}",
                 _num(wk.get("slope_mb_h")), _num(wk.get("eta_limit_s"), "{:.0f}s"),
                 wk.get("relaunches", 0)] for wid, wk in ws[:15]],
               ["worker", "mem MB", "MB/h", "to limit", "relaunches"])
    restores = [(wid, wk["restore"]) for wid, wk in ws if wk.get("restore")]
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
    stuck = sorted(((s, t) for t, ty in b["types"].items() for s in ty["stuck"]),
                   key=lambda x: -x[0]["silent_s"])[:10]
    if stuck:
        typer.echo("stuck or silent (top 10):")
        _table([[s["agent"], t, s["state"], f"{s['silent_s'] / 60:.0f} min"] for s, t in stuck],
               ["agent", "type", "state", "silent"])
    if h.get("recent_events"):
        typer.echo("control-plane events (last 10):")
        _table([[e["ts"][:19].replace("T", " "), flock_status.event_name(e["type"], "swarm"),
                 e["data"].get("target") or e["data"].get("generation") or "",
                 str(e["data"].get("detail") or e["data"].get("error") or "")[:80]]
                for e in h["recent_events"]], ["time (UTC)", "event", "target", "detail"])


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
    api, opts = _api(), {"timeout": timeout, "reply_to": reply_to}
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
    """One line per normalised event (`flock.status.normalise`): time, turn, name, then the
    text or a summary (`status --events`)."""
    ts, name, d = str(ev.get("ts", ""))[11:19], ev["name"], ev["data"]
    turn = d.get("turn_id") or "-"
    if name == "agent.message":
        m = d.get("message") or {}
        role = m.get("role", "?")
        if role == "toolResult":
            return f"{ts} {turn:>5} tool     {m.get('toolName', '')}: {_short(_blocks_text(m.get('content')), 120)}"
        return f"{ts} {turn:>5} {role:<8} {_short(_blocks_text(m.get('content')))}"
    if name == "agent.turn_ended":
        used = " ".join(f"{model} in={u.get('input', 0)} out={u.get('output', 0)}"
                        for model, u in (d.get("tokens") or {}).items())
        return f"{ts} {turn:>5} {name} {d.get('status')} {used}".rstrip()
    who = "" if name.startswith("agent.") else ev.get("agent") or ev.get("worker") or ""
    summary = " ".join(f"{k}={_short(v, 60)}" for k, v in d.items()
                       if v is not None and k != "turn_id")
    return " ".join(x for x in (ts, f"{turn:>5}", name, who, summary) if x)


agent_app = typer.Typer(cls=_Group, help="Spawn and remove child agents of an agent "
                        "(docs/spawn.md).", no_args_is_help=True)


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
    ack = _api().spawn(parent, type, prompt=prompt, key=key)
    typer.echo(json.dumps(ack) if as_json else ack["name"])


@agent_app.command()
def remove(name: str) -> None:
    """Remove a spawned agent and everything it spawned."""
    typer.echo(json.dumps(_api().remove(name)))


# Hidden: `flock world serve` runs a world (python -m flock.world serve).
app.add_typer(world_app, name="world", hidden=True)


if __name__ == "__main__":
    app()
