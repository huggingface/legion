"""`flock` CLI: argument parsing, spec and file loading, local-run processes and output over
`flock.client.Client` (connection config: see flock/client.py)."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from typer.core import TyperGroup

from flock import ops
from flock.client import Client, ClientError, local_run
from flock.models import SwarmSpec
from flock.ops import SpecError, check_spec, read_spec, source_header
from flock.specrender import at_ref


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


def load_spec(spec_file: str, env_files=()) -> SwarmSpec:
    """`read_spec`, exiting 1 with the reasons on stderr."""
    try:
        return read_spec(spec_file, env_files)
    except SpecError as e:
        for line in e.lines:
            typer.echo(line, err=True)
        raise typer.Exit(1) from None


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
    checked = _checked(spec, spec_file)
    started = _start_local(spec, spec_file, env_files or ()) if spec.compute.kind == "local" \
        else None
    try:
        try:
            done = ops.launch(Client(), spec, spec_file, source=source, types=type_, agents=agent,
                              adopt=adopt, checkpoint=checkpoint, no_checkpoint=no_checkpoint,
                              preflight=preflight, strict=strict, checked=checked)
        except SpecError as e:
            _show_launch(spec.swarm, e.done, strict)
            _fail(e.lines)
        _show_launch(spec.swarm, done, strict)
    except BaseException:  # also Ctrl-C during the preflight
        if started is not None:  # nothing created: do not leave a controller behind
            _end_local(started)
        raise


def _show_launch(swarm: str, done: dict, strict: bool) -> None:
    """What `ops.launch` did, in order (its warnings were shown by `_checked`)."""
    if a := done.get("adopted"):
        typer.echo(f"adopted {swarm}: {a['agents']} agents from {a['checkpoint']}")
    if why := done.get("adopt_skipped"):
        typer.echo(f"{why}: creating from the spec (agents restore from the bucket)")
    if "resumed" in done:
        _show_held(done["resumed"])
    if "preflight" in done:
        _print_preflight(done["preflight"], strict)
    _warn_unverified(done.get("unverified", []), "preflight skipped")
    if up := done.get("tools"):
        names = ", ".join(up["tools"]) or "none"
        typer.echo(f"tools: version {up['version']} ({up['bytes']} bytes): {names}")
    if "status" in done:
        _print_status(done["status"])


def _start_local(spec: SwarmSpec, spec_file: str, env_files) -> dict | None:
    """`ops.start_local`, saying where the run is; exit 1 when it cannot start."""
    try:
        run = ops.start_local(spec, spec_file, env_files)
    except SpecError as e:
        _fail(e.lines)
    if run is not None:
        typer.echo(f"local run {run['run_dir']} at {run['url']} (log: {ops.LOCAL_DIR}/launch.log)")
    return run


def _end_local(run: dict) -> None:
    ops.end_local(run)
    typer.echo(f"local run ended; bundle: {run['run_dir']}/run.json")


def _resume(swarm: str, types: list[str] | None, agents: list[str] | None) -> None:
    """Lift holds (all, or the named ones) and the release: workers relaunch."""
    _show_held(Client().resume(swarm, types=types, agents=agents))


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
    unsynced = ops.release_wait(api, swarm, progress=lambda n: typer.echo(
        f"  {n} workers still running"))
    if unsynced:
        _fail([*(f"NOT synced: {n}: {e}" for n, e in sorted(unsynced.items())),
               (f"released, but {len(unsynced)} agents' final sync failed: their bucket copies "
               "are older than their last turns")])
    typer.echo(f"released: every agent synced, compute freed; `flock launch {swarm}` resumes")


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
    sources: dict[str, str] = {}
    try:
        spec = read_spec(spec_file, env_files or (), sources=sources)
        done = ops.update(Client(), spec, spec_file, sources=sources, preflight=preflight,
                          strict=strict, replicas=replicas, check=check, source=ref_header)
    except SpecError as e:
        for line in e.lines:
            typer.echo(line, err=True)
        raise typer.Exit(1) from None
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


# Hidden aliases: agents of runs started before flock-agent still call `flock spawn|remove`.
app.command("spawn", hidden=True)(spawn)
app.command("remove", hidden=True)(remove)

if __name__ == "__main__":
    app()
