"""`flock` CLI: a thin client over the controller API.

Config: FLOCK_URL and FLOCK_TOKEN (sent as X-Flock-Token; a `*.hf.jobs` URL or
FLOCK_PROXY_AUTH=hf also sends HF_TOKEN as the bearer the jobs proxy wants). Without FLOCK_URL,
a live `flock run` in this directory (`.flock/current`, docs/local.md) is used. Inside an agent the worker sets FLOCK_AGENT_TOKEN (that
agent's scoped token, which wins over FLOCK_TOKEN), FLOCK_AGENT and FLOCK_PARENT; see
docs/spawn.md for what an agent may call.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
import yaml
from pydantic import ValidationError

from flock.models import AGENT_NAME_RE, SwarmSpec, comms_tools, load_spec_yaml, loop_refusals

app = typer.Typer(help="Host coding-harness agent swarms on HF Jobs.", no_args_is_help=True)


def proxy_auth(url: str, env=os.environ) -> bool:
    """True when `url` is behind the HF Jobs proxy (a job-mode controller seen from outside its
    network group): the proxy wants `Authorization: Bearer <HF token>` (docs/deploy-job.md)."""
    if env.get("FLOCK_PROXY_AUTH") == "hf":
        return True
    return (httpx.URL(url).host or "").endswith(".hf.jobs")


def auth_headers(url: str, env=os.environ) -> dict[str, str]:
    """`X-Flock-Token` always. `Authorization` carries the HF token behind the jobs proxy, and
    the flock token otherwise (what Space-mode controllers before X-Flock-Token expect)."""
    token = env.get("FLOCK_AGENT_TOKEN") or env.get("FLOCK_TOKEN", "")
    headers = {"X-Flock-Token": token}
    if proxy_auth(url, env):
        hf = env.get("HF_TOKEN")
        if not hf:
            from huggingface_hub import get_token

            hf = get_token()
        if not hf:
            typer.echo(f"{url} is behind the HF Jobs proxy: set HF_TOKEN (or `hf auth login`)",
                       err=True)
            raise typer.Exit(1)
        headers["Authorization"] = f"Bearer {hf}"
    else:
        headers["Authorization"] = f"Bearer {token}"
    return headers


CURRENT_FILE = Path(".flock") / "current"  # written by `flock run` (docs/local.md)


def local_run(env=os.environ) -> dict | None:
    """The live `flock run` this CLI talks to when FLOCK_URL is unset: `$FLOCK_CURRENT` or
    `.flock/current` in the working directory, if its process is still alive."""
    if env.get("FLOCK_URL"):
        return None
    from flock.current import read_current

    return read_current(Path(env.get("FLOCK_CURRENT") or CURRENT_FILE))


def client() -> httpx.Client:
    """Overridden in tests to point at an in-process app."""
    env = os.environ
    if (run := local_run(env)) is not None:
        env = {**env, "FLOCK_URL": run["url"], "FLOCK_TOKEN": run["token"]}
        env.pop("FLOCK_AGENT_TOKEN", None)
    url = env.get("FLOCK_URL", "http://localhost:7860")
    return httpx.Client(base_url=url, headers=auth_headers(url, env), timeout=60)


RETRY_ATTEMPTS = 5  # requests in all, per call
RETRY_BASE = 1.0  # s, first backoff before jitter
RETRY_CAP = 30.0  # s, longest single wait, Retry-After included
_sleep = time.sleep  # tests replace it


def _retry_after(r: httpx.Response) -> float | None:
    raw = r.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time()) if when.tzinfo else None


def _retryable(method: str, r: httpx.Response | None) -> bool:
    """429 (the Space edge throttling 1,000 agents) and 502/503/504 never reached the app or
    were not processed: retry any method. A 500 or a dropped connection may have been
    processed, so only reads retry on those."""
    if r is None:
        return method == "GET"
    return r.status_code in (429, 502, 503, 504) or (r.status_code == 500 and method == "GET")


def _request(c: httpx.Client, method: str, path: str, **kw) -> httpx.Response:
    """One controller request, retried up to RETRY_ATTEMPTS times with exponential backoff and
    full jitter, never sooner than Retry-After."""
    for attempt in range(RETRY_ATTEMPTS):
        last = attempt == RETRY_ATTEMPTS - 1
        try:
            r = c.request(method, path, **kw)
        except httpx.TransportError:
            if last or not _retryable(method, None):
                raise
            r = None
        if r is not None and (last or not _retryable(method, r)):
            return r
        backoff = random.uniform(0, min(RETRY_CAP, RETRY_BASE * 2 ** attempt))
        wait = max(backoff, (_retry_after(r) or 0.0) if r is not None else 0.0)
        _sleep(min(RETRY_CAP, wait))
    raise AssertionError("unreachable")


def _call(method: str, path: str, *, timeout: float | None = None, **kw) -> Any:
    if timeout is not None:  # per request, without the TestClient's `timeout` deprecation
        kw["extensions"] = {"timeout": httpx.Timeout(timeout).as_dict()}
    with client() as c:
        r = _request(c, method, path, **kw)
    if r.status_code >= 400:
        typer.echo(f"error {r.status_code}: {r.text}", err=True)
        raise typer.Exit(1)
    return r.json()


def _as_agent() -> bool:
    """Running inside an agent with its scoped token: names are agents, never swarms."""
    return bool(os.environ.get("FLOCK_AGENT_TOKEN"))


def _is_swarm(name: str) -> bool:
    """A swarm, unless it names an agent: a numeric agent name (`...-NNNN`) that is not a
    swarm, or any other name (`naming: words`, e.g. `amber-otter`) that is an agent and not a
    swarm. A name without a hyphen is no agent's: no lookup."""
    if "-" not in name:
        return True
    with client() as c:
        if _request(c, "GET", f"/swarms/{name}").status_code == 200:
            return True
        if AGENT_NAME_RE.match(name):
            return False
        return _request(c, "GET", f"/agents/{name}").status_code != 200


def source_header(spec_file: str | Path | None) -> dict[str, str]:
    """`X-Flock-Source` for an apply (`flock history`): who applied it and, when the spec's
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
    """A spec that cannot be applied: the lines say why (printed to stderr by the CLI, shown in
    the `flock run --watch` view)."""

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


def load_spec(spec_file: str) -> SwarmSpec:
    """`read_spec`, exiting 1 with the reasons on stderr."""
    try:
        return read_spec(spec_file)
    except SpecError as e:
        for line in e.lines:
            typer.echo(line, err=True)
        raise typer.Exit(1) from None


def read_spec(spec_file: str) -> SwarmSpec:
    """Read a spec and inline each type's system_prompt_file (plus system_prompt_append_file)
    and loop prompt files, and the comms kickoff and protocol files (relative to the spec
    file). Raises SpecError."""
    path = Path(spec_file)
    try:
        spec = SwarmSpec.model_validate(load_spec_yaml(path.read_text(encoding="utf-8")))
    except ValidationError as e:
        lines = []
        for err in e.errors():
            where = ".".join(str(x) for x in err["loc"])
            lines.append(f"{spec_file}: {where + ': ' if where else ''}{err['msg']}")
        raise SpecError(lines) from None
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
    rep = _call("POST", f"/swarms/{spec.swarm}/preflight", json=spec.model_dump(mode="json"),
                timeout=300)
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


@app.command()
def apply(
    spec_file: str,
    preflight: bool = typer.Option(
        None, "--preflight/--skip-preflight",
        help="Probe provider-backed models before applying (default: on when the spec has any)."),
    strict: bool = typer.Option(
        False, "--strict", help="Treat models that were not probed (unverified) as failures."),
) -> None:
    """Apply a swarm spec (YAML). Refuses when a provider-backed model fails preflight; a model
    that could not be probed is reported as unverified and applied anyway unless --strict.
    A tools folder (docs/tools.md) is checked, uploaded and named in the spec by its hash;
    re-applying with a changed folder switches running agents at their next turn."""
    spec = load_spec(spec_file)
    _check_loops(spec)
    _warn_extra(spec)
    bundle = _tools_bundle(spec, spec_file)
    if preflight is None:
        preflight = _provider_backed(spec)
    if preflight:
        if not _preflight(spec, strict):
            typer.echo("preflight failed: not applying (fix the provider or pass --skip-preflight)",
                       err=True)
            raise typer.Exit(1)
    elif unprobed := _unprobed(spec):
        _warn_unverified(unprobed, "preflight skipped")
        if strict:
            typer.echo("--strict: not applying unverified models", err=True)
            raise typer.Exit(1)
    spec.tools_version = None
    if bundle is not None:
        up = _call("POST", f"/swarms/{spec.swarm}/tools", content=bundle.data,
                   headers={"content-type": "application/gzip"}, timeout=120)
        spec.tools_version = up["version"]
        names = ", ".join(up["tools"]) or "none"
        typer.echo(f"tools: version {up['version']} ({len(bundle.data)} bytes): {names}")
    st = _call("PUT", f"/swarms/{spec.swarm}", params={"replicas": "spec"},
               json=spec.model_dump(mode="json"), headers=source_header(spec_file))
    _print_status(st)


def _detail(r: httpx.Response) -> str:
    try:
        d = r.json().get("detail")
    except ValueError:
        d = None
    return d if isinstance(d, str) else r.text


def update_swarm(c: httpx.Client, spec_file: str, *, preflight: bool | None = None,
                 strict: bool = False, replicas: str = "keep", check: bool = False) -> list[str]:
    """Re-read the experiment folder and apply it to the running swarm (docs/live.md): the
    spec, its prompt files and the tools folder, as `flock apply` reads them. Returns the
    lines to show (warnings, then what changed, or "no changes"); raises SpecError with the
    refusal when the folder cannot be applied, and then nothing was changed. Preflight runs
    by default only when the spec names a provider-backed model the running one does not.
    `replicas="keep"` keeps the live counts of existing types; `check` runs the controller's
    validation (dry run) and stores nothing."""
    from flock.live import diff_specs, keep_replicas, world_tool_changes
    from flock.tools.loader import ToolsError
    from flock.tools.spec import prepare

    spec = read_spec(spec_file)
    refused, warnings = loop_problems(spec)
    if refused:
        raise SpecError(refused)
    warnings += extra_warnings(spec)
    try:
        bundle = prepare(spec, spec_file)
    except ToolsError as e:
        raise SpecError([f"{spec_file}: tools: {e}"]) from None
    r = _request(c, "GET", f"/swarms/{spec.swarm}")
    if r.status_code == 404:
        raise SpecError([(f"no swarm {spec.swarm!r} is running on this controller: start it "
                          "with `flock run` (compute: local) or `flock apply`")])
    if r.status_code >= 400:
        raise SpecError([f"error {r.status_code}: {_detail(r)}"])
    old = SwarmSpec.model_validate(r.json()["spec"])
    if preflight is None:
        preflight = bool(set(_unprobed(spec)) - set(_unprobed(old)))
    if preflight:
        r = _request(c, "POST", f"/swarms/{spec.swarm}/preflight", json=spec.model_dump(mode="json"),
                     extensions={"timeout": httpx.Timeout(300).as_dict()})
        if r.status_code >= 400:
            raise SpecError([f"preflight error {r.status_code}: {_detail(r)}"])
        rep = r.json()
        bad = [f"preflight {x['model']}: {'unverified' if x.get('unverified') else 'FAIL'}"
               + (f" ({x['error']})" if x.get("error") else "")
               for x in rep["results"] if not x["ok"] and (strict or not x.get("unverified"))]
        if bad:
            raise SpecError([*bad, ("preflight failed: not updating (fix the provider or pass "
                                    "--skip-preflight)")])
    spec.tools_version = None
    names: tuple[list[str], list[str]] | None = None
    if bundle is not None:
        r = _request(c, "POST", f"/swarms/{spec.swarm}/tools", content=bundle.data,
                     headers={"content-type": "application/gzip"},
                     extensions={"timeout": httpx.Timeout(120).as_dict()})
        if r.status_code >= 400:
            raise SpecError([f"refused: tools upload ({r.status_code}): {_detail(r)}"])
        spec.tools_version = r.json()["version"]
        if spec.tools_version != old.tools_version:
            listed = _request(c, "GET", f"/swarms/{spec.swarm}/tools").json().get("bundles", [])
            before = next((b["tools"] for b in listed if b["version"] == old.tools_version), [])
            names = (before, r.json()["tools"])
    kept = keep_replicas(old, spec) if replicas == "keep" else []
    lines = diff_specs(old, spec, names)
    if not lines and not check:
        return [*warnings, *kept, "no changes"]
    r = _request(c, "PUT", f"/swarms/{spec.swarm}", json=spec.model_dump(mode="json"),
                 params={"replicas": "spec", **({"dry_run": 1} if check else {})},
                 headers=source_header(spec_file))
    if r.status_code >= 400:
        raise SpecError([f"refused: {_detail(r)}"])
    if check:  # the stored spec is untouched: the new manifest is in the reply
        new = SwarmSpec.model_validate(r.json()["spec"])
    else:
        new = SwarmSpec.model_validate(_request(c, "GET", f"/swarms/{spec.swarm}").json()["spec"])
    wt = world_tool_changes(old, new)
    return [*warnings, *kept, *lines, *([wt] if wt else []),
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
) -> None:
    """Re-read an experiment folder (spec.yaml, prompts, tools/) and apply what changed to the
    running swarm (docs/live.md): new tools version, prompts, loop and comms settings, limits,
    replicas. Running agents take tools, prompts and grants at their next turn boundary.
    Prints what changed; a refused change applies nothing."""
    path = Path(target)
    spec_file = path / "spec.yaml" if path.is_dir() else path
    try:
        with client() as c:
            if replicas not in ("keep", "spec"):
                raise SpecError([f"--replicas {replicas!r}: use keep or spec"])
            lines = update_swarm(c, str(spec_file), preflight=preflight, strict=strict,
                                 replicas=replicas, check=check)
    except SpecError as e:
        for line in e.lines:
            typer.echo(line, err=True)
        raise typer.Exit(1) from None
    for line in lines:
        typer.echo(line, err=line.startswith("warning:"))


def _grant_swarm(target: str, swarm: str | None) -> str:
    """The swarm a grant target is in: --swarm, else the agent's swarm, else the one swarm
    with a type of that name."""
    if swarm:
        return swarm
    with client() as c:
        if AGENT_NAME_RE.match(target) or "-" in target:  # numeric or words agent names
            r = _request(c, "GET", f"/agents/{target}")
            if r.status_code == 200:
                return r.json()["swarm"]
        r = _request(c, "GET", "/swarms")
    if r.status_code >= 400:
        typer.echo(f"error {r.status_code}: {r.text}", err=True)
        raise typer.Exit(1)
    found = [st["swarm"] for st in r.json() if target in st["types"]]
    if len(found) != 1:
        typer.echo(f"{target!r}: no agent of that name and {len(found)} swarms with such a type; "
                   "pass --swarm", err=True)
        raise typer.Exit(1)
    return found[0]


def _current_grant(swarm: str, target: str):
    from flock.models import Grant

    rows = _call("GET", f"/swarms/{swarm}/grants")["grants"]
    row = next((g for g in rows if g["target"] == target), None)
    return Grant.model_validate(row["grant"]) if row else Grant()


def _print_grant(ack: dict) -> None:
    from flock.models import Grant

    g = Grant.model_validate(ack["grant"])
    what = "revoked" if g.empty() else g.describe()
    typer.echo(f"{ack['kind']} {ack['target']}: {what} (from each agent's next turn)")


@app.command()
def grant(
    target: str = typer.Argument(..., help="a type (every agent of it) or one agent"),
    capabilities: Annotated[list[str] | None, typer.Argument(metavar="CAPABILITY[=VALUE]...")] = None,
    swarm: str = typer.Option(None, "--swarm", help="default: the swarm of the target"),
) -> None:
    """Grant capabilities to a running type or agent, from each agent's next turn
    (docs/live.md): `tools=+name,-name` (folder or comms tools; `+name` alone works too),
    `comms.post=on|off`, `bash=on|off` (llm harness), `limits.max_turns=N`,
    `limits.budget_usd=X`. Grants are an overlay on the spec: `flock update` keeps them."""
    from flock.live import grant_patch

    if not capabilities:
        typer.echo("usage: flock grant TARGET CAPABILITY[=VALUE]... (e.g. tools=+mul bash=off)",
                   err=True)
        raise typer.Exit(2)
    swarm = _grant_swarm(target, swarm)
    try:
        g = grant_patch(_current_grant(swarm, target), capabilities)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from None
    _print_grant(_call("PUT", f"/swarms/{swarm}/grants/{target}", json=g.model_dump(mode="json")))


@app.command()
def revoke(
    target: str,
    capabilities: Annotated[list[str] | None, typer.Argument(metavar="[CAPABILITY]...")] = None,
    swarm: str = typer.Option(None, "--swarm", help="default: the swarm of the target"),
) -> None:
    """Take back a target's grant (all of it, or the named capabilities: `tools`,
    `tools=name`, `comms.post`, `bash`, `limits`, `limits.max_turns`, `limits.budget_usd`)."""
    from flock.live import revoke_patch

    swarm = _grant_swarm(target, swarm)
    if not capabilities:
        _call("DELETE", f"/swarms/{swarm}/grants/{target}")
        typer.echo(f"{target}: all grants revoked (from each agent's next turn)")
        return
    try:
        g = revoke_patch(_current_grant(swarm, target), capabilities)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from None
    _print_grant(_call("PUT", f"/swarms/{swarm}/grants/{target}", json=g.model_dump(mode="json")))


@app.command()
def grants(swarm: str = typer.Argument(None, help="default: every swarm")) -> None:
    """List the grants in force (docs/live.md)."""
    from flock.models import Grant

    names = [swarm] if swarm else [st["swarm"] for st in _call("GET", "/swarms")]
    rows = [[sw, g["kind"], g["target"], Grant.model_validate(g["grant"]).describe()]
            for sw in names for g in _call("GET", f"/swarms/{sw}/grants")["grants"]]
    if not rows:
        typer.echo("no grants")
        return
    _table(rows, ["swarm", "kind", "target", "grant"])


@app.command()
def run(
    spec_file: str,
    data_dir: str = typer.Option(".flock", "--data-dir",
                                  help="Where the run dir and `current` go."),
    until: str = typer.Option(
        None, "--until",
        help="Stop after a duration (30s, 10m, 2h) or `idle`: every agent idle with nothing "
             "pending, after at least one turn (and one per agent with --prompt), every loop over."),
    prompt_text: str = typer.Option(None, "--prompt",
                                    help="Send this prompt to every agent once at start."),
    preflight: bool = typer.Option(
        None, "--preflight/--skip-preflight",
        help="Probe provider-backed models first (default: on when the spec has any)."),
    strict: bool = typer.Option(False, "--strict", help="Unverified models count as failures."),
    grace: float = typer.Option(10.0, "--grace",
                                help="Seconds in-flight turns get to finish at stop."),
    watch: bool = typer.Option(
        False, "--watch",
        help="Watch the experiment folder (spec, prompts, tools/) and apply each saved change "
             "to the running swarm, as `flock update` does (docs/live.md)."),
) -> None:
    """Run a `compute: local` swarm on this machine: controller on 127.0.0.1, workers as
    subprocesses, a live view in the terminal, a run bundle at the end (docs/local.md). Other
    commands (`status`, `agents`, `prompt`, `events`, ...) reach it from this directory."""
    import asyncio

    from flock.models import parse_duration

    try:
        from flock.controller.localrun import RunOptions, run_local
    except ImportError as e:  # fastapi / uvicorn: the `controller` extra
        typer.echo(f"flock run needs the controller extra (pip install 'flock[controller]'): {e}",
                   err=True)
        raise typer.Exit(1) from None
    spec = load_spec(spec_file)
    if spec.compute.kind != "local":
        typer.echo(f"{spec_file}: compute is {spec.compute.kind!r}; `flock run` runs "
                   "`compute: local` specs. For hf-jobs, deploy a controller and use "
                   "`flock apply` (docs/deploy-job.md), or set `compute: local`.", err=True)
        raise typer.Exit(1)
    _check_loops(spec)
    _warn_extra(spec)
    bundle = _tools_bundle(spec, spec_file)
    seconds, idle = None, False
    if until == "idle":
        idle = True
    elif until:
        try:
            seconds = parse_duration(until).total_seconds()
        except ValueError as e:
            typer.echo(f"--until: {e} (or `idle`)", err=True)
            raise typer.Exit(2) from None
    if preflight is None:
        preflight = _provider_backed(spec)
    elif not preflight and (unprobed := _unprobed(spec)):
        _warn_unverified(unprobed, "preflight skipped")
    opts = RunOptions(spec=spec, data_dir=Path(data_dir), until=seconds, until_idle=idle,
                      prompt=prompt_text, preflight=preflight, strict=strict, grace=grace,
                      tools=bundle.data if bundle is not None else None,
                      watch=Path(spec_file) if watch else None)
    result = asyncio.run(run_local(opts))
    raise typer.Exit(result.exit_code)


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


@app.command("preflight")
def preflight_cmd(
    spec_file: str,
    strict: bool = typer.Option(False, "--strict", help="Exit 1 on unverified models too."),
) -> None:
    """Send one small tool-calling request to each provider-backed model in a spec (through
    the controller, which holds the keys). Exit 1 if any model fails; models that could not be
    probed are reported as unverified (exit 1 only with --strict)."""
    spec = load_spec(spec_file)
    _warn_extra(spec)
    if not _preflight(spec, strict):
        raise typer.Exit(1)


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
    if not swarm and _as_agent():
        swarm = os.environ.get("FLOCK_AGENT")
    if swarm and (_as_agent() or not _is_swarm(swarm)):
        typer.echo(json.dumps(_call("GET", f"/agents/{swarm}"), indent=2))
        return
    if watch or as_json:
        _health_view(swarm, as_json=as_json, watch=watch)
        return
    if swarm:
        _print_status(_call("GET", f"/swarms/{swarm}")["status"])
        return
    swarms = _call("GET", "/swarms")
    if not swarms:
        typer.echo("no swarms")
    for st in swarms:
        _print_status(st)
        typer.echo("")


def _health_of(swarm: str) -> dict | None:
    with client() as c:
        r = _request(c, "GET", f"/swarms/{swarm}/health")
    return r.json() if r.status_code == 200 else None


def _health_view(swarm: str | None, *, as_json: bool, watch: bool) -> None:
    while True:
        sts = [_call("GET", f"/swarms/{swarm}")["status"]] if swarm else _call("GET", "/swarms")
        docs = [{"status": st, "health": _health_of(st["swarm"])} for st in sts]
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
def usage(
    swarm: str,
    by: Annotated[str, typer.Option("--by", help="type (all hours) or hour (UTC hour x type)")
                  ] = "type",
) -> None:
    """Tokens a swarm used, from turn ends (docs/health.md)."""
    if by not in ("type", "hour"):
        raise typer.BadParameter("--by type or --by hour")
    rows = _call("GET", f"/swarms/{swarm}/usage", params={"by": by})["rows"]
    keys = ["input", "output", "cache_read", "cache_write", "reasoning"]
    head = (["hour"] if by == "hour" else []) + ["type", *keys]
    _table([([r["hour"]] if by == "hour" else []) + [r["type"], *(r.get(k, 0) for k in keys)]
            for r in rows], head)


@app.command()
def history(
    swarm: str,
    diff: Annotated[tuple[int, int] | None, typer.Option(
        "--diff", metavar="A B", help="What changed from generation A to generation B.")] = None,
) -> None:
    """A swarm's spec generations (docs/live.md): when each apply that changed the spec ran, who
    ran it, from which commit, and what it changed. --diff A B diffs two stored specs."""
    if diff is not None:
        from flock.live import diff_specs

        a, b = (SwarmSpec.model_validate(_call("GET", f"/swarms/{swarm}/history/{g}")["spec"])
                for g in diff)
        for line in diff_specs(a, b) or ["no changes"]:
            typer.echo(line)
        return
    rows = []
    for h in _call("GET", f"/swarms/{swarm}/history"):
        src, changes = h["source"], h["diff_summary"]
        git = " ".join(x for x in (src.get("git_sha", "")[:8] + ("+dirty" if src.get("dirty")
                                                                   else ""), src.get("ref")) if x)
        more = f" (+{len(changes) - 1} more)" if len(changes) > 1 else ""
        rows.append([h["generation"], h["ts"][:19].replace("T", " "), src.get("by") or "-",
                     git or "-", (changes[0] if changes else "-") + more])
    _table(rows, ["gen", "applied (UTC)", "by", "git", "changes"])


@app.command()
def agents(
    swarm: str,
    type: str = typer.Option(None, "--type"),
    state: str = typer.Option(None, "--state"),
    parent: str = typer.Option(None, "--parent", help="only the children of this agent"),
) -> None:
    """List a swarm's agents."""
    params = {k: v for k, v in (("type", type), ("state", state), ("parent", parent)) if v}
    rows, cursor = [], None
    while True:
        page = _call(
            "GET",
            f"/swarms/{swarm}/agents",
            params={**params, "cursor": cursor} if cursor else params,
        )
        for a in page["agents"]:
            rows.append(
                [
                    a["name"],
                    a["state"],
                    a.get("provider") or "-",
                    a["worker"],
                    a["turns"],
                    a["pending"],
                    f"${a['cost_usd']:.4f}",  # a short turn costs a fraction of a cent
                    a["last_heartbeat"],
                ]
            )
        cursor = page["cursor"]
        if not cursor:
            break
    _table(rows, ["name", "state", "provider", "worker", "turns", "pending", "cost",
                  "last_heartbeat"])


@app.command()
def agent(name: str) -> None:
    """One agent's status as JSON."""
    typer.echo(json.dumps(_call("GET", f"/agents/{name}"), indent=2))


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
    body = {"timeout": timeout, "reply_to": reply_to}
    if children:
        parent = parent or os.environ.get("FLOCK_AGENT")
        if not parent or swarm is not None or len(args) != 1:
            typer.echo("usage: flock prompt --children [--parent P] [--type T] [--state STATE] "
                       "TEXT (parent defaults to $FLOCK_AGENT)", err=True)
            raise typer.Exit(2)
        body.update(type=type, state=state, text=args[0])
        if skip_pending is not None:
            body["skip_pending"] = skip_pending
        typer.echo(json.dumps(_call("POST", f"/agents/{parent}/children/prompts", json=body)))
        return
    if type is not None:
        if swarm is None or len(args) != 1:
            typer.echo("usage: flock prompt --swarm S --type T [--state STATE] TEXT", err=True)
            raise typer.Exit(2)
        body.update(type=type, state=state, text=args[0])
        if skip_pending is not None:
            body["skip_pending"] = skip_pending
        typer.echo(json.dumps(_call("POST", f"/swarms/{swarm}/prompts", json=body)))
        return
    if swarm is not None or state is not None or len(args) != 2:
        typer.echo("usage: flock prompt NAME TEXT (or --swarm S --type T TEXT)", err=True)
        raise typer.Exit(2)
    body["text"] = args[1]
    typer.echo(json.dumps(_call("POST", f"/agents/{args[0]}/prompt", json=body)))


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
    deadline = time.monotonic() + timeout
    # agent-shaped names (`...-NNNN`) are agents, without the lookup `status` makes; other
    # names (`naming: words`) are agents unless a swarm has that name
    route = (f"/agents/{name}/events" if _as_agent() or AGENT_NAME_RE.match(name)
             or not _is_swarm(name) else f"/swarms/{name}/events")
    while True:
        poll = follow or wait
        page = _call("GET", route, params={"after": after, "wait": 25 if poll else 0})
        for ev in page["events"]:
            typer.echo(pretty_event(ev) if pretty else json.dumps(ev))
        after = page["next"]
        if wait and (page["events"] or time.monotonic() >= deadline):
            typer.echo(f"next: {after}", err=True)
            break
        if not poll:
            break


@app.command()
def loop(
    swarm: str,
    type: str = typer.Option(None, "--type", help="the agent type whose loop to stop or start"),
    stop: bool = typer.Option(False, "--stop", help="stop the loop (then its final round)"),
    start: bool = typer.Option(False, "--start", help="start the loop again from round 0"),
    final: bool = typer.Option(True, "--final/--no-final", help="with --stop: run the final round"),
) -> None:
    """A swarm's loops (docs/loop.md): state per type and per agent, or --stop / --start one."""
    if stop and start:
        typer.echo("--stop or --start, not both", err=True)
        raise typer.Exit(1)
    if stop or start:
        if not type:
            typer.echo("--stop/--start need --type", err=True)
            raise typer.Exit(1)
        body = {"type": type, "action": "stop" if stop else "start", "final": final}
        typer.echo(_loop_line(_call("POST", f"/swarms/{swarm}/loop", json=body)))
        return
    loops = [lp for lp in _call("GET", f"/swarms/{swarm}/loop") if not type or lp["type"] == type]
    if not loops:
        typer.echo("no loops")
    for lp in loops:
        typer.echo(_loop_line(lp))
        rows = [[n, a["phase"], a["rounds"], a["done_reason"], a["final"], a["errors"]]
                for n, a in (lp.get("per_agent") or {}).items()]
        if rows:
            _table(rows, ["agent", "phase", "rounds", "done", "final", "errors"])


@app.command()
def scale(swarm: str, type: str, replicas: int) -> None:
    """Set a type's replicas (edits the stored spec and re-applies it)."""
    spec = SwarmSpec.model_validate(_call("GET", f"/swarms/{swarm}")["spec"])
    if type not in spec.types:
        typer.echo(f"no type {type!r} in swarm {swarm!r}", err=True)
        raise typer.Exit(1)
    spec.types[type].replicas = replicas
    _print_status(_call("PUT", f"/swarms/{swarm}", params={"replicas": "spec"},
                        json=spec.model_dump(mode="json"), headers=source_header(None)))


@app.command()
def remove(swarm: str) -> None:
    """Drain and remove a swarm. Given a spawned agent's name (or inside an agent), remove that
    agent and everything it spawned."""
    if _as_agent() or not _is_swarm(swarm):
        typer.echo(json.dumps(_call("DELETE", f"/agents/{swarm}")))
        return
    typer.echo(json.dumps(_call("DELETE", f"/swarms/{swarm}")))


@app.command()
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
    body = {"type": type, "prompt": prompt, "key": key}
    ack = _call("POST", f"/agents/{parent}/spawn", json=body)
    typer.echo(json.dumps(ack) if as_json else ack["name"])


@app.command()
def whoami() -> None:
    """Who the configured token acts as: the controller, or an agent with its lineage."""
    typer.echo(json.dumps(_call("GET", "/whoami"), indent=2))


@app.command()
def workers(swarm: str) -> None:
    """List a swarm's workers."""
    found = _call("GET", "/workers", params={"swarm": swarm})

    def state(w: dict) -> str:  # compute external: a slot with no worker yet is waiting for one
        waiting = str(w["job_id"] or "").startswith("external-") and w["state"] == "launching"
        return "external" if waiting else w["state"]

    rows = [
        [
            w["id"],
            w["type"],
            state(w),
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
    elif any(str(w["job_id"] or "").startswith("external-") for w in found):
        typer.echo("compute external: job `external-...` = a slot waiting for (or served by) a "
                   "worker started outside flock; env files in the compute launch_dir", err=True)
    elif any(str(w["job_id"] or "").isdigit() for w in found):  # compute slurm
        typer.echo("full job logs: <compute log_dir>/<worker id>-<job>.out", err=True)
    else:
        typer.echo("full job logs: hf jobs logs <job>", err=True)


def add_pause_commands(app: typer.Typer) -> None:
    """`flock pause` / `flock resume` (docs/loop.md "Pause and resume")."""

    def _body(type_: list[str] | None, agent: list[str] | None) -> dict:
        return {"types": type_ or None, "agents": agent or None}

    def _show(r: dict) -> None:
        h = r["held"]
        held = [f"type {n}" for n in h["types"]] + [f"agent {n}" for n in h["agents"]]
        typer.echo(f"{r['swarm']}: held: {', '.join(held) or 'nothing'}")

    @app.command()
    def pause(
        swarm: str,
        reason: Annotated[str, typer.Option("--reason", help="why (shown in status and events)")],
        type_: Annotated[list[str] | None, typer.Option("--type", help="a type to hold "
                                                         "(repeatable; default: every type)")] = None,
        agent: Annotated[list[str] | None, typer.Option("--agent", help="an agent to hold "
                                                         "(repeatable)")] = None,
        interrupt: Annotated[bool, typer.Option("--interrupt", help="also stop the turns in "
                                                "flight")] = False,
    ) -> None:
        """Hold types or agents: no new prompts or loop rounds reach them (queued ones wait),
        rounds are not counted, until `flock resume`. Not `flock loop --stop`."""
        _show(_call("POST", f"/swarms/{swarm}/pause", json={
            **_body(type_, agent), "interrupt": interrupt, "reason": reason}))

    @app.command()
    def resume(
        swarm: str,
        type_: Annotated[list[str] | None, typer.Option("--type", help="a type to resume "
                                                         "(repeatable)")] = None,
        agent: Annotated[list[str] | None, typer.Option("--agent", help="an agent to resume "
                                                         "(repeatable)")] = None,
    ) -> None:
        """Lift holds (default: every hold); queued prompts and wakes go out."""
        _show(_call("POST", f"/swarms/{swarm}/resume", json=_body(type_, agent)))


add_pause_commands(app)


world_app = typer.Typer(help="Worlds: shared state behind tools (docs/world.md).",
                        no_args_is_help=True)
app.add_typer(world_app, name="world")


@world_app.command("serve")
def world_serve(
    target: str = typer.Argument(..., help="module:Class, imported from the current directory."),
    host: str = typer.Option(None, "--host", help="Default: $FLOCK_WORLD_HOST or 127.0.0.1."),
    port: int = typer.Option(None, "--port", help="Default: $FLOCK_WORLD_PORT or 8765."),
    token_env: str = typer.Option("FLOCK_WORLD_TOKEN", "--token-env",
                                  help="The variable holding the shared secret (X-World-Token)."),
    insecure: bool = typer.Option(False, "--insecure", help="Serve without a secret."),
) -> None:
    """Serve a flock.world.World subclass over HTTP (World API v1). A world flock starts gets
    its port and secret in the environment."""
    try:
        from flock.home import cache_defaults
        from flock.world.server import serve

        cache_defaults()  # no usable HOME (a container): $FLOCK_HOMELESS_ROOT, else the temp dir
    except ImportError as e:  # fastapi / uvicorn: the `controller` extra
        typer.echo(f"flock world serve needs the controller extra: {e}", err=True)
        raise typer.Exit(1) from None
    serve(target, host=host, port=port, token_env=token_env, insecure=insecure)


if __name__ == "__main__":
    app()
