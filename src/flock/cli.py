"""`flock` CLI: a thin client over the controller API.

Config: FLOCK_URL and FLOCK_TOKEN (sent as X-Flock-Token; a `*.hf.jobs` URL or
FLOCK_PROXY_AUTH=hf also sends HF_TOKEN as the bearer the jobs proxy wants). Inside an agent the worker sets FLOCK_AGENT_TOKEN (that
agent's scoped token, which wins over FLOCK_TOKEN), FLOCK_AGENT and FLOCK_PARENT; see
docs/spawn.md for what an agent may call.
"""

from __future__ import annotations

import json
import os
import random
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
import yaml
from pydantic import ValidationError

from flock.models import AGENT_NAME_RE, SwarmSpec

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


def client() -> httpx.Client:
    """Overridden in tests to point at an in-process app."""
    url = os.environ.get("FLOCK_URL", "http://localhost:7860")
    return httpx.Client(base_url=url, headers=auth_headers(url), timeout=60)


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
    """Only a name shaped like an agent's (`...-NNNN`) costs a lookup."""
    if not AGENT_NAME_RE.match(name):
        return True
    with client() as c:
        return _request(c, "GET", f"/swarms/{name}").status_code == 200


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


def load_spec(spec_file: str) -> SwarmSpec:
    """Read a spec and inline each type's system_prompt_file (relative to the spec file)."""
    path = Path(spec_file)
    try:
        spec = SwarmSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as e:
        for err in e.errors():
            where = ".".join(str(x) for x in err["loc"])
            typer.echo(f"{spec_file}: {where + ': ' if where else ''}{err['msg']}", err=True)
        raise typer.Exit(1) from None
    for name, ts in spec.types.items():
        if ts.system_prompt_file and ts.system_prompt is None:
            src = Path(ts.system_prompt_file)
            src = src if src.is_absolute() else path.parent / src
            if not src.is_file():
                typer.echo(f"type {name}: system_prompt_file {src} not found", err=True)
                raise typer.Exit(1)
            ts.system_prompt = src.read_text(encoding="utf-8")
    c = spec.comms
    if c is not None:  # kickoff and protocol files, next to the spec (docs/comms.md)
        def read(rel: str, what: str) -> str:
            src = Path(rel)
            src = src if src.is_absolute() else path.parent / src
            if not src.is_file():
                typer.echo(f"comms.{what}: {src} not found", err=True)
                raise typer.Exit(1)
            return src.read_text(encoding="utf-8")

        if c.kickoff_file:
            c.kickoff = read(c.kickoff_file, "kickoff_file")
        c.protocols_text = [read(f, "protocols") for f in c.protocols]
    return spec


def _provider_backed(spec: SwarmSpec) -> bool:
    return any(ts.resolved_model is not None for ts in spec.types.values())


# Harnesses that cannot add request-body fields, so a model's `extra` is kept but not sent.
# The `llm` harness (and any harness not listed) sends it.
EXTRA_NOT_SENT = frozenset({"claude", "codex", "pi"})


def _warn_extra(spec: SwarmSpec) -> None:
    for name, ts in spec.types.items():
        if ts.harness.value in EXTRA_NOT_SENT and ts.resolved_model is not None and ts.resolved_model.extra:
            typer.echo(f"warning: type {name}: `extra` is kept in the spec but not sent by the "
                       f"{ts.harness.value} harness (docs/open-models.md)", err=True)


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
    st = _call("PUT", f"/swarms/{spec.swarm}", json=spec.model_dump(mode="json"))
    _print_status(st)


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
def status(swarm: str = typer.Argument(None)) -> None:
    """Swarm status: counts by type and state. Given an agent name (or inside an agent), that
    agent's status as JSON."""
    if not swarm and _as_agent():
        swarm = os.environ.get("FLOCK_AGENT")
    if swarm and (_as_agent() or not _is_swarm(swarm)):
        typer.echo(json.dumps(_call("GET", f"/agents/{swarm}"), indent=2))
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
    --pretty prints time, turn, kind and the text or a summary instead of JSON."""
    deadline = time.monotonic() + timeout
    while True:
        poll = follow or wait
        page = _call(
            "GET", f"/agents/{name}/events", params={"after": after, "wait": 25 if poll else 0}
        )
        for ev in page["events"]:
            typer.echo(pretty_event(ev) if pretty else json.dumps(ev))
        after = page["next"]
        if wait and (page["events"] or time.monotonic() >= deadline):
            typer.echo(f"next: {after}", err=True)
            break
        if not poll:
            break


@app.command()
def scale(swarm: str, type: str, replicas: int) -> None:
    """Set a type's replicas (edits the stored spec and re-applies it)."""
    spec = SwarmSpec.model_validate(_call("GET", f"/swarms/{swarm}")["spec"])
    if type not in spec.types:
        typer.echo(f"no type {type!r} in swarm {swarm!r}", err=True)
        raise typer.Exit(1)
    spec.types[type].replicas = replicas
    _print_status(_call("PUT", f"/swarms/{swarm}", json=spec.model_dump(mode="json")))


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
            w.get("last_error") or "",
        ]
        for w in _call("GET", "/workers", params={"swarm": swarm})
    ]
    _table(rows, ["id", "type", "state", "job", "agents", "crashes", "deadline", "last_heartbeat",
                  "last_error"])
    typer.echo("full job logs: hf jobs logs <job>", err=True)


if __name__ == "__main__":
    app()
