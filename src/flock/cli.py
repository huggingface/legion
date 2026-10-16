"""`flock` CLI: a thin client over the controller API.

Config: FLOCK_URL and FLOCK_TOKEN. Inside an agent the worker sets FLOCK_AGENT_TOKEN (that
agent's scoped token, which wins over FLOCK_TOKEN), FLOCK_AGENT and FLOCK_PARENT; see
docs/spawn.md for what an agent may call.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import typer
import yaml

from flock.models import AGENT_NAME_RE, SwarmSpec

app = typer.Typer(help="Host coding-harness agent swarms on HF Jobs.", no_args_is_help=True)


def client() -> httpx.Client:
    """Overridden in tests to point at an in-process app."""
    url = os.environ.get("FLOCK_URL", "http://localhost:7860")
    token = os.environ.get("FLOCK_AGENT_TOKEN") or os.environ.get("FLOCK_TOKEN", "")
    return httpx.Client(base_url=url, headers={"Authorization": f"Bearer {token}"}, timeout=60)


def _call(method: str, path: str, **kw) -> Any:
    with client() as c:
        r = c.request(method, path, **kw)
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
        return c.get(f"/swarms/{name}").status_code == 200


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


def load_spec(spec_file: str) -> SwarmSpec:
    """Read a spec and inline each type's system_prompt_file (relative to the spec file)."""
    path = Path(spec_file)
    spec = SwarmSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    for name, ts in spec.types.items():
        if ts.system_prompt_file and ts.system_prompt is None:
            src = Path(ts.system_prompt_file)
            src = src if src.is_absolute() else path.parent / src
            if not src.is_file():
                typer.echo(f"type {name}: system_prompt_file {src} not found", err=True)
                raise typer.Exit(1)
            ts.system_prompt = src.read_text(encoding="utf-8")
    return spec


@app.command()
def apply(spec_file: str) -> None:
    """Apply a swarm spec (YAML)."""
    spec = load_spec(spec_file)
    st = _call("PUT", f"/swarms/{spec.swarm}", json=spec.model_dump(mode="json"))
    _print_status(st)


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
                    a["worker"],
                    a["turns"],
                    a["pending"],
                    f"${a['cost_usd']:.2f}",
                    a["last_heartbeat"],
                ]
            )
        cursor = page["cursor"]
        if not cursor:
            break
    _table(rows, ["name", "state", "worker", "turns", "pending", "cost", "last_heartbeat"])


@app.command()
def agent(name: str) -> None:
    """One agent's status as JSON."""
    typer.echo(json.dumps(_call("GET", f"/agents/{name}"), indent=2))


@app.command()
def prompt(
    name: str,
    text: str,
    timeout: str = typer.Option(None, "--timeout"),
    reply_to: str = typer.Option(None, "--reply-to"),
) -> None:
    """Send a prompt to an agent; prints the ack."""
    body = {"text": text, "timeout": timeout, "reply_to": reply_to}
    typer.echo(json.dumps(_call("POST", f"/agents/{name}/prompt", json=body)))


@app.command()
def events(
    name: str,
    after: int = typer.Option(0, "--after"),
    follow: bool = typer.Option(False, "--follow", "-f"),
    wait: bool = typer.Option(False, "--wait", help="block until an event after --after"),
    timeout: float = typer.Option(600, "--timeout", help="seconds --wait blocks at most"),
) -> None:
    """Print an agent's event stream (JSONL); --follow keeps long-polling; --wait returns once
    something new arrived (the last line on stderr is the cursor for the next --after)."""
    deadline = time.monotonic() + timeout
    while True:
        poll = follow or wait
        page = _call(
            "GET", f"/agents/{name}/events", params={"after": after, "wait": 25 if poll else 0}
        )
        for ev in page["events"]:
            typer.echo(json.dumps(ev))
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
        ]
        for w in _call("GET", "/workers", params={"swarm": swarm})
    ]
    _table(rows, ["id", "type", "state", "job", "agents", "crashes", "deadline", "last_heartbeat"])


if __name__ == "__main__":
    app()
