"""`python -m flock.world serve|doctor|knobs|set` (docs/world.md): a world author's and a
world host's commands, not the operator CLI. knobs/set reach the controller as `flock` does."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from flock.cli import _call, _table, source_header

world_app = typer.Typer(help="Worlds: shared state behind tools (docs/world.md).",
                        no_args_is_help=True)


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
        typer.echo(f"python -m flock.world serve needs the controller extra: {e}", err=True)
        raise typer.Exit(1) from None
    serve(target, host=host, port=port, token_env=token_env, insecure=insecure)


@world_app.command("doctor")
def world_doctor(
    url: str = typer.Argument(..., help="The world's URL."),
    token: str = typer.Option(None, "--token", envvar="FLOCK_WORLD_TOKEN", help="The world secret."),
    calls: int = typer.Option(0, "--calls", help="Concurrent calls of each read-only tool (0: none)."),
) -> None:
    """Check a world against the author checklist (docs/world.md); non-zero on a hard failure."""
    from flock.world.doctor import doctor

    lines, failed = doctor(url, token, calls)
    typer.echo("\n".join(lines))
    raise typer.Exit(1 if failed else 0)


# -- the world's admin API through the controller (docs/world.md, "Admin API") ----------------

@world_app.command("knobs")
def world_knobs(swarm: str) -> None:
    """The world's knobs (value, default, range) and paused lanes."""
    out = _call("GET", f"/swarms/{swarm}/world/knobs")
    rows = [[n, json.dumps(k["value"]), json.dumps(k["default"]), k["type"],
             f"{'' if k['min'] is None else k['min']}..{'' if k['max'] is None else k['max']}",
             k["doc"]] for n, k in out["knobs"].items()]
    _table(rows, ["knob", "value", "default", "type", "range", "doc"])
    if out.get("parked"):
        typer.echo("world is parked (not built yet; it retries)")
    for lane, why in out.get("paused", {}).items():
        typer.echo(f"paused: {lane}{f' ({why})' if why else ''}")


@world_app.command("set")
def world_set(
    swarm: str,
    assignments: Annotated[list[str], typer.Argument(help="name=value (JSON, else a string)")],
    reason: str = typer.Option("", "--reason", help="Recorded in the swarm's events."),
) -> None:
    """Set world knobs at runtime; the world keeps them across restarts."""
    values = {}
    for a in assignments:
        name, eq, raw = a.partition("=")
        if not eq or not name:
            typer.echo(f"expected name=value, got {a!r}", err=True)
            raise typer.Exit(2)
        try:
            values[name] = json.loads(raw)
        except ValueError:
            values[name] = raw
    out = _call("PATCH", f"/swarms/{swarm}/world/knobs", json=values, params={"reason": reason},
                headers=source_header(None))
    for c in out["changes"]:
        typer.echo(f"{c['name']}: {json.dumps(c['old'])} -> {json.dumps(c['new'])}")


if __name__ == "__main__":
    world_app()
