"""`flock status --events`: one readable line per normalised event; `--json` for JSONL."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from flock import cli
from flock.models import Event
from flock.status import normalise
from flock.worker.events import SessionLog

TS = datetime(2026, 10, 19, 8, 54, 50, tzinfo=UTC)


def stream(tmp_path) -> list[dict]:
    log = SessionLog(tmp_path / "a.jsonl", swarm="s", agent="leader-0000", harness="harbor",
                     cwd="/w")
    evs = [
        log.custom("session_started", {"session_id": "abc", "resumed": False, "parent": None},
                   ts=TS),
        log.message({"role": "user", "content": "Go: spawn the 5 analyst children now."},
                    turn_id="t-1", ts=TS),
        log.message({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hidden"},
            {"type": "text", "text": "Spawning\n now."},
            {"type": "toolCall", "id": "1", "name": "Bash",
             "arguments": {"command": "flock spawn analyst --key c1"}}]},
            turn_id="t-1", ts=TS),
        log.message({"role": "toolResult", "toolName": "Bash",
                     "content": [{"type": "text", "text": "leader-0000-analyst-0001"}]},
                    turn_id="t-1", ts=TS),
        log.custom("turn_ended", {"status": "completed", "error": None,
                                  "tokens": {"claude-haiku-5-5": {"input": 120, "output": 30}}},
                   turn_id="t-1", ts=TS),
        Event(type="spawned", ts=TS, agent="leader-0000",
              data={"child": "leader-0000-analyst-0001", "type": "analyst", "depth": 0,
                    "key": "c1"}),
    ]
    return [json.loads(e.model_dump_json()) for e in evs]


AGENT = {"name": "leader-0000", "swarm": "s", "type": "leader"}


def test_pretty_lines(tmp_path):
    lines = [cli.pretty_event(normalise(e, "s", AGENT)) for e in stream(tmp_path)]
    assert lines == [
        "08:54:50     - agent.session_started session_id=abc resumed=False",
        "08:54:50   t-1 user     Go: spawn the 5 analyst children now.",
        '08:54:50   t-1 assistant Spawning now. [Bash {"command": "flock spawn analyst --key c1"}]',
        "08:54:50   t-1 tool     Bash: leader-0000-analyst-0001",
        "08:54:50   t-1 agent.turn_ended completed claude-haiku-5-5 in=120 out=30",
        "08:54:50     - agent.spawned child=leader-0000-analyst-0001 type=analyst depth=0 key=c1",
    ]
