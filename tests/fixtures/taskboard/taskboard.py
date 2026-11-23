"""A shared task board world (docs/world.md), a test fixture: agents list, add, claim and finish tasks.

    cd tests/fixtures/taskboard && LEGION_WORLD_TOKEN=dev python -m legion.world serve taskboard:TaskBoard

State is one SQLite file in the current directory (`$TASKBOARD_DB`, default `taskboard.db`), so
it survives a restart of the world. On first start the board is seeded from `tasks.txt` next to
it (one task per line, `name: description`; `#` starts a comment). A claim is exclusive: a
second agent's claim fails with the owner's name. When legion removes an agent (`agent_done`),
its claimed tasks go back to open. Every change is logged in the `events` table with the
agent, request id and turn legion sent.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Literal

from legion.world import Agent, ToolError, World, tool

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
  name TEXT PRIMARY KEY, description TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
  owner TEXT, note TEXT NOT NULL DEFAULT '', created_by TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, agent TEXT NOT NULL,
  action TEXT NOT NULL, task TEXT, request_id TEXT, turn TEXT);
"""
NAME_MAX, TEXT_MAX = 60, 400


class TaskBoard(World):
    name = "taskboard"
    version = "0.1.0"
    description = ("A shared task board. Claim a task before you work on it, finish it with a "
                   "one-line note, release it if you stop. Add tasks you find missing.")

    def __init__(self, db: str | None = None, seed: str | None = None) -> None:
        self.db = sqlite3.connect(db or os.environ.get("TASKBOARD_DB", "taskboard.db"),
                                  check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        if self.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0:
            path = Path(seed or "tasks.txt")
            for line in path.read_text().splitlines() if path.is_file() else []:
                line = line.split("#", 1)[0].strip()
                if line:
                    name, _, desc = line.partition(":")
                    self._insert(name.strip(), desc.strip(), "seed")

    # ------------------------------------------------------------------ helpers
    def _insert(self, name: str, description: str, who: str) -> None:
        self.db.execute("INSERT INTO tasks VALUES (?, ?, 'open', NULL, '', ?, ?)",
                        (name[:NAME_MAX], description[:TEXT_MAX], who, time.time()))

    def _task(self, name: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM tasks WHERE name = ?", (name,)).fetchone()
        if row is None:
            names = [r[0] for r in self.db.execute("SELECT name FROM tasks ORDER BY name")]
            raise ToolError(f"no task {name!r}; tasks: {', '.join(names) or 'none'} "
                            "(add_task creates one)", "unknown_task")
        return row

    def _set(self, agent: Agent, name: str, action: str, status: str, owner: str | None,
             note: str = "") -> dict:
        self.db.execute("UPDATE tasks SET status = ?, owner = ?, note = ?, updated = ? "
                        "WHERE name = ?", (status, owner, note[:TEXT_MAX], time.time(), name))
        self._log(agent, action, name)
        return {"task": name, "status": status, "owner": owner}

    def _log(self, agent: Agent, action: str, task: str | None) -> None:
        self.db.execute("INSERT INTO events (ts, agent, action, task, request_id, turn) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (time.time(), agent.name, action, task, agent.request_id, agent.turn))

    # ------------------------------------------------------------------ tools
    @tool
    def list_tasks(self, agent: Agent,
                   status: Literal["open", "claimed", "done"] | None = None) -> dict:
        """List the tasks on the board: name, status, owner, description and note.

        Args:
            status: only tasks with this status (open, claimed, done); all when left out
        """
        q, args = "SELECT * FROM tasks", ()
        if status:
            q, args = q + " WHERE status = ?", (status,)
        rows = self.db.execute(q + " ORDER BY updated", args).fetchall()
        return {"tasks": [{k: r[k] for k in ("name", "status", "owner", "description", "note")}
                          for r in rows],
                "you": agent.name}

    @tool
    def add_task(self, agent: Agent, name: str, description: str = "") -> dict:
        """Add an open task to the board.

        Args:
            name: a short unique name, e.g. "greeting replies"
            description: one or two lines on what done looks like
        """
        name = " ".join(name.split())
        if not name:
            raise ToolError("the task name is empty", "bad_name")
        if self.db.execute("SELECT 1 FROM tasks WHERE name = ?", (name[:NAME_MAX],)).fetchone():
            raise ToolError(f"task {name!r} already exists", "exists")
        self._insert(name, description, agent.name)
        self._log(agent, "add", name[:NAME_MAX])
        return {"task": name[:NAME_MAX], "status": "open"}

    @tool
    def claim_task(self, agent: Agent, name: str) -> dict:
        """Claim an open task so no one else works on it. Fails if someone else holds it.

        Args:
            name: the task's name, as list_tasks shows it
        """
        t = self._task(name)
        if t["status"] == "done":
            raise ToolError(f"{name!r} is already done", "done")
        if t["status"] == "claimed" and t["owner"] != agent.name:
            raise ToolError(f"{name!r} is claimed by {t['owner']}", "held")
        return self._set(agent, name, "claim", "claimed", agent.name)

    @tool
    def finish_task(self, agent: Agent, name: str, note: str) -> dict:
        """Mark a task you claimed as done.

        Args:
            name: the task's name
            note: one line: what you did and where the result is
        """
        t = self._task(name)
        if t["owner"] != agent.name or t["status"] != "claimed":
            raise ToolError(f"claim {name!r} first (status {t['status']}, owner "
                            f"{t['owner'] or 'none'})", "not_yours")
        return self._set(agent, name, "finish", "done", agent.name, note)

    @tool
    def release_task(self, agent: Agent, name: str, note: str = "") -> dict:
        """Give a task you claimed back to the board (status open).

        Args:
            name: the task's name
            note: optional: why, and how far you got
        """
        t = self._task(name)
        if t["owner"] != agent.name or t["status"] != "claimed":
            raise ToolError(f"you do not hold {name!r}", "not_yours")
        return self._set(agent, name, "release", "open", None, note)

    # ------------------------------------------------------------------ hooks
    def on_agent_done(self, agent: Agent, data: dict) -> None:
        held = [r[0] for r in self.db.execute(
            "SELECT name FROM tasks WHERE owner = ? AND status = 'claimed'", (agent.name,))]
        for name in held:
            self._set(agent, name, "release:" + str(data.get("reason") or "removed"), "open", None)
