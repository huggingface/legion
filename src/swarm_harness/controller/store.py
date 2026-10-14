"""SQLite store for the controller.

One connection, WAL, guarded by a lock: request handlers and the reconcile thread share
it. Times are stored as epoch seconds (REAL). Events are append-only JSONL files per
agent under `<data>/events/<swarm>/<agent>.jsonl`; the store keeps a per-agent line count
so `after=` cursors are line offsets.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from swarm_harness.models import Event, SwarmSpec, TokenUsage

SCHEMA = """
CREATE TABLE IF NOT EXISTS swarms (
  name TEXT PRIMARY KEY,
  spec TEXT NOT NULL,
  applied_at REAL NOT NULL,
  removing INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS types (
  swarm TEXT NOT NULL,
  type TEXT NOT NULL,
  health TEXT NOT NULL DEFAULT 'ok',
  next_worker INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (swarm, type)
);
CREATE TABLE IF NOT EXISTS agents (
  name TEXT PRIMARY KEY,
  swarm TEXT NOT NULL,
  type TEXT NOT NULL,
  slot INTEGER NOT NULL,
  worker TEXT,
  hb_state TEXT,
  turn_id TEXT,
  idle_since REAL,
  session_id TEXT,
  resumed INTEGER,
  tokens_base TEXT NOT NULL DEFAULT '{}',
  tokens_last TEXT NOT NULL DEFAULT '{}',
  cost REAL NOT NULL DEFAULT 0,
  harness_estimate_usd REAL,
  turns INTEGER NOT NULL DEFAULT 0,
  last_sync REAL,
  msgboard_principal TEXT,
  resume_pending INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  last_heartbeat REAL
);
CREATE INDEX IF NOT EXISTS agents_by_swarm ON agents (swarm, type);
CREATE INDEX IF NOT EXISTS agents_by_worker ON agents (worker);
CREATE TABLE IF NOT EXISTS workers (
  id TEXT PRIMARY KEY,
  swarm TEXT NOT NULL,
  type TEXT NOT NULL,
  idx INTEGER NOT NULL,
  job_id TEXT,
  state TEXT NOT NULL DEFAULT 'launching',
  deadline REAL,
  launched_at REAL,
  last_heartbeat REAL,
  last_inspect REAL,
  draining INTEGER NOT NULL DEFAULT 0,
  crashes INTEGER NOT NULL DEFAULT 0,
  consecutive_crashes INTEGER NOT NULL DEFAULT 0,
  next_launch_at REAL,
  retire_at REAL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS workers_by_swarm ON workers (swarm, type, idx);
CREATE TABLE IF NOT EXISTS commands (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  worker TEXT NOT NULL,
  body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS commands_by_worker ON commands (worker, id);
CREATE TABLE IF NOT EXISTS pending_prompts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent TEXT NOT NULL,
  body TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS prompts_by_agent ON pending_prompts (agent, id);
CREATE TABLE IF NOT EXISTS event_index (
  agent TEXT PRIMARY KEY,
  swarm TEXT NOT NULL,
  lines INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS launches (ts REAL NOT NULL);
"""

Row = dict[str, Any]


def _tokens(raw: str) -> dict[str, TokenUsage]:
    return {k: TokenUsage.model_validate(v) for k, v in json.loads(raw or "{}").items()}


def _dump_tokens(tokens: dict[str, TokenUsage]) -> str:
    return json.dumps({k: v.model_dump() for k, v in tokens.items()}, sort_keys=True)


def agent_tokens(row: Row) -> dict[str, TokenUsage]:
    """Total tokens for an agent: earlier incarnations (base) plus the current one (last)."""
    total = _tokens(row["tokens_base"])
    for model, usage in _tokens(row["tokens_last"]).items():
        total[model] = total.get(model, TokenUsage()) + usage
    return total


class Store:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            self.data_dir / "controller.db", check_same_thread=False, isolation_level=None
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        # Called with ("commands", worker_id) / ("events", agent) / ("reconcile", "") after
        # writes; the API turns these into wake-ups for long-polls. Must be thread-safe.
        self.listener: Callable[[str, str], None] = lambda kind, key: None

    # ----------------------------------------------------------------- low level

    def _q(self, sql: str, args: Iterable[Any] = ()) -> list[Row]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, tuple(args)).fetchall()]

    def _one(self, sql: str, args: Iterable[Any] = ()) -> Row | None:
        rows = self._q(sql, args)
        return rows[0] if rows else None

    def _x(self, sql: str, args: Iterable[Any] = ()) -> int:
        with self._lock:
            cur = self._db.execute(sql, tuple(args))
            return cur.lastrowid or 0

    def _update(self, table: str, key: str, value: Any, fields: dict[str, Any]) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)  # keys come from code, never from input
        self._x(f"UPDATE {table} SET {cols} WHERE {key} = ?", [*fields.values(), value])

    def transaction(self):
        """Context manager for a multi-statement write."""
        store = self

        class _Tx:
            def __enter__(self):
                store._lock.acquire()
                store._db.execute("BEGIN IMMEDIATE")

            def __exit__(self, exc_type, exc, tb):
                try:
                    store._db.execute("ROLLBACK" if exc_type else "COMMIT")
                finally:
                    store._lock.release()

        return _Tx()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ----------------------------------------------------------------- swarms and types

    def put_swarm(self, spec: SwarmSpec, now: float) -> None:
        with self.transaction():
            self._x(
                "INSERT INTO swarms (name, spec, applied_at, removing) VALUES (?, ?, ?, 0) "
                "ON CONFLICT(name) DO UPDATE SET spec = excluded.spec, "
                "applied_at = excluded.applied_at, removing = 0",
                [spec.swarm, spec.model_dump_json(), now],
            )
            for type_name in spec.types:
                self._x(
                    "INSERT INTO types (swarm, type) VALUES (?, ?) "
                    "ON CONFLICT(swarm, type) DO UPDATE SET health = 'ok'",
                    [spec.swarm, type_name],
                )
            # Re-applying the spec clears crashlooping and its crash streaks.
            self._x("UPDATE workers SET consecutive_crashes = 0 WHERE swarm = ?", [spec.swarm])
        self.listener("reconcile", "")

    def get_swarm(self, name: str) -> Row | None:
        row = self._one("SELECT * FROM swarms WHERE name = ?", [name])
        if row:
            row["spec"] = SwarmSpec.model_validate_json(row["spec"])
        return row

    def list_swarms(self) -> list[Row]:
        rows = self._q("SELECT * FROM swarms ORDER BY name")
        for row in rows:
            row["spec"] = SwarmSpec.model_validate_json(row["spec"])
        return rows

    def set_removing(self, name: str) -> None:
        self._x("UPDATE swarms SET removing = 1 WHERE name = ?", [name])
        self.listener("reconcile", "")

    def delete_swarm(self, name: str) -> None:
        with self.transaction():
            self._x("DELETE FROM swarms WHERE name = ?", [name])
            self._x("DELETE FROM types WHERE swarm = ?", [name])

    def type_names(self, swarm: str) -> list[str]:
        """Types with any state in the store (spec'd or leftover from a removed type)."""
        rows = self._q(
            "SELECT type FROM types WHERE swarm = ? UNION SELECT type FROM agents WHERE swarm = ? "
            "UNION SELECT type FROM workers WHERE swarm = ?",
            [swarm, swarm, swarm],
        )
        return sorted(r["type"] for r in rows)

    def type_health(self, swarm: str, type_name: str) -> str:
        row = self._one("SELECT health FROM types WHERE swarm = ? AND type = ?", [swarm, type_name])
        return row["health"] if row else "ok"

    def set_type_health(self, swarm: str, type_name: str, health: str) -> None:
        self._x(
            "INSERT INTO types (swarm, type, health) VALUES (?, ?, ?) "
            "ON CONFLICT(swarm, type) DO UPDATE SET health = excluded.health",
            [swarm, type_name, health],
        )

    def next_worker_index(self, swarm: str, type_name: str) -> int:
        with self.transaction():
            self._x("INSERT OR IGNORE INTO types (swarm, type) VALUES (?, ?)", [swarm, type_name])
            row = self._one(
                "SELECT next_worker FROM types WHERE swarm = ? AND type = ?", [swarm, type_name]
            )
            idx = row["next_worker"]
            self._x(
                "UPDATE types SET next_worker = ? WHERE swarm = ? AND type = ?",
                [idx + 1, swarm, type_name],
            )
        return idx

    def bump_worker_index(self, swarm: str, type_name: str, at_least: int) -> None:
        self._x("INSERT OR IGNORE INTO types (swarm, type) VALUES (?, ?)", [swarm, type_name])
        self._x(
            "UPDATE types SET next_worker = MAX(next_worker, ?) WHERE swarm = ? AND type = ?",
            [at_least, swarm, type_name],
        )

    def type_owner(self, type_name: str) -> str | None:
        """The swarm owning a type name (agent names are global, so type names must be)."""
        row = self._one("SELECT swarm FROM types WHERE type = ? LIMIT 1", [type_name])
        return row["swarm"] if row else None

    # ----------------------------------------------------------------- agents

    def get_agent(self, name: str) -> Row | None:
        return self._one("SELECT * FROM agents WHERE name = ?", [name])

    def list_agents(
        self, swarm: str | None = None, type_name: str | None = None, worker: str | None = None
    ) -> list[Row]:
        where, args = [], []
        for col, val in (("swarm", swarm), ("type", type_name), ("worker", worker)):
            if val is not None:
                where.append(f"{col} = ?")
                args.append(val)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        return self._q(f"SELECT * FROM agents {clause} ORDER BY swarm, type, slot", args)

    def insert_agent(self, name: str, swarm: str, type_name: str, slot: int, now: float) -> None:
        self._x(
            "INSERT OR IGNORE INTO agents (name, swarm, type, slot, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [name, swarm, type_name, slot, now],
        )

    def update_agent(self, name: str, **fields: Any) -> None:
        self._update("agents", "name", name, fields)

    def delete_agent(self, name: str) -> None:
        with self.transaction():
            self._x("DELETE FROM agents WHERE name = ?", [name])
            self._x("DELETE FROM pending_prompts WHERE agent = ?", [name])

    def record_agent_heartbeat(
        self, name: str, fields: dict[str, Any], tokens: dict[str, TokenUsage], cost_fn
    ) -> None:
        """Store one AgentHeartbeat. Tokens are cumulative per worker incarnation; if a model's
        counters go backwards the harness restarted counting, so fold the old values into base."""
        with self.transaction():
            row = self.get_agent(name)
            if row is None:
                return
            base, last = _tokens(row["tokens_base"]), _tokens(row["tokens_last"])
            for model, old in last.items():
                new = tokens.get(model, TokenUsage())
                if any(getattr(new, f) < getattr(old, f) for f in TokenUsage.model_fields):
                    base[model] = base.get(model, TokenUsage()) + old
            fields["tokens_base"] = _dump_tokens(base)
            fields["tokens_last"] = _dump_tokens(tokens)
            total = dict(base)
            for model, usage in tokens.items():
                total[model] = total.get(model, TokenUsage()) + usage
            fields["cost"] = cost_fn(total)
            self._update("agents", "name", name, fields)

    # ----------------------------------------------------------------- workers

    def get_worker(self, worker_id: str) -> Row | None:
        return self._one("SELECT * FROM workers WHERE id = ?", [worker_id])

    def list_workers(self, swarm: str | None = None, type_name: str | None = None) -> list[Row]:
        where, args = [], []
        for col, val in (("swarm", swarm), ("type", type_name)):
            if val is not None:
                where.append(f"{col} = ?")
                args.append(val)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        return self._q(f"SELECT * FROM workers {clause} ORDER BY swarm, type, idx", args)

    def insert_worker(self, worker_id: str, swarm: str, type_name: str, idx: int, now: float):
        self._x(
            "INSERT OR IGNORE INTO workers (id, swarm, type, idx, created_at) VALUES (?, ?, ?, ?, ?)",
            [worker_id, swarm, type_name, idx, now],
        )

    def update_worker(self, worker_id: str, **fields: Any) -> None:
        self._update("workers", "id", worker_id, fields)

    def delete_worker(self, worker_id: str) -> None:
        with self.transaction():
            self._x("DELETE FROM workers WHERE id = ?", [worker_id])
            self._x("DELETE FROM commands WHERE worker = ?", [worker_id])
            self._x("UPDATE agents SET worker = NULL WHERE worker = ?", [worker_id])

    # ----------------------------------------------------------------- commands

    def enqueue_command(self, worker_id: str, body: str) -> None:
        self._x("INSERT INTO commands (worker, body) VALUES (?, ?)", [worker_id, body])
        self.listener("commands", worker_id)

    def commands_for(self, worker_id: str) -> list[Row]:
        return self._q("SELECT id, body FROM commands WHERE worker = ? ORDER BY id", [worker_id])

    def clear_commands(self, worker_id: str) -> None:
        self._x("DELETE FROM commands WHERE worker = ?", [worker_id])

    def delete_commands(self, ids: list[int]) -> None:
        if ids:
            marks = ",".join("?" * len(ids))
            self._x(f"DELETE FROM commands WHERE id IN ({marks})", ids)

    # ----------------------------------------------------------------- pending prompts

    def add_prompt(self, agent: str, body: str, now: float) -> None:
        self._x(
            "INSERT INTO pending_prompts (agent, body, created_at) VALUES (?, ?, ?)",
            [agent, body, now],
        )
        row = self.get_agent(agent)
        if row and row["worker"]:
            self.listener("commands", row["worker"])

    def prompts_for_worker(self, worker_id: str) -> list[Row]:
        return self._q(
            "SELECT p.id, p.body FROM pending_prompts p JOIN agents a ON a.name = p.agent "
            "WHERE a.worker = ? ORDER BY p.id",
            [worker_id],
        )

    def delete_prompts(self, ids: list[int]) -> None:
        if ids:
            marks = ",".join("?" * len(ids))
            self._x(f"DELETE FROM pending_prompts WHERE id IN ({marks})", ids)

    def pending_counts(self, swarm: str | None = None) -> dict[str, int]:
        sql = "SELECT p.agent, COUNT(*) AS n FROM pending_prompts p"
        args: list[Any] = []
        if swarm is not None:
            sql += " JOIN agents a ON a.name = p.agent WHERE a.swarm = ?"
            args.append(swarm)
        return {r["agent"]: r["n"] for r in self._q(sql + " GROUP BY p.agent", args)}

    # ----------------------------------------------------------------- launch budget

    def record_launch(self, now: float) -> None:
        self._x("INSERT INTO launches (ts) VALUES (?)", [now])

    def launches_since(self, since: float) -> int:
        self._x("DELETE FROM launches WHERE ts < ?", [since])
        return self._one("SELECT COUNT(*) AS n FROM launches WHERE ts >= ?", [since])["n"]

    # ----------------------------------------------------------------- events

    def _events_path(self, swarm: str, agent: str) -> Path:
        return self.data_dir / "events" / swarm / f"{agent}.jsonl"

    def append_events(self, swarm: str, agent: str, events: list[Event]) -> int:
        """Append events to the agent's stream; returns the new line count."""
        if not events:
            return self.event_count(agent)
        path = self._events_path(swarm, agent)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                for ev in events:
                    f.write(ev.model_dump_json() + "\n")
            self._x(
                "INSERT INTO event_index (agent, swarm, lines) VALUES (?, ?, ?) "
                "ON CONFLICT(agent) DO UPDATE SET lines = lines + excluded.lines",
                [agent, swarm, len(events)],
            )
            n = self.event_count(agent)
        self.listener("events", agent)
        return n

    def event_count(self, agent: str) -> int:
        row = self._one("SELECT lines FROM event_index WHERE agent = ?", [agent])
        return row["lines"] if row else 0

    def read_events(self, agent: str, after: int, limit: int = 1000) -> list[dict[str, Any]]:
        row = self._one("SELECT swarm, lines FROM event_index WHERE agent = ?", [agent])
        if row is None or after >= row["lines"]:
            return []
        out: list[dict[str, Any]] = []
        with self._events_path(row["swarm"], agent).open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= row["lines"] or len(out) >= limit:
                    break
                if i >= after:
                    out.append(json.loads(line))
        return out
