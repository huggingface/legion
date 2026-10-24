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

from flock.models import Event, Grant, SwarmSpec, TokenUsage

DB_NAME = "controller.db"

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
  launch_error TEXT,
  launch_failures INTEGER NOT NULL DEFAULT 0,
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
  incarnation TEXT,
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
CREATE TABLE IF NOT EXISTS agent_tokens (
  agent TEXT PRIMARY KEY,
  swarm TEXT NOT NULL,
  token_hash TEXT NOT NULL UNIQUE,
  nonce TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_bundles (
  swarm TEXT NOT NULL,
  version TEXT NOT NULL,
  body BLOB NOT NULL,
  tools TEXT NOT NULL,
  size INTEGER NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (swarm, version)
);
CREATE TABLE IF NOT EXISTS loops (
  swarm TEXT NOT NULL,
  type TEXT NOT NULL,
  body TEXT NOT NULL,
  PRIMARY KEY (swarm, type)
);
CREATE TABLE IF NOT EXISTS loop_agents (
  agent TEXT PRIMARY KEY,
  swarm TEXT NOT NULL,
  type TEXT NOT NULL,
  body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS loop_agents_by_type ON loop_agents (swarm, type);
CREATE TABLE IF NOT EXISTS swarm_flags (
  swarm TEXT PRIMARY KEY,
  body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS grants (
  swarm TEXT NOT NULL,
  target TEXT NOT NULL,
  kind TEXT NOT NULL,
  body TEXT NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY (swarm, target)
);
CREATE TABLE IF NOT EXISTS spawn_keys (
  parent TEXT NOT NULL,
  key TEXT NOT NULL,
  child TEXT NOT NULL,
  PRIMARY KEY (parent, key)
);
CREATE TABLE IF NOT EXISTS agent_names (
  name TEXT PRIMARY KEY,
  swarm TEXT NOT NULL,
  type TEXT NOT NULL,
  slot INTEGER,
  parent TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS agent_names_by_type ON agent_names (swarm, type, slot);
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
            self.data_dir / DB_NAME, check_same_thread=False, isolation_level=None
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._migrate()
        # Called with ("commands", worker_id) / ("events", agent) / ("reconcile", "") after
        # writes; the API turns these into wake-ups for long-polls. Must be thread-safe.
        self.listener: Callable[[str, str], None] = lambda kind, key: None
        # HMAC key for per-agent tokens (controller/spawn.py), derived from the controller
        # token and never stored: the table holds only hashes.
        self.agent_token_key: bytes | None = None

    def _migrate(self) -> None:
        """Additive column changes for databases created by an older version."""
        types_cols = {r["name"] for r in self._q("PRAGMA table_info(types)")}
        if "launch_error" not in types_cols:  # last worker-launch error per type
            self._x("ALTER TABLE types ADD COLUMN launch_error TEXT")
            self._x("ALTER TABLE types ADD COLUMN launch_failures INTEGER NOT NULL DEFAULT 0")

        cols = {r["name"] for r in self._q("PRAGMA table_info(workers)")}
        if "incarnation" not in cols:
            self._x("ALTER TABLE workers ADD COLUMN incarnation TEXT")
        if "last_error" not in cols:  # last launch failure or crash, shown by `flock workers`
            self._x("ALTER TABLE workers ADD COLUMN last_error TEXT")
        if "last_seen" not in cols:  # last poll / event post / heartbeat from the current job
            self._x("ALTER TABLE workers ADD COLUMN last_seen REAL")
        if "preempted" not in cols:  # the current job's last heartbeat said `preempted`
            self._x("ALTER TABLE workers ADD COLUMN preempted INTEGER NOT NULL DEFAULT 0")
        # Spawn (docs/spawn.md): lineage on agents, the never-reused spawn counter and the cost
        # of removed agents on swarms.
        cols = {r["name"] for r in self._q("PRAGMA table_info(agents)")}
        if "parent" not in cols:
            self._x("ALTER TABLE agents ADD COLUMN parent TEXT")
            self._x("ALTER TABLE agents ADD COLUMN depth INTEGER NOT NULL DEFAULT 0")
        self._x("CREATE INDEX IF NOT EXISTS agents_by_parent ON agents (parent)")
        if "removing" not in cols:  # set while a cascade removal is in flight
            self._x("ALTER TABLE agents ADD COLUMN removing INTEGER NOT NULL DEFAULT 0")
        cols = {r["name"] for r in self._q("PRAGMA table_info(swarms)")}
        if "spawned" not in cols:
            self._x("ALTER TABLE swarms ADD COLUMN spawned INTEGER NOT NULL DEFAULT 0")
            self._x("ALTER TABLE swarms ADD COLUMN removed_cost REAL NOT NULL DEFAULT 0")

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

    def changes(self) -> int:
        """Rows changed through this connection since it opened (every write goes through it)."""
        with self._lock:
            return self._db.total_changes

    def backup_to(self, path: str | Path) -> None:
        """Consistent copy of the database, WAL included (SQLite online backup API)."""
        path = Path(path)
        path.unlink(missing_ok=True)
        dst = sqlite3.connect(path)
        try:
            with self._lock:
                self._db.backup(dst)
        finally:
            dst.close()

    def repair_events(self) -> int:
        """After a restore, make each event file and its index agree: a file longer than the
        index (lines appended after the DB snapshot) is cut back, a missing or shorter one sets
        the index to what is there. Returns the number of agents fixed."""
        fixed = 0
        with self._lock:
            for row in self._q("SELECT agent, swarm, lines FROM event_index"):
                path = self._events_path(row["swarm"], row["agent"])
                lines = path.read_bytes().splitlines(keepends=True) if path.exists() else []
                lines = [ln for ln in lines if ln.endswith(b"\n")]  # drop a torn last line
                if len(lines) == row["lines"]:
                    continue
                keep = lines[: row["lines"]]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"".join(keep))
                self._x("UPDATE event_index SET lines = ? WHERE agent = ?", [len(keep), row["agent"]])
                fixed += 1
        return fixed

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
                    "ON CONFLICT(swarm, type) DO UPDATE SET health = 'ok', launch_error = NULL, "
                    "launch_failures = 0",
                    [spec.swarm, type_name],
                )
            # Re-applying the spec clears crashlooping, launch_refused and their streaks.
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
            self._x("DELETE FROM tool_bundles WHERE swarm = ?", [name])
            self._x("DELETE FROM loops WHERE swarm = ?", [name])
            self._x("DELETE FROM loop_agents WHERE swarm = ?", [name])
            self._x("DELETE FROM swarm_flags WHERE swarm = ?", [name])
            self._x("DELETE FROM grants WHERE swarm = ?", [name])
            self._x("DELETE FROM agent_names WHERE swarm = ?", [name])

    # ----------------------------------------------------------------- grants (docs/live.md)

    def put_grant(self, swarm: str, target: str, kind: str, grant: Grant, now: float) -> None:
        """Set (replace) the grant of a type or an agent; an empty grant deletes it."""
        if grant.empty():
            self._x("DELETE FROM grants WHERE swarm = ? AND target = ?", [swarm, target])
            return
        self._x("INSERT INTO grants (swarm, target, kind, body, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(swarm, target) DO UPDATE SET kind = excluded.kind, "
                "body = excluded.body, updated_at = excluded.updated_at",
                [swarm, target, kind, grant.model_dump_json(exclude_defaults=True), now])

    def list_grants(self, swarm: str) -> list[Row]:
        rows = self._q("SELECT target, kind, body, updated_at FROM grants WHERE swarm = ? "
                       "ORDER BY kind DESC, target", [swarm])
        for row in rows:
            row["grant"] = Grant.model_validate_json(row.pop("body"))
        return rows

    def agent_grants(self, swarm: str) -> tuple[dict[str, Grant], dict[str, Grant]]:
        """(type grants, agent grants) of a swarm, by target."""
        types: dict[str, Grant] = {}
        agents: dict[str, Grant] = {}
        for row in self.list_grants(swarm):
            (types if row["kind"] == "type" else agents)[row["target"]] = row["grant"]
        return types, agents

    # ----------------------------------------------------------------- tools bundles (docs/tools.md)

    TOOL_BUNDLES_KEPT = 10  # per swarm, newest first, plus the one the spec names

    def put_tool_bundle(self, swarm: str, version: str, body: bytes, tools: list[str],
                        now: float) -> None:
        """Store a bundle (idempotent) and drop the oldest beyond TOOL_BUNDLES_KEPT, never the
        version the applied spec names."""
        with self.transaction():
            self._x(
                "INSERT INTO tool_bundles (swarm, version, body, tools, size, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(swarm, version) DO UPDATE SET "
                "created_at = excluded.created_at",
                [swarm, version, body, json.dumps(tools), len(body), now],
            )
            sw = self.get_swarm(swarm)
            current = sw["spec"].tools_version if sw else None
            old = self._q("SELECT version FROM tool_bundles WHERE swarm = ? "
                          "ORDER BY created_at DESC LIMIT -1 OFFSET ?",
                          [swarm, self.TOOL_BUNDLES_KEPT])
            for row in old:
                if row["version"] != current:
                    self._x("DELETE FROM tool_bundles WHERE swarm = ? AND version = ?",
                            [swarm, row["version"]])

    def get_tool_bundle(self, swarm: str, version: str, *, body: bool = True) -> Row | None:
        cols = "*" if body else "swarm, version, tools, size, created_at"
        row = self._one(f"SELECT {cols} FROM tool_bundles WHERE swarm = ? AND version = ?",
                        [swarm, version])
        if row:
            row["tools"] = json.loads(row["tools"])
        return row

    def list_tool_bundles(self, swarm: str) -> list[Row]:
        rows = self._q("SELECT version, tools, size, created_at FROM tool_bundles WHERE swarm = ? "
                       "ORDER BY created_at DESC", [swarm])
        for row in rows:
            row["tools"] = json.loads(row["tools"])
        return rows

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

    def launch_error(self, swarm: str, type_name: str) -> tuple[str | None, int]:
        """(last launch error, consecutive transient launch failures) for a type."""
        row = self._one("SELECT launch_error, launch_failures FROM types WHERE swarm = ? AND type = ?",
                        [swarm, type_name])
        return (row["launch_error"], row["launch_failures"]) if row else (None, 0)

    def set_launch_error(self, swarm: str, type_name: str, error: str | None, failures: int) -> None:
        self._x(
            "INSERT INTO types (swarm, type, launch_error, launch_failures) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(swarm, type) DO UPDATE SET launch_error = excluded.launch_error, "
            "launch_failures = excluded.launch_failures",
            [swarm, type_name, error, failures],
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
        self,
        swarm: str | None = None,
        type_name: str | None = None,
        worker: str | None = None,
        parent: str | None = None,
    ) -> list[Row]:
        where, args = [], []
        for col, val in (
            ("swarm", swarm), ("type", type_name), ("worker", worker), ("parent", parent)
        ):
            if val is not None:
                where.append(f"{col} = ?")
                args.append(val)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        return self._q(f"SELECT * FROM agents {clause} ORDER BY swarm, type, slot", args)

    def agent_states(self, swarm: str, types: Iterable[str]) -> list[Row]:
        """Just the columns `agent_state` reads, for the agents of `types` (admission runs this
        on every poll that carries prompts, so it skips the token columns)."""
        types = sorted(types)
        if not types:
            return []
        marks = ",".join("?" * len(types))
        return self._q(
            "SELECT name, type, worker, hb_state, last_heartbeat FROM agents "
            f"WHERE swarm = ? AND type IN ({marks})",
            [swarm, *types],
        )

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
            self.delete_agent_rows(name)

    def delete_agent_rows(self, name: str) -> None:
        """delete_agent's statements, for a caller already inside a transaction."""
        # Removed agents keep counting against limits.budget_usd.
        self._x(
            "UPDATE swarms SET removed_cost = removed_cost + "
            "(SELECT cost FROM agents WHERE name = ?) "
            "WHERE name = (SELECT swarm FROM agents WHERE name = ?)",
            [name, name],
        )
        self._x("DELETE FROM agents WHERE name = ?", [name])
        self._x("DELETE FROM pending_prompts WHERE agent = ?", [name])
        self._x("DELETE FROM agent_tokens WHERE agent = ?", [name])  # revoke
        self._x("DELETE FROM spawn_keys WHERE parent = ?", [name])
        self._x("DELETE FROM loop_agents WHERE agent = ?", [name])
        self._x("DELETE FROM grants WHERE kind = 'agent' AND target = ?", [name])

    def mark_removing(self, name: str) -> None:
        """First phase of a cascade removal: refuse new children, revoke the token."""
        self._x("UPDATE agents SET removing = 1 WHERE name = ?", [name])
        self._x("DELETE FROM agent_tokens WHERE agent = ?", [name])

    # ----------------------------------------------------------------- spawn

    def insert_child(
        self, name: str, swarm: str, type_name: str, n: int, parent: str, depth: int, now: float
    ) -> None:
        self._x(
            "INSERT INTO agents (name, swarm, type, slot, parent, depth, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [name, swarm, type_name, n, parent, depth, now],
        )

    # ----------------------------------------------------------------- names (naming: words)

    def reserve_name(self, name: str, swarm: str, type_name: str, slot: int | None,
                     parent: str | None, now: float) -> bool:
        """Reserve `name` for good (while the swarm exists): False if it is taken by a
        reservation or an agent anywhere on this controller."""
        with self._lock:
            if self.get_agent(name) is not None:
                return False
            cur = self._db.execute(
                "INSERT OR IGNORE INTO agent_names (name, swarm, type, slot, parent, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)", (name, swarm, type_name, slot, parent, now))
            return cur.rowcount == 1

    def reserved_slot_names(self, swarm: str, type_name: str) -> dict[int, str]:
        """Replica slot -> its reserved name (a recreated slot keeps it)."""
        return {r["slot"]: r["name"] for r in self._q(
            "SELECT slot, name FROM agent_names WHERE swarm = ? AND type = ? AND parent IS NULL "
            "AND slot IS NOT NULL", [swarm, type_name])}

    def reserved_name(self, name: str) -> Row | None:
        """A words name's reservation (it outlives the agent): type, slot, parent."""
        return self._one("SELECT * FROM agent_names WHERE name = ?", [name])

    def reserved_names_count(self, swarm: str) -> int:
        return self._one("SELECT COUNT(*) AS n FROM agent_names WHERE swarm = ?", [swarm])["n"]

    def next_spawn_number(self, swarm: str) -> int:
        """The swarm's spawn counter, incremented; never reused while the swarm exists."""
        with self._lock:
            self._x("UPDATE swarms SET spawned = spawned + 1 WHERE name = ?", [swarm])
            return self._one("SELECT spawned FROM swarms WHERE name = ?", [swarm])["spawned"]

    def spawned_count(self, swarm: str) -> int:
        row = self._one("SELECT spawned FROM swarms WHERE name = ?", [swarm])
        return row["spawned"] if row else 0

    def children_by_parent(self, swarm: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in self._q(
            "SELECT parent, name FROM agents WHERE swarm = ? AND parent IS NOT NULL ORDER BY name",
            [swarm],
        ):
            out.setdefault(r["parent"], []).append(r["name"])
        return out

    def swarm_cost(self, swarm: str) -> float:
        """Cost of every agent the swarm has had: live rows plus removed ones."""
        row = self._one(
            "SELECT s.removed_cost + COALESCE((SELECT SUM(cost) FROM agents WHERE swarm = ?), 0) "
            "AS c FROM swarms s WHERE s.name = ?",
            [swarm, swarm],
        )
        return row["c"] if row else 0.0

    def count_agents(self, swarm: str, type_name: str | None = None) -> int:
        sql, args = "SELECT COUNT(*) AS n FROM agents WHERE swarm = ?", [swarm]
        if type_name is not None:
            sql += " AND type = ?"
            args.append(type_name)
        return self._one(sql, args)["n"]

    def spawn_key_child(self, parent: str, key: str) -> str | None:
        row = self._one("SELECT child FROM spawn_keys WHERE parent = ? AND key = ?", [parent, key])
        return row["child"] if row else None

    def put_spawn_key(self, parent: str, key: str, child: str) -> None:
        self._x("INSERT INTO spawn_keys (parent, key, child) VALUES (?, ?, ?)", [parent, key, child])

    # ----------------------------------------------------------------- agent tokens

    def get_agent_token(self, agent: str) -> Row | None:
        return self._one("SELECT * FROM agent_tokens WHERE agent = ?", [agent])

    def agent_for_token_hash(self, token_hash: str) -> str | None:
        row = self._one("SELECT agent FROM agent_tokens WHERE token_hash = ?", [token_hash])
        return row["agent"] if row else None

    def put_agent_token(
        self, agent: str, swarm: str, token_hash: str, nonce: str, now: float
    ) -> None:
        self._x(
            "INSERT INTO agent_tokens (agent, swarm, token_hash, nonce, created_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(agent) DO UPDATE SET "
            "token_hash = excluded.token_hash, nonce = excluded.nonce",
            [agent, swarm, token_hash, nonce, now],
        )

    def record_agent_heartbeat(
        self, name: str, fields: dict[str, Any], tokens: dict[str, TokenUsage], cost_fn
    ) -> None:
        """Store one AgentHeartbeat. Tokens are cumulative (the worker restores them from the
        bucket); if a model's counters go backwards the count restarted (a failed restore), so
        fold the old values into base. An empty map means "not known yet" (an agent still
        restoring on a relaunched worker) and leaves tokens and cost alone: folding there
        counted every restored agent twice (1k run)."""
        with self.transaction():
            row = self.get_agent(name)
            if row is None:
                return
            if not tokens and row["tokens_last"] not in ("", "{}"):
                if fields.get("harness_estimate_usd") is None:
                    fields.pop("harness_estimate_usd", None)
                self._update("agents", "name", name, fields)
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
            # Cost grows by the price of the new tokens only (totals never go backwards), so
            # a price change on re-apply applies from then on and does not reprice history.
            prev = dict(_tokens(row["tokens_base"]))
            for model, usage in last.items():
                prev[model] = prev.get(model, TokenUsage()) + usage
            delta = cost_fn(total) - cost_fn(prev)
            fields["cost"] = round((row["cost"] or 0.0) + max(0.0, delta), 6)
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

    def add_prompts(self, rows: list[tuple[str, str]], now: float) -> None:
        """Many (agent, body) prompts in one transaction; each worker is woken once."""
        if not rows:
            return
        with self.transaction():
            self._db.executemany(
                "INSERT INTO pending_prompts (agent, body, created_at) VALUES (?, ?, ?)",
                [(agent, body, now) for agent, body in rows],
            )
        names = {agent for agent, _ in rows}
        workers = {a["worker"] for a in self.list_agents() if a["name"] in names and a["worker"]}
        for w in sorted(workers):
            self.listener("commands", w)

    def prompts_for_worker(self, worker_id: str) -> list[Row]:
        return self._q(
            "SELECT p.id, p.agent, p.body FROM pending_prompts p JOIN agents a ON a.name = p.agent "
            "WHERE a.worker = ? ORDER BY p.id",
            [worker_id],
        )

    def oldest_pending(self, swarm: str, types: Iterable[str]) -> list[str]:
        """Agents of `types` with pending prompts, oldest first prompt first (prompt ids grow
        with creation): the FIFO order `max_working` admission hands out free slots in."""
        types = sorted(types)
        if not types:
            return []
        marks = ",".join("?" * len(types))
        rows = self._q(
            "SELECT p.agent, MIN(p.id) AS first FROM pending_prompts p "
            "JOIN agents a ON a.name = p.agent "
            f"WHERE a.swarm = ? AND a.type IN ({marks}) GROUP BY p.agent ORDER BY first",
            [swarm, *types],
        )
        return [r["agent"] for r in rows]

    def delete_prompts(self, ids: list[int]) -> None:
        if ids:
            marks = ",".join("?" * len(ids))
            self._x(f"DELETE FROM pending_prompts WHERE id IN ({marks})", ids)

    def drop_prompts(self, swarm: str, keep=None, type_name: str | None = None,
                     dropped: list[tuple[str, str]] | None = None) -> int:
        """Delete the swarm's pending prompts (of one type, if given), except those whose body
        `keep(body)` says to keep. Returns how many were dropped; `dropped` collects their
        (agent, body)."""
        sql = ("SELECT p.id, p.agent, p.body FROM pending_prompts p JOIN agents a ON a.name = p.agent "
               "WHERE a.swarm = ?")
        args: list[Any] = [swarm]
        if type_name is not None:
            sql += " AND a.type = ?"
            args.append(type_name)
        rows = [r for r in self._q(sql, args) if keep is None or not keep(r["body"])]
        self.delete_prompts([r["id"] for r in rows])
        if dropped is not None:
            dropped.extend((r["agent"], r["body"]) for r in rows)
        return len(rows)

    def pending_counts(self, swarm: str | None = None) -> dict[str, int]:
        sql = "SELECT p.agent, COUNT(*) AS n FROM pending_prompts p"
        args: list[Any] = []
        if swarm is not None:
            sql += " JOIN agents a ON a.name = p.agent WHERE a.swarm = ?"
            args.append(swarm)
        return {r["agent"]: r["n"] for r in self._q(sql + " GROUP BY p.agent", args)}

    # ----------------------------------------------------------------- loops (controller/loop.py)

    def get_loops(self, swarm: str) -> dict[str, dict[str, Any]]:
        rows = self._q("SELECT type, body FROM loops WHERE swarm = ?", [swarm])
        return {r["type"]: json.loads(r["body"]) for r in rows}

    def put_loop(self, swarm: str, type_name: str, body: dict[str, Any]) -> None:
        self._x(
            "INSERT INTO loops (swarm, type, body) VALUES (?, ?, ?) "
            "ON CONFLICT(swarm, type) DO UPDATE SET body = excluded.body",
            [swarm, type_name, json.dumps(body, sort_keys=True)],
        )

    def delete_loop(self, swarm: str, type_name: str) -> None:
        with self.transaction():
            self._x("DELETE FROM loops WHERE swarm = ? AND type = ?", [swarm, type_name])
            self._x("DELETE FROM loop_agents WHERE swarm = ? AND type = ?", [swarm, type_name])

    def loop_agents(self, swarm: str, type_name: str) -> dict[str, dict[str, Any]]:
        rows = self._q("SELECT agent, body FROM loop_agents WHERE swarm = ? AND type = ?",
                       [swarm, type_name])
        return {r["agent"]: json.loads(r["body"]) for r in rows}

    def get_loop_agent(self, agent: str) -> dict[str, Any] | None:
        row = self._one("SELECT body FROM loop_agents WHERE agent = ?", [agent])
        return json.loads(row["body"]) if row else None

    def put_loop_agent(self, agent: str, swarm: str, type_name: str, body: dict[str, Any]) -> None:
        self._x(
            "INSERT INTO loop_agents (agent, swarm, type, body) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(agent) DO UPDATE SET body = excluded.body",
            [agent, swarm, type_name, json.dumps(body, sort_keys=True)],
        )

    def delete_loop_agent(self, agent: str) -> None:
        self._x("DELETE FROM loop_agents WHERE agent = ?", [agent])

    def swarm_flags(self, swarm: str) -> dict[str, Any]:
        row = self._one("SELECT body FROM swarm_flags WHERE swarm = ?", [swarm])
        return json.loads(row["body"]) if row else {}

    def put_swarm_flags(self, swarm: str, body: dict[str, Any]) -> None:
        self._x(
            "INSERT INTO swarm_flags (swarm, body) VALUES (?, ?) "
            "ON CONFLICT(swarm) DO UPDATE SET body = excluded.body",
            [swarm, json.dumps(body, sort_keys=True)],
        )

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

    def event_agents(self, swarm: str) -> list[str]:
        """Every stream of the swarm with events, removed agents' included."""
        return [r["agent"] for r in self._q(
            "SELECT agent FROM event_index WHERE swarm = ? ORDER BY agent", [swarm])]

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
