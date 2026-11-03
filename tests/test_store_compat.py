"""A controller.db from before the step-3 deletions (grants, policies, notices, broadcasts)
still opens: the leftover table and rows are never read, and the stored spec's removed keys
are ignored."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from flock.controller.store import DB_NAME, Store

FIXTURE = Path(__file__).parent / "fixtures" / "stored_specs" / "run4.json"

# main @ 8949085: the grants table, dropped from the schema since
GRANTS = """CREATE TABLE IF NOT EXISTS grants (
  swarm TEXT NOT NULL, target TEXT NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL,
  updated_at REAL NOT NULL, PRIMARY KEY (swarm, target))"""


def test_old_db_with_removed_feature_rows_opens(tmp_path):
    raw = json.loads(FIXTURE.read_text())
    raw.update(policies={"mode": "enforce", "rules": [], "sinks": ["file:alerts.log"]},
               report={"every": "1h", "sinks": []})
    name = raw["swarm"]
    Store(tmp_path).close()  # today's schema, then main's extra table and rows on top
    db = sqlite3.connect(tmp_path / DB_NAME)
    db.execute(GRANTS)
    db.execute("INSERT INTO grants VALUES (?, 'agent', 'type', ?, 0)",
               [name, json.dumps({"bash": False})])
    db.execute("INSERT INTO swarms (name, spec, applied_at) VALUES (?, ?, 0)",
               [name, json.dumps(raw)])
    for key, body in [("notice:*", {"text": "final round", "since": 0}),
                      ("broadcast:b-1", {"id": "b-1", "ts": 0, "agents": [], "started": [],
                                         "finished": []}),
                      ("policy_mutes", {"worker-mem": None})]:
        db.execute("INSERT INTO controls (swarm, key, body) VALUES (?, ?, ?)",
                   [name, key, json.dumps(body)])
    db.commit()
    db.close()

    store = Store(tmp_path)
    row = store.get_swarm(name)
    assert row is not None and set(row["spec"].types) == set(raw["types"])
    assert "policies" not in row["spec"].model_dump()
    store.delete_swarm(name)  # removing it works with the leftover rows in place
    assert store.get_swarm(name) is None
    store.close()
