"""pi session format v3 on disk round-trips with the Event records sent to the controller."""

from __future__ import annotations

import itertools
import json

from flock.models import Event, EventBatch
from flock.worker.events import Outbox, SessionLog, event_to_entry, read_session


def test_session_file_round_trip(tmp_path):
    path = tmp_path / "_sessions" / "researcher-0001.jsonl"
    log = SessionLog(path, swarm="sw", agent="researcher-0001", harness="codex", cwd="/work/agents/researcher-0001")
    sent = [
        log.custom("session_started", {"session_id": "abc", "resumed": False}),
        log.custom("turn_started", {}, turn_id="t-1"),
        log.message({"role": "user", "content": "hi"}, turn_id="t-1"),
        log.message({"role": "assistant", "content": [{"type": "text", "text": "yo"}], "model": "m"}, turn_id="t-1"),
        log.custom("turn_ended", {"status": "completed"}, turn_id="t-1"),
    ]
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    header = lines[0]
    assert header["type"] == "session" and header["version"] == 3 and header["harness"] == "codex"
    assert header["id"] == "sw/researcher-0001" and header["name"] == "researcher-0001"
    assert header["cwd"] == "/work/agents/researcher-0001"
    # parentId chain, 8-hex ids, custom entries for harness-neutral events
    assert lines[1]["parentId"] is None
    for prev, cur in itertools.pairwise(lines[1:]):
        assert cur["parentId"] == prev["id"] and len(cur["id"]) == 8
    assert lines[1]["type"] == "custom" and lines[1]["customType"] == "session_started"
    assert lines[3]["type"] == "message" and isinstance(lines[3]["message"]["timestamp"], int)
    assert lines[3]["turnId"] == "t-1" and lines[3]["timestamp"].endswith("Z")

    hdr, events = read_session(path, "researcher-0001")
    assert hdr == header
    assert events == sent
    assert [e.type for e in events] == ["session_started", "turn_started", "message", "message", "turn_ended"]
    # the wire form round-trips too, and its data is exactly the on-disk line
    batch = EventBatch(worker="w", events=sent)
    back = EventBatch.model_validate_json(batch.model_dump_json())
    assert [event_to_entry(e) for e in back.events] == lines[1:]


def test_session_log_reopens_after_restore(tmp_path):
    path = tmp_path / "a.jsonl"
    a = SessionLog(path, swarm="sw", agent="a-0001", harness="claude", cwd="/w")
    first = a.custom("sync", {})
    b = SessionLog(path, swarm="sw", agent="a-0001", harness="claude", cwd="/w")
    second = b.custom("sync", {})
    assert second.data["parentId"] == first.data["id"] and second.data["id"] != first.data["id"]
    assert sum(1 for x in path.read_text().splitlines() if '"type": "session"' in x) == 1


def test_outbox_put_back_keeps_order():
    from datetime import UTC, datetime

    ob = Outbox(maxlen=3)
    evs = [Event(type="sync", ts=datetime.now(UTC), agent="a", data={"n": i}) for i in range(4)]
    for e in evs:
        ob.add(e)
    assert len(ob) == 3
    got = ob.take(2)
    ob.put_back(got)
    assert [e.data["n"] for e in ob.take()] == [1, 2, 3]
