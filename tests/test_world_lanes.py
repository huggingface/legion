import contextlib
import threading
import time

import pytest

from flock.world.lanes import Lane, LaneBusy, Lanes


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _wait(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def test_slots_are_respected_and_configure_changes_them():
    clock = Clock()
    lanes = Lanes({"push": Lane(slots=2)}, clock=clock, wall=clock)
    with contextlib.ExitStack() as stack:
        stack.enter_context(lanes.hold("push"))
        stack.enter_context(lanes.hold("push"))
        with pytest.raises(LaneBusy) as exc, lanes.hold("push", deadline=clock()):
            pass
        assert exc.value.reason == "slots" and exc.value.retry_after_s == 60
        assert lanes.metrics()["push"]["in_flight"] == 2
        lanes["push"].configure(slots=3)
        with lanes.hold("push", deadline=clock()):
            pass
    m = lanes.metrics()["push"]
    assert m["in_flight"] == 0 and m["slots"] == 3 and m["waits_total"] == 1


def test_breaker_trips_after_k_transient_failures_and_recovers_after_quiet_minutes():
    clock = Clock()
    lanes = Lanes({"push": Lane(slots=2)}, clock=clock, wall=clock)
    for _ in range(2):
        with lanes.hold("push") as slot:
            slot.fail("transient")
        clock.t += 60
    clock.t += 15 * 60  # the first two left the 15-minute window
    with lanes.hold("push") as slot:
        slot.fail("transient")
    assert not lanes.metrics()["push"]["tripped"]
    for _ in range(2):
        with lanes.hold("push") as slot:
            slot.fail("transient")
    m = lanes.metrics()["push"]
    assert (m["tripped"], m["effective_slots"], m["slots"], m["transient"]) == (1, 1, 2, 5)
    with pytest.raises(LaneBusy) as exc, lanes.hold("push", deadline=clock()):
        pass  # tripped: 5 s between writes
    assert exc.value.reason == "gap" and exc.value.retry_after_s == 5
    clock.t += 9 * 60
    assert lanes.metrics()["push"]["tripped"]
    with lanes.hold("push") as slot:
        slot.ok()
    clock.t += 60  # ten minutes without a transient failure
    m = lanes.metrics()["push"]
    assert (m["tripped"], m["effective_slots"], m["ok"]) == (0, 2, 1)


def test_paused_lane_waits_then_raises_lane_busy():
    clock = Clock()
    paused = {"push"}
    lanes = Lanes({"push": Lane(), "comment": Lane()}, clock=clock, wall=clock,
                  paused=lambda name: name in paused)
    with lanes.hold("comment", deadline=clock()):
        pass
    with lanes.hold("push", deadline=clock(), pausable=False):
        pass
    errors = []

    def call():
        try:
            with lanes.hold("push", deadline=clock() + 30):
                errors.append(None)
        except LaneBusy as exc:
            errors.append(exc)

    t = threading.Thread(target=call)
    t.start()
    time.sleep(0.4)
    assert errors == [] and lanes.metrics()["push"]["paused"] == 1
    clock.t += 31
    t.join(2)
    assert errors[0].reason == "paused"
    paused.clear()
    lanes["push"].paused = True  # the attribute pauses too
    t = threading.Thread(target=call)
    t.start()
    time.sleep(0.3)
    lanes["push"].paused = False
    t.join(2)
    assert errors[1] is None


def test_outbox_survives_restart_retries_forever_drops_on_final_and_coalesces(tmp_path):
    clock = Clock()
    lanes = Lanes({"comment": Lane(gap_s=1.0)}, dir=tmp_path, clock=clock, wall=clock)
    box = lanes.outbox("comment")  # no drainer: entries pile up
    box.put(7, "a")
    box.put(8, "x")
    box.put(7, "b")
    assert box.depth() == (3, 0.0)
    lanes.close()

    results = {7: ["transient"] * 6 + ["ok"], 8: ["final"]}
    calls = []

    def drain(key, payloads):
        calls.append((key, payloads))
        return results[key].pop(0)

    clock.t += 5
    lanes = Lanes({"comment": Lane(gap_s=1.0)}, dir=tmp_path, clock=clock, wall=clock)
    box = lanes.outbox("comment", drain=drain, backoff=(5.0, 20.0))
    assert box.depth() == (3, 5.0)
    for _ in range(200):  # walk the fake clock through the capped backoff
        if not box.depth()[0]:
            break
        time.sleep(0.03)
        clock.t += 5
    assert box.depth()[0] == 0
    assert calls[0] == (7, ["a", "b"])  # oldest first, coalesced
    assert calls.count((7, ["a", "b"])) == 7 and (8, ["x"]) in calls
    m = lanes.metrics()["comment"]
    assert (m["ok"], m["transient"], m["final"], m["outbox_depth"]) == (1, 6, 1, 0)
    lanes.close()
    restarted = Lanes({"comment": Lane()}, dir=tmp_path, clock=clock, wall=clock)
    assert restarted.outbox("comment").depth()[0] == 0
    assert (tmp_path / "outbox-comment.jsonl").read_text() == ""


def test_outbox_put_while_draining_keeps_the_new_payload(tmp_path):
    lanes = Lanes({"comment": Lane()}, dir=tmp_path)
    started, go, calls = threading.Event(), threading.Event(), []

    def drain(key, payloads):
        calls.append(payloads)
        started.set()
        go.wait(2)
        return "ok"

    box = lanes.outbox("comment", drain=drain)
    box.put(1, "first")
    assert started.wait(2)
    box.put(1, "second")
    go.set()
    assert _wait(lambda: box.depth()[0] == 0)
    assert calls == [["first"], ["second"]]
    lanes.close()


def test_breaker_group_trips_all_its_lanes():
    clock = Clock()
    lanes = Lanes({"push": Lane(slots=2), "merge": Lane(slots=2), "comment": Lane(slots=2)},
                  clock=clock, wall=clock, breaker_groups=[("push", "merge")])
    for name in ("push", "merge", "push"):
        with lanes.hold(name) as slot:
            slot.fail("transient")
    m = lanes.metrics()
    assert [m[n]["tripped"] for n in ("push", "merge", "comment")] == [1, 1, 0]
    assert m["merge"]["effective_slots"] == 1 and m["comment"]["effective_slots"] == 2
    clock.t += 600
    assert lanes.metrics()["merge"]["tripped"] == 0


def test_outbox_load_skips_a_truncated_last_line(tmp_path):
    lanes = Lanes({"comment": Lane()}, dir=tmp_path)
    lanes.outbox("comment").put(1, "kept")
    with open(tmp_path / "outbox-comment.jsonl", "a") as out:
        out.write('{"put": 2, "payl')  # cut by a crash
    restarted = Lanes({"comment": Lane()}, dir=tmp_path)
    box = restarted.outbox("comment")
    assert {k: e["payloads"] for k, e in box.entries.items()} == {1: ["kept"]}
    box.put(3, "after")  # the rewritten file has no cut line left
    again = Lanes({"comment": Lane()}, dir=tmp_path).outbox("comment")
    assert {k: e["payloads"] for k, e in again.entries.items()} == {1: ["kept"], 3: ["after"]}


def test_drainer_survives_bad_results_and_batches_respect_max_chars(tmp_path):
    lanes = Lanes({"comment": Lane()}, dir=tmp_path)
    box = lanes.outbox("comment", max_chars=10)
    for text in ("aaaaaa", "bbbbbb", "cc"):
        box.put(1, text)
    lanes.close()
    calls, results = [], ["weird", "ok", "split", "ok", "ok"]

    def drain(key, payloads):
        calls.append(payloads)
        return results.pop(0)

    lanes = Lanes({"comment": Lane()}, dir=tmp_path)
    box = lanes.outbox("comment", drain=drain, max_chars=10)
    assert _wait(lambda: box.depth()[0] == 0)
    assert box.thread.is_alive()
    # max_chars keeps "bbbbbb" out of the first write; "split" sends the rest one by one
    assert calls == [["aaaaaa"], ["aaaaaa"], ["bbbbbb", "cc"], ["bbbbbb"], ["cc"]]
    lanes.close()
