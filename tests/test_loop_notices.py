"""`loop.notices` (docs/loop.md): the controller keeps each agent's board notifications
(`comms_mention`, `comms_thread_reply` from its worker) and names them in one line after the
agent's next round prompt, then forgets them. Controller only: workers never see the key."""

from __future__ import annotations

import json

import httpx
import pytest

from flock.controller.api import create_app
from flock.controller.loop import notice_line
from flock.models import LoopMeta, SwarmSpec, worker_type_spec
from flock.worker.loop import render

from .conftest import TOKEN, make_spec
from .test_api import W0, hb_body
from .test_loop_controller import A0, A1, Sim, prompts

HEAD = "New for you since your last round: "
CTRL = {"prompt": "round {round} ({why})", "every": "1h", "on": ["comms.mention"],
        "notices": True}  # a duration `every`: the controller sends every round prompt
WORKER = {"prompt": "round {round}", "on": ["comms.mention", "comms.thread"],
          "notices": True}  # `turn_end`: the worker re-sends the loop_start text each round


def mention(mid: int, thread: int, who: str) -> tuple[str, dict]:
    return ("comms_mention", {"msg_id": mid, "thread": thread, "from": who})


def reply(mid: int, thread: int, who: str) -> tuple[str, dict]:
    return ("comms_thread_reply", {"msg_id": mid, "thread": thread, "from": who})


def line(text: str) -> str | None:
    tail = text.rsplit("\n\n", 1)
    return tail[1] if len(tail) == 2 and tail[1].startswith(HEAD) else None


@pytest.fixture
async def sim(store, clock, tick):
    app = create_app(store, TOKEN, clock=clock, loop_interval=0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        s = Sim(store, clock, tick, c)
        s.loops = app.state.loops
        yield s


async def started(sim, loop: dict, **kw) -> dict[str, dict]:
    """Applied, first round out and done by both agents (controller loops)."""
    await sim.apply(loop, **kw)
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for a in (A0, A1):
        await sim.turn(a, got[a])
    return got


# ------------------------------------------------------------------ the line
def test_line_groups_by_thread_humans_first_and_caps():
    n = [{"msg_id": 1, "thread": 19455, "from": "tan-jackal", "kind": "reply", "human": False},
         {"msg_id": 2, "thread": 19455, "from": "ochre-spoon", "kind": "reply", "human": False},
         {"msg_id": 3, "thread": 19820, "from": "cmpatino", "kind": "mention", "human": True}]
    assert notice_line(n) == (HEAD + "@cmpatino mentioned you in thread 19820; 2 replies in your "
                              "thread 19455 (tan-jackal, ochre-spoon). Read them if useful.")
    many = [{"msg_id": i, "thread": 100 + i % 8, "from": f"a{i % 5}{{round}}", "kind": "reply",
             "human": False} for i in range(40)]
    many.append({"msg_id": 99, "thread": 7, "from": "amber-otter", "kind": "mention", "human": False})
    out = notice_line(many)
    assert out.startswith(HEAD + "amber-otter mentioned you in thread 7; 5 replies in your thread 100 (")
    assert out.count("replies in your thread") == 4  # 5 groups in all, the mention first
    assert ", +2 more)" in out  # 5 names in thread 100, 3 shown
    assert "+20 more in threads 104, 105, 106, 107." in out
    assert "{" not in out  # names never carry a placeholder into the worker's renderer


# ------------------------------------------------------------------ controller-side rounds
async def test_notices_accumulate_render_once_then_clear(sim):
    await started(sim, CTRL)
    await sim.events(A0, mention(11, 19820, "cmpatino"), reply(12, 19455, A1),
                     reply(12, 19455, A1), reply(13, 19455, "ochre-spoon"))
    sim.tick(2.1)  # the mention wakes A0; its prompt names the four (one duplicate dropped)
    got = prompts(await sim.commands())
    assert list(got) == [A0]
    assert got[A0]["text"].startswith("round {round} ({why})\n\n")
    assert line(got[A0]["text"]) == (
        HEAD + "@cmpatino mentioned you in thread 19820; 2 replies in your thread 19455 "
        f"(@ochre-spoon, {A1}). Read them if useful.")  # not an agent of ours: a person
    assert sim.store.get_loop_agent(A0)["notices"] == []
    await sim.turn(A0, got[A0])
    sim.clock.advance(3600)
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    assert set(got) == {A0, A1} and line(got[A0]["text"]) is None  # said once, then gone
    assert got[A0]["text"] == got[A1]["text"] == "round {round} ({why})"


async def test_a_notice_waits_for_the_next_round_of_any_kind(sim):
    """A reply (not in `on:`, so no wake) rides in the next `every` round; one that came while
    the prompt was out waits for the round after."""
    await started(sim, CTRL)
    await sim.events(A1, reply(5, 40, A0))
    sim.tick(3)
    assert await sim.commands() == []  # no wake for a thread reply here
    sim.clock.advance(3600)
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    assert line(got[A1]["text"]) == HEAD + f"1 reply in your thread 40 ({A0}). Read them if useful."
    await sim.events(A1, reply(6, 40, A0))  # in its turn: next round
    await sim.turn(A1, got[A1])
    await sim.turn(A0, got[A0])
    sim.clock.advance(3600)
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    assert line(got[A1]["text"]) == HEAD + f"1 reply in your thread 40 ({A0}). Read them if useful."
    assert line(got[A0]["text"]) is None


async def test_off_by_default(sim):
    await started(sim, {**CTRL, "notices": False})
    spec = sim.store.get_swarm("demo")["spec"]
    assert "notices" not in spec.model_dump_json()  # the dump (and its hashes) as before
    await sim.events(A0, mention(11, 19820, "cmpatino"))
    sim.tick(2.1)
    got = prompts(await sim.commands())
    assert got[A0]["text"] == "round {round} ({why})"  # the wake as it was
    assert "notices" not in sim.store.get_loop_agent(A0)


async def test_notices_survive_a_controller_restart(sim, store, clock):
    await started(sim, CTRL)
    await sim.events(A1, reply(5, 40, "cmpatino"))
    app = create_app(store, TOKEN, clock=clock, loop_interval=0)  # a new controller, same state
    sim.loops = app.state.loops
    sim.clock.advance(3600)
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    assert line(got[A1]["text"]) == HEAD + "1 reply in your thread 40 (@cmpatino). Read them if useful."


async def test_workers_never_see_the_key(sim):
    await sim.apply(CTRL)
    spec = sim.store.get_swarm("demo")["spec"]
    assert spec.types["researcher"].loop.notices is True  # kept on the controller
    assert "notices" not in worker_type_spec(spec.types["researcher"]).model_dump_json()
    r = await sim.c.post(f"/workers/{W0}/heartbeat", json=hb_body(sim.clock, W0, []))
    ts = r.json()["type_spec"]
    assert "notices" not in ts["loop"] and r.json()["spec_rev"]
    spec.types["researcher"].replicas = 3  # a third agent: its assign carries the type spec
    assert (await sim.c.put("/swarms/demo", json=spec.model_dump(mode="json"))).status_code == 200
    sim.reconcile()
    cmds = await sim.commands()
    assigns = [c for c in cmds if c["kind"] == "assign"]
    assert assigns and all("notices" not in c["spec"]["loop"] for c in assigns)


# ------------------------------------------------------------------ worker-side rounds
async def test_worker_loop_gets_the_line_for_one_round(sim):
    await sim.apply(WORKER)
    await sim.heartbeat()
    sim.tick()
    start = {c["agent"]: c for c in await sim.commands()}
    assert start[A0]["text"] == "round {round}"
    await sim.events(A0, ("loop_prompt", {"kind": "loop", "round": 1, "why": "start"}),
                     ("turn_started", {}))
    await sim.events(A0, reply(5, 40, A1), mention(6, 41, "cmpatino"))
    sim.tick(0.5)
    [c] = await sim.commands()  # the next round's text, while round 1 runs
    assert c["kind"] == "loop_start" and c["agent"] == A0 and c["run"] == start[A0]["run"]
    assert c["text"] == ("round {round}\n\n" + HEAD + "@cmpatino mentioned you in thread 41; "
                         f"1 reply in your thread 40 ({A1}). Read them if useful.")
    # the worker renders it like any loop prompt: the line goes through untouched
    meta = LoopMeta.model_validate(c["meta"])
    assert render(c["text"], meta, spend=0, last_summary=None, now=sim.clock()).endswith(
        "Read them if useful.") and render(c["text"], meta, spend=0, last_summary=None,
                                            now=sim.clock()).startswith("round 1\n\n")
    await sim.events(A0, reply(7, 40, A1))  # meanwhile: for the round after
    sim.tick(1)
    assert await sim.commands() == []  # round 1 still runs; round 2 carries the line
    sim.clock.advance(6)
    await sim.events(A0, ("turn_ended", {"status": "completed"}),
                     ("loop_prompt", {"kind": "loop", "round": 2, "why": "turn_end"}))
    sim.tick()
    [c] = await sim.commands()
    assert c["text"] == "round {round}\n\n" + HEAD + f"1 reply in your thread 40 ({A1}). Read them if useful."
    sim.clock.advance(6)
    await sim.events(A0, ("turn_started", {}), ("turn_ended", {"status": "completed"}),
                     ("loop_prompt", {"kind": "loop", "round": 3, "why": "turn_end"}))
    sim.tick()
    [c] = await sim.commands()
    assert c["kind"] == "loop_start" and c["text"] == "round {round}"  # plain again
    sim.clock.advance(6)
    await sim.events(A0, ("turn_started", {}), ("turn_ended", {"status": "completed"}),
                     ("loop_prompt", {"kind": "loop", "round": 4, "why": "turn_end"}))
    sim.tick()
    assert await sim.commands() == []


async def test_a_round_before_the_fetch_does_not_take_the_line(sim):
    """A `loop_prompt` that arrives while the armed loop_start is still queued (or just after
    it left) was not rendered from it: the line stays armed for the next one."""
    await sim.apply({**WORKER, "on": []})  # no wake prompts in the way
    await sim.heartbeat()
    sim.tick()
    await sim.commands()
    await sim.events(A0, reply(5, 40, A1))
    sim.tick(0.5)  # armed, queued, not fetched yet
    await sim.events(A0, ("loop_prompt", {"kind": "loop", "round": 1, "why": "start"}))
    sim.tick(1)
    [c] = await sim.commands()  # fetched now
    assert line(c["text"])
    sim.tick(1)
    await sim.events(A0, ("loop_prompt", {"kind": "loop", "round": 2, "why": "turn_end"}))
    sim.tick(1)  # within NOTICE_FETCH_MARGIN of the fetch: maybe emitted before it
    assert await sim.commands() == []
    sim.clock.advance(6)
    await sim.events(A0, ("loop_prompt", {"kind": "loop", "round": 3, "why": "turn_end"}))
    sim.tick()
    assert [c["text"] for c in await sim.commands()] == ["round {round}"]


@pytest.mark.parametrize("harness, wake, event", [
    ("llm", "comms.mention", mention(11, 19820, "cmpatino")),
    ("harbor", "board.mention", ("resource", {"resource": "board", "kind": "mention", "data": {
        "msg_id": 11, "thread": 19820, "from": "cmpatino", "identity_type": "user"}})),
])
async def test_llm_and_harbor_types_get_the_line(sim, harness, wake, event):
    """The line is plain text after the template, so any harness renders it. A harbor type's
    board events are `resource` events from the board resource: wake `board.mention`."""
    kw = {"harbor": {"task": {}, "config": {}}} if harness == "harbor" else {}
    got = await started(sim, {**CTRL, "on": [wake]}, harness=harness, model="fake:m", secrets=[], **kw)
    assert got[A0]["loop"]["kind"] == "loop"
    await sim.events(A0, event)
    sim.tick(2.1)
    got = prompts(await sim.commands())
    assert got[A0]["loop"]["why"] == f"wake: {wake}"
    assert line(got[A0]["text"]) == HEAD + "@cmpatino mentioned you in thread 19820. Read them if useful."
    spec = sim.store.get_swarm("demo")["spec"]
    assert spec.types["researcher"].harness.value == harness


def test_spec_accepts_and_round_trips_the_key():
    spec = make_spec(researcher={"loop": CTRL})
    again = SwarmSpec.model_validate(json.loads(spec.model_dump_json()))
    assert again.types["researcher"].loop.notices is True
    assert make_spec(researcher={"loop": {"prompt": "p"}}).types["researcher"].loop.notices is False
