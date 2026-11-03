"""experiments/run3: the wave schedule and runner, the rendered spec variants, a jobs-mode dry
run (the controller with FakeJobsClient, the world behind world.url), and the single-job
bootstrap end to end with the fakes: 3 waves (0 -> 20 -> 40, the third capped), every new agent
gets the newcomer prompt first, promotion from the start, interviews booked without a quota,
the stop marker, and the bundle in the bucket."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from flock import cli, ops
from flock.cli import load_spec
from flock.controller.api import create_app as controller_app
from flock.controller.jobs import FakeJobsClient
from flock.controller.reconcile import Settings, reconcile

from .test_chatbot_experiment import fakes, script
from .test_local_run_e2e import _env
from .test_world import ROOT

__all__ = ["fakes"]

EXP = ROOT / "experiments" / "run3"
TOKEN = "ctl-t0ken"
MAIN = ("import sys\nfrom . import words\n\nfor line in sys.stdin:\n"
        "    print(words.reply(line.strip()), flush=True)\n")
WORDS = ("def reply(line):\n    return ('Hi Priya! I am Bo, nice to meet you. I am good. Japan, the"
         " food, the guitar, the dog Biscuit, pasta, a nurse? Bye: ' + line)\n")


def _mod(name: str):
    s = importlib.util.spec_from_file_location(f"run3_{name}", EXP / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    sys.modules[f"run3_{name}"] = m  # a dataclass needs its module registered
    sys.path.insert(0, str(EXP))
    try:
        s.loader.exec_module(m)
    finally:
        sys.path.remove(str(EXP))
    return m


@pytest.fixture
def waves():
    return _mod("waves")


def test_wave_schedule(waves):
    s = waves.load(EXP / "waves.yaml")
    assert (s.cap, s.type, s.think_type, s.think_share) == (1000, "agent", "agent-think", 0)
    assert s.events[:5] == [(0, 20), (300, 40), (600, 60), (900, 80), (1200, 100)]
    assert s.events[5] == (1200 + 3600, 120) and s.events[-1] == (1200 + 10 * 3600, 300)
    assert len(s.events) == 15
    assert [s.target(t) for t in (0, 299, 300, 1199, 1200, 4799, 4800, 10 ** 6)] == [
        20, 20, 40, 80, 100, 100, 120, 300]
    assert s.next(1200) == (4800, 120) and s.next(10 ** 6) is None
    assert s.split(300) == {"agent": 300, "agent-think": 0}
    assert s.staggers == [10] + [0] * 14 and (s.stagger(0), s.stagger(301)) == (10, 0)
    assert [waves.seconds(x) for x in (0, "90", "30s", "5m", "1h20m", "2d")] == [
        0, 90, 30, 300, 4800, 172800]
    t = waves.parse({"cap": 50, "think_type": "agent-think", "think_share": 0.25,
                     "waves": [{"at": "0", "add": 20}, {"every": "1m", "add": 20, "until": 300}]})
    assert t.events == [(0, 20), (60, 40), (120, 50)]  # capped at 50, then over
    assert t.staggers == [10, 0, 0]  # wave 1 defaults to 10 s, the others to 0
    assert waves.parse({"stagger_s": 0, "waves": [{"at": 0, "add": 1},
                                                  {"at": 5, "add": 1, "stagger_s": 3}]}
                       ).staggers == [0, 3]
    assert t.split(50) == {"agent": 38, "agent-think": 12}  # round half to even
    assert waves.parse({"waves": [{"at": "2m", "add": 5},
                                  {"every": "1h", "add": 5, "until": 12, "from": "1h"}]}
                       ).events == [(120, 5), (7200, 10), (10800, 12)]
    for bad, why in (({"waves": [{"at": "5m", "add": 1}, {"at": "1m", "add": 1}]}, "time order"),
                     ({"waves": [{"at": "5x", "add": 1}]}, "bad duration"),
                     ({"waves": [{"at": 0, "add": 0}]}, "add must be"),
                     ({"think_share": 2, "waves": []}, "think_share"),
                     ({"waves": [{"at": 0, "add": 1, "stagger_s": -1}]}, "stagger_s")):
        with pytest.raises(ValueError, match=why):
            waves.parse(bad)


def _render(tmp: Path, variant: str, *extra: str, **envs: str) -> Path:
    keep = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "MODEL_",
                                                                     "CHATBOT_", "RUN3_"))}
    out = tmp / f"{variant}{''.join(extra).replace('--compute', '.')}.local.yaml"
    r = subprocess.run([sys.executable, str(EXP / "render.py"), "--variant", variant, *extra,
                        "--out", str(out)], env={**keep, **envs}, capture_output=True,
                       text=True, timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    dst = EXP / out.name  # next to spec.yaml: the prompt files resolve
    shutil.copyfile(out, dst)
    return dst


def test_render_variants(tmp_path):
    files = []
    try:
        files.append(_render(tmp_path, "selfhosted", OPENAI_BASE_URL="http://gpu.internal:8000/v1",
                             MODEL_WORKER="deepseek-ai/DeepSeek-V4.1-Flash",
                             CHATBOT_CHANNEL="run3-test", CHATBOT_RUN="r1"))
        s = load_spec(str(files[-1]))
        assert (s.swarm, s.compute.kind, s.comms.channel) == ("run3-r1", "local", "run3-test")
        p = s.providers["vllm"]
        assert (p.base_url, p.api_key_env, p.self_hosted) == ("http://gpu.internal:8000/v1",
                                                             "OPENAI_API_KEY", True)
        assert p.pricing == {"*": [0, 0, 0]} and s.limits.budget_usd is None
        ids = {k: m.model for k, m in s.models.items()}
        assert set(ids.values()) == {"vllm:deepseek-ai/DeepSeek-V4.1-Flash"}  # one served id
        assert s.models["worker"].extra["chat_template_kwargs"] == {"thinking": False}
        assert s.models["worker-think"].extra["chat_template_kwargs"] == {"thinking": True}
        assert s.models["worker-think"].extra["max_tokens"] >= 4096
        assert sorted(s.types) == ["agent", "agent-pi", "agent-think", "interviewer"]
        # agent-pi: the Pi harness on the same served model, thinking left on (Pi sends no
        # `extra`), the same tools, comms, loop and round prompt; its first prompt is a Pi note
        # followed by exactly agent's first prompt (kept in sync by hand)
        pi, ag = s.types["agent-pi"], s.types["agent"]
        assert (pi.harness.value, pi.replicas, pi.agents_per_worker, pi.max_turns) == (
            "pi", 0, 25, 20)
        assert pi.resolved_model.model_id == ag.resolved_model.model_id
        assert pi.effective_reasoning == "high" and not pi.resolved_model.extra
        assert (pi.tools, pi.comms, pi.system_prompt) == (ag.tools, ag.comms, ag.system_prompt)
        assert pi.loop.prompt == ag.loop.prompt and pi.loop.final == ag.loop.final
        assert (pi.loop.every, pi.loop.on, pi.loop.wake, pi.loop.while_) == (
            ag.loop.every, ag.loop.on, ag.loop.wake, ag.loop.while_)
        assert s.spawn["agent-pi"] == s.spawn["agent"]
        assert ops.extra_warnings(s) == []  # agent-pi's alias has no `extra` Pi would drop
        assert s.naming == "words" and s.types["interviewer"].name_prefix == "iv-"
        for name in ("agent", "agent-think"):
            t = s.types[name]
            assert (t.replicas, t.agents_per_worker, t.tool_choice, t.bash) == (0, 25, "required",
                                                                               False)
            assert t.max_turns == 20 and "at most 18 calls" in t.loop.prompt
            assert "exactly as `run_bot` returned it" in t.loop.prompt
            lp = t.loop
            assert (lp.every, lp.while_.until, lp.wake.min_gap, lp.wake.max) == (
                "turn_end", "48h", "10s", 20)
            assert "search" in lp.prompt and len(lp.prompt) < 1500  # short, action-oriented
        iv = s.types["interviewer"]
        assert iv.ephemeral and iv.model == "interviewer" and iv.workers.max_extra == 40
        assert s.spawn["agent"].types == ["interviewer"] and s.spawn["agent"].max == 1
        assert "protocols_text" not in str(s.comms.rules)
        files.append(_render(tmp_path, "jobs", MODEL_WORKER="m", MODEL_INTERVIEWER="m-iv",
                             RUN3_WORLD_URL="https://w.example.invalid"))
        j = load_spec(str(files[-1]))
        assert (j.compute.kind, j.world.url, j.world.token_env) == (
            "hf-jobs", "https://w.example.invalid", "CHATBOT_WORLD_TOKEN")
        assert j.models["interviewer"].model == "vllm:m-iv" and j.world.command is None
        assert j.types["agent"].secrets == ["OPENAI_API_KEY"]
        files.append(_render(tmp_path, "smoke", OPENAI_BASE_URL="http://gpu.internal:8000/v1",
                             MODEL_WORKER="m", CHATBOT_RUN="smoke",
                             OPENAI_BASE_URLS="http://g1:8000/v1, http://g2:8000/v1"))
        sm = load_spec(str(files[-1]))
        assert sm.providers["vllm"].endpoints == ["http://g1:8000/v1", "http://g2:8000/v1"]
        assert sm.swarm == "run3-smoke" and sm.limits.budget_usd is None
        for name in ("agent", "agent-think"):
            lp = sm.types[name].loop
            assert (lp.every, lp.while_.until, lp.while_.max_rounds, lp.while_.agent_usd) == (
                "turn_end", "30m", 120, None)
        assert sm.providers["vllm"].pricing == {"*": [0, 0, 0]}
        # the cluster smoke: the smoke's 30-min loop and round cap, on compute slurm
        files.append(_render(tmp_path, "smoke", "--compute", "slurm", MODEL_WORKER="m",
                             RUN3_BUCKET="hf://buckets/ns/run3"))
        sl = load_spec(str(files[-1]))
        assert sl.compute.kind == "slurm" and sl.bucket == "hf://buckets/ns/run3"
        assert sl.world.url == "http://${WORLD_HOST}:7860" and sl.world.command is None
        assert (sl.compute.types["interviewer"].cpus, sl.compute.types["interviewer"].mem) == (
            2, "3800M")
        assert (sl.compute.types["agent"].cpus, sl.compute.types["agent"].mem) == (4, "7G")
        assert (sl.compute.types["agent-pi"].cpus, sl.compute.types["agent-pi"].mem) == (4, "7G")
        assert sl.types["agent-pi"].secrets == ["OPENAI_API_KEY"]  # Pi needs the key var set
        assert sl.types["interviewer"].workers.max_extra >= 1
        for name in ("agent", "agent-think"):
            lp = sl.types[name].loop
            assert (lp.while_.until, lp.while_.max_rounds) == ("30m", 120)
            assert sl.types[name].secrets == ["OPENAI_API_KEY"]
        # the local world runs through flock's own interpreter, never `flock` from PATH
        assert s.world.command[0] == "{python}"
        files.append(_render(tmp_path, "router-smoke"))
        r = load_spec(str(files[-1]))
        assert list(r.providers) == ["hf"] and r.limits.budget_usd == 15
        assert r.types["agent"].loop.while_.agent_usd == 2.5 and r.types["agent"].max_budget_usd
    finally:
        for f in files:
            f.unlink()
    # the committed generated specs are what render.py writes
    rd = _mod("render")
    st = load_spec(str(EXP / "spec.standin.yaml"))  # the router as a stand-in endpoint
    p = st.providers["vllm"]
    assert (p.base_url, p.api_key_env, p.self_hosted) == ("https://router.huggingface.co/v1",
                                                         "HF_TOKEN", False)
    assert p.pricing == {rd.ROUTER: [0.30, 1.20, 0.30]}
    assert {m.model for m in st.models.values()} == {f"vllm:{rd.ROUTER}"}
    lp = st.types["agent"].loop
    assert (lp.every, lp.while_.until, lp.while_.agent_usd, lp.wake.max) == ("60s", "20m", 3.0, 20)
    assert (st.limits.budget_usd, st.types["agent"].max_budget_usd,
            st.types["interviewer"].max_budget_usd) == (30, 0.33, 0.15)
    for variant, name in (("smoke", "spec.smoke.yaml"), ("standin", "spec.standin.yaml"),
                          ("jobs", "spec.jobs.yaml"),
                          ("router-smoke", "spec.router-smoke.yaml")):
        want = rd.HEAD.format(v=variant) + yaml.safe_dump(rd.render(variant, True),
                                                          sort_keys=False, width=100)
        assert (EXP / name).read_text() == want, f"{name}: run render.py --committed"


class _Controller:
    """A controller (TestClient) with FakeJobsClient, the world behind world.url."""

    def __init__(self, store_dir: Path, world_app, secrets: dict, board_url: str):
        from flock.controller.store import Store

        def world(req: httpx.Request) -> httpx.Response:
            r = self.world.request(req.method, req.url.path, content=req.content,
                                   headers=dict(req.headers))
            return httpx.Response(r.status_code, content=r.content, headers=r.headers)

        self.store, self.jobs = Store(store_dir), FakeJobsClient()
        self.world = TestClient(world_app)
        app = controller_app(self.store, TOKEN, secret_source=secrets, loop_interval=0,
                             world_transport=httpx.MockTransport(world))
        self.app = app
        self.http = self.client()
        self.settings = Settings(public_url="http://c", token=TOKEN, secret_source=secrets,
                                 rng=random.Random(0))

    def client(self) -> TestClient:
        return TestClient(self.app, headers={"Authorization": f"Bearer {TOKEN}"})

    def tick(self):
        from datetime import UTC, datetime

        return reconcile(self.store, self.jobs, datetime.now(UTC), self.settings)


def test_jobs_mode_dry_run(tmp_path, fakes, waves, monkeypatch):
    """spec.jobs.yaml applied to a controller with FakeJobsClient: the world is reached at
    world.url (a Space), waves raise replicas through `flock update`'s path, workers follow at
    agents_per_worker 25, the cap holds; nothing is launched for real."""
    from typer.testing import CliRunner

    from flock.world.server import create_app as world_app

    _, board, board_url = fakes
    secrets = {"CHATBOT_WORLD_TOKEN": "w0rld-t0ken", "CHATBOT_BOARD_INVITE": board.invite,
               "OPENAI_API_KEY": "fake-key"}
    world = _load_world().Chatbot3World(str(tmp_path / "ws"))
    ctl = _Controller(tmp_path / "ctl", world_app(world, "w0rld-t0ken"), secrets, board_url)
    monkeypatch.setattr("flock.client.connect", lambda *_: ctl.client())
    spec = _render(tmp_path, "jobs", MODEL_WORKER="m", CHATBOT_BOARD_URL=board_url,
                   CHATBOT_CHANNEL="chatbot-test", RUN3_WORLD_URL="https://world.example.invalid")
    try:
        # in process: flock.client.connect is the controller app
        res = CliRunner().invoke(cli.app, ["launch", str(spec), "--skip-preflight"])
        assert res.exit_code == 0, res.output
        stored = ctl.store.get_swarm("run3")["spec"]
        assert stored.world.manifest.name == "chatbot3" and stored.compute.kind == "hf-jobs"
        ctl.tick()
        assert ctl.jobs.launches == []  # replicas 0: no worker yet
        staggered = tmp_path / "waves-staggered.yaml"  # wave 1: 5 agents 0.2 s apart
        staggered.write_text("cap: 1000\nstagger_s: 0.2\nwaves:\n  - {at: 0, add: 5}\n")
        t0 = time.monotonic()
        assert waves.main(["tick", "--spec", str(spec), "--start", str(time.time()),
                           "--waves", str(staggered)]) == 0
        assert time.monotonic() - t0 >= 0.8  # 5 steps, 4 gaps
        assert ctl.store.get_swarm("run3")["spec"].types["agent"].replicas == 5
        start = time.time() - 301  # the schedule at 5 min: 40 agents
        assert waves.main(["tick", "--spec", str(spec), "--start", str(start),
                           "--waves", str(EXP / "waves.yaml")]) == 0
        assert ctl.store.get_swarm("run3")["spec"].types["agent"].replicas == 40
        ctl.tick()
        agent_jobs = [x for x in ctl.jobs.launches if x["labels"]["type"] == "agent"]
        assert len(agent_jobs) == math.ceil(40 / 25)
        sec = agent_jobs[0]["secrets"]
        assert {"FLOCK_WORLD_TOKEN", "CHATBOT_BOARD_INVITE", "OPENAI_API_KEY"} <= set(sec)
        assert "w0rld-t0ken" not in json.dumps(ctl.http.get("/swarms/run3").json())
        # the operator past the schedule: add, then the cap
        assert waves.main(["add", "960", "--spec", str(spec)]) == 0
        assert ctl.store.get_swarm("run3")["spec"].types["agent"].replicas == 1000
        assert waves.main(["add", "5", "--spec", str(spec)]) == 0  # capped: no change
        assert ctl.store.get_swarm("run3")["spec"].types["agent"].replicas == 1000
        for _ in range(30):  # the launch budget spreads the launches over passes
            ctl.tick()
        workers = ctl.store.list_workers("run3", "agent")
        assert len(workers) == math.ceil(1000 / 25) == 40
        assert waves.main(["set", "300", "--spec", str(spec)]) == 0  # the override may lower
        assert ctl.store.get_swarm("run3")["spec"].types["agent"].replicas == 300
        # Pi agents on top: only agent-pi changes; later waves and adds keep it
        def reps():
            ts = ctl.store.get_swarm("run3")["spec"].types
            return {t: ts[t].replicas for t in ("agent", "agent-think", "agent-pi")}

        assert waves.main(["add", "25", "--spec", str(spec), "--type", "agent-pi"]) == 0
        assert reps() == {"agent": 300, "agent-think": 0, "agent-pi": 25}
        assert waves.main(["add", "5", "--spec", str(spec)]) == 0
        assert reps() == {"agent": 305, "agent-think": 0, "agent-pi": 25}
        assert waves.main(["set", "20", "--spec", str(spec), "--type", "agent-pi"]) == 0
        assert reps() == {"agent": 305, "agent-think": 0, "agent-pi": 20}
        assert waves.main(["pin", "--spec", str(spec)]) == 0
        assert load_spec(str(spec)).types["agent-pi"].replicas == 20
        log = [json.loads(x) for x in Path(str(spec) + ".waves.jsonl").read_text().splitlines()]
        assert [x["total"] for x in log] == [5, 40, 1000, 300, 25, 305, 20]
        assert [x["why"][:10] for x in log] == ["schedule 0", "schedule 5", "add 960", "set 300",
                                               "add 25 --t", "add 5", "set 20 --t"]
        assert log[4]["after"] == {"agent-pi": 25}
        assert [(x["steps"], x["stagger_s"]) for x in log] == [(5, 0.2)] + [(1, 0)] * 6
    finally:
        for f in (spec, Path(str(spec) + ".waves.jsonl"), Path(str(spec) + ".waves.json")):
            f.unlink(missing_ok=True)
        ctl.store.close()


def _load_world():
    s = importlib.util.spec_from_file_location("chatbot3", ROOT / "worlds/chatbot3/chatbot3.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ the single job, end to end
INTERVIEW = script(*[("chat", {"booking": "{booking}", "message": f"hello {i}"}) for i in range(6)],
                   ("submit_verdict", {"booking": "{booking}", "evidence_script": ["echoes"],
                                       "evidence_agent": ["warm"], "verdict": "script",
                                       "confidence": 0.8}),
                   ("post", {"text": "INTERVIEW {booking}: script (0.8)", "thread": "{thread}"}),
                   ("done", {"summary": "interviewed"}))
WAIT = script(("read", {}))


def test_world_space_entry_point(tmp_path):
    """What the Space's start.sh runs: `python -m flock.world serve chatbot3:Chatbot3World` from
    worlds/chatbot3 serves the manifest behind the world token; the scripts parse."""
    import socket

    for f in ("start.sh", "prepare.sh"):
        assert subprocess.run(["bash", "-n", str(EXP / "world-space" / f)], check=False).returncode == 0
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**_env(tmp_path), "FLOCK_WORLD_TOKEN": "w0rld-t0ken", "CHATBOT_WS": str(tmp_path / "ws")}
    proc = subprocess.Popen([sys.executable, "-m", "flock.world", "serve",
                             "chatbot3:Chatbot3World", "--port", str(port)],
                            cwd=ROOT / "worlds" / "chatbot3", env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url, deadline = f"http://127.0.0.1:{port}/manifest", time.monotonic() + 30
        while True:
            try:
                r = httpx.get(url, headers={"X-World-Token": "w0rld-t0ken"}, timeout=2)
                break
            except httpx.TransportError:
                assert time.monotonic() < deadline and proc.poll() is None
                time.sleep(0.3)
        assert r.status_code == 200 and r.json()["name"] == "chatbot3"
        assert httpx.get(url, headers={"X-World-Token": "nope"}, timeout=2).status_code == 401
        assert (tmp_path / "ws" / "agents").is_dir()
    finally:
        proc.kill()
        proc.wait()


def test_waves_run_starts_its_clock_at_the_first_controller_answer(tmp_path, monkeypatch, waves,
                                                                 capsys):
    """Jobs/Slurm mode: `waves.py run` may start before the controller answers (a cold node).
    The clock starts at the first answer, so the first tick never merges two waves."""
    spec = _render(tmp_path, "jobs", MODEL_WORKER="m")
    sched = tmp_path / "w.yaml"
    sched.write_text("cap: 100\nwaves:\n  - {at: 0, add: 5}\n")
    answers = iter([RuntimeError("down"), RuntimeError("down"), None, None, None])
    sleeps: list[float] = []
    fake_now = [1000.0]

    def current(c, swarm, types):
        a = next(answers)
        if a is not None:
            raise a
        return {"agent": 5}

    def sleep(s):
        sleeps.append(s)
        fake_now[0] += s

    monkeypatch.setattr(waves, "current", current)
    monkeypatch.setattr(waves.time, "sleep", sleep)
    monkeypatch.setattr(waves.time, "time", lambda: fake_now[0])
    try:
        assert waves.main(["run", "--spec", str(spec), "--waves", str(sched)]) == 0
        state = json.loads(Path(str(spec) + ".waves.json").read_text())
        assert sleeps == [5, 5] and state["start"] == 1010.0  # after two failed contacts
        out = capsys.readouterr().out
        assert "waiting for the controller" in out and "waves clock started at" in out
    finally:
        for f in (spec, Path(str(spec) + ".waves.json")):
            f.unlink(missing_ok=True)
