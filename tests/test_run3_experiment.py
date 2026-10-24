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
import tarfile
import time
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from flock import cli
from flock.cli import load_spec
from flock.controller.api import create_app as controller_app
from flock.controller.jobs import FakeJobsClient
from flock.controller.reconcile import Settings, reconcile
from flock.models import load_spec_yaml

from .test_chatbot_experiment import _kill, _no_secrets, _until, _wait, fakes, script
from .test_local_run_e2e import _bundle_events, _env
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
        assert sorted(s.types) == ["agent", "agent-think", "interviewer"]
        assert s.naming == "words" and s.types["interviewer"].name_prefix == "iv-"
        for name in ("agent", "agent-think"):
            t = s.types[name]
            assert (t.replicas, t.agents_per_worker, t.tool_choice, t.bash) == (0, 25, "required",
                                                                               False)
            assert t.max_turns == 20 and "at most 18 calls" in t.loop.prompt
            assert "exactly as `run_bot` returned it" in t.loop.prompt
            assert "exactly as `run_bot` returned it" in t.loop.first
            lp = t.loop
            assert (lp.every, lp.while_.until, lp.wake.min_gap, lp.wake.max) == (
                "turn_end", "48h", "10s", 20)
            assert lp.first.startswith("You just joined a running effort")
            assert "search" in lp.prompt and len(lp.prompt) < 1400  # short, action-oriented
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
    monkeypatch.setattr(cli, "client", ctl.client)
    spec = _render(tmp_path, "jobs", MODEL_WORKER="m", CHATBOT_BOARD_URL=board_url,
                   CHATBOT_CHANNEL="chatbot-test", RUN3_WORLD_URL="https://world.example.invalid")
    try:
        # in process: cli.client is the controller app
        res = CliRunner().invoke(cli.app, ["apply", str(spec), "--skip-preflight"])
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
        log = [json.loads(x) for x in Path(str(spec) + ".waves.jsonl").read_text().splitlines()]
        assert [x["total"] for x in log] == [5, 40, 1000, 300]
        assert [x["why"][:10] for x in log] == ["schedule 0", "schedule 5", "add 960", "set 300"]
        assert [(x["steps"], x["stagger_s"]) for x in log] == [(5, 0.2), (1, 0), (1, 0), (1, 0)]
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
NEWCOMER_SCRIPT = script(
    ("search", {"query": "best"}),
    ("best_status", {}),
    ("write_file", {"path": "agents/{me}/bot/__main__.py", "content": MAIN}),
    ("write_file", {"path": "agents/{me}/bot/words.py", "content": WORDS}),
    ("promote_best", {"path": "agents/{me}/bot", "reason": "first bot"}),
    ("book_interview", {"path": "agents/{me}/bot"}),
    ("spawn_interviewer", {"booking": "{me}-iv1-strong", "thread": "new"}),
    ("done", {"summary": "joined"}))
INTERVIEW = script(*[("chat", {"booking": "{booking}", "message": f"hello {i}"}) for i in range(6)],
                   ("submit_verdict", {"booking": "{booking}", "evidence_script": ["echoes"],
                                       "evidence_agent": ["warm"], "verdict": "script",
                                       "confidence": 0.8}),
                   ("post", {"text": "INTERVIEW {booking}: script (0.8)", "thread": "{thread}"}),
                   ("done", {"summary": "interviewed"}))
WAIT = script(("read", {}))


def _payload(tmp_path: Path, waves: str) -> Path:
    src, bucket = tmp_path / "src", tmp_path / "bucket"
    exp = src / "experiments" / "run3"
    shutil.copytree(EXP, exp, ignore=shutil.ignore_patterns("*.local.yaml*", "__py*", "ws"))
    for w in ("chatbot3", "chatbot2", "chatbot", "eliza"):
        shutil.copytree(ROOT / "worlds" / w, src / "worlds" / w,
                        ignore=shutil.ignore_patterns("ws", "__py*"))
    tool = exp / "tools" / "interview.py"  # the interviewer's task: a script for the fake model
    text = tool.read_text()
    a = text.index("PROMPT = (")
    tool.write_text(text[:a] + f"PROMPT = {INTERVIEW!r}" + text[text.index("\n\n\n", a):])
    d = load_spec_yaml((exp / "spec.yaml").read_text())
    for t in d["types"].values():
        if lp := t.get("loop"):
            lp.pop("prompt_file", None)
            first = lp["first"]  # the real newcomer prompt, then the fake model's script
            t["loop"] = {**lp, "first": first + NEWCOMER_SCRIPT, "prompt": WAIT, "every": "5s",
                         "stagger": "100ms", "final": {"prompt": script(("done", {}))}}
            t["max_turns"] = 30
    (exp / "spec.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    w = exp / waves  # the variant's schedule, minutes -> seconds
    w.write_text(w.read_text().replace("at: 5m", "at: 8s").replace("at: 10m", "at: 16s")
                 + "stagger_s: 0.2\n")  # wave 1's agents join 0.2 s apart
    (bucket / "payload").mkdir(parents=True)
    with tarfile.open(bucket / "payload" / "repo.tar.gz", "w:gz") as tar:
        for x in ("experiments", "worlds"):
            tar.add(src / x, arcname=x)
    return bucket


def _second_server(tmp_path: Path):
    """Another fake OpenAI server with its own request log (requests-2.jsonl)."""
    import threading
    from http.server import ThreadingHTTPServer

    from . import fake_openai

    h = type("Handler2", (fake_openai.Handler,), {"log_path": str(tmp_path / "requests-2.jsonl")})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []


VARIANTS = {  # variant: waves file, cap, wave totals, the loop's `while`
    "smoke": ("waves-smoke.yaml", 40, [(0, 20), (20, 40)], {"until": "30m", "max_rounds": 120}),
    "standin": ("waves-standin.yaml", 20, [(0, 10), (10, 20)],
                {"until": "20m", "max_rounds": 40, "agent_usd": 3.0})}


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_single_job_three_waves_end_to_end(tmp_path, fakes, variant):
    """The bootstrap rehearsal of the smoke variants (RUN3_VARIANT, its waves file with the
    minutes as seconds): smoke 0 -> 20 -> 40 (the third wave capped), standin 0 -> 10 -> 20."""
    waves, cap, totals, loop_while = VARIANTS[variant]
    model_url, board, board_url = fakes
    bucket, home = _payload(tmp_path, waves), tmp_path / "job"
    env = {**_env(tmp_path), "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
           "PYTHON": sys.executable, "CHATBOT_PREINSTALLED": "1", "CHATBOT_HOME": str(home),
           "CHATBOT_BUCKET": str(bucket), "CHATBOT_BOARD_URL": board_url,
           "CHATBOT_CHANNEL": "chatbot-test", "CHATBOT_BOARD_INVITE": board.invite,
           "CHATBOT_WORLD_TOKEN": "w0rld-t0ken", "CHATBOT_SYNC_S": "5", "CHATBOT_CHECK_S": "1",
           "CHATBOT_TICK_S": "1", "CHATBOT_FLOCK_ARGS": "--skip-preflight",
           "OPENAI_BASE_URL": model_url, "OPENAI_API_KEY": "fake-key", "MODEL_WORKER": "fake-model",
           "HF_TOKEN": "fake-key",
           "RUN3_VARIANT": variant, "RUN3_WAVES": waves, "CHATBOT_RUN": "t1",
           # a cold start: the controller is up well after the second wave would be due if
           # the clock started with the script (it starts when the controller is healthy)
           "CHATBOT_TEST_START_DELAY_S": "10"}
    out = open(tmp_path / "job.out", "w")  # noqa: SIM115 (the process holds it)
    proc = subprocess.Popen(["bash", "-c", (EXP / "job" / "bootstrap.sh").read_text()], env=env,
                            stdout=out, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    ws, wlog = home / "ws", home / "logs" / "waves.jsonl"
    try:
        _until(proc, tmp_path, "flock run started")
        deadline = time.monotonic() + 240
        while True:  # every agent joined, each had its first turn; interviews filed verdicts
            log = _jsonl(ws / "log.jsonl")
            joined = {x["agent"] for x in log if x["action"] == "book_interview"}
            verdicts = sum(x["action"] == "verdict" for x in log)
            if len(joined) == cap and verdicts >= cap // 4 and time.time() - wlog.stat().st_ctime > 10:
                break
            assert proc.poll() is None and time.monotonic() < deadline, (
                len(joined), verdicts, _jsonl(wlog), (tmp_path / "job.out").read_text()[-3000:])
            time.sleep(1)
        (bucket / "control").mkdir(exist_ok=True)
        if variant == "smoke":  # a live server switch: every agent moves at its next turn
            srv2 = _second_server(tmp_path)
            url2 = f"http://127.0.0.1:{srv2.server_port}/v1"
            (bucket / "control" / "endpoints").write_text(url2 + "\n")
            _until(proc, tmp_path, "endpoints 1 ok", timeout=60)
            deadline = time.monotonic() + 60
            while not (tmp_path / "requests-2.jsonl").is_file() or len(
                    (tmp_path / "requests-2.jsonl").read_text().splitlines()) < 10:
                assert time.monotonic() < deadline and proc.poll() is None
                time.sleep(1)
        (bucket / "control" / "stop").write_text("stop\n")
        assert _wait(proc, 180) == 0, (tmp_path / "job.out").read_text()[-3000:]
    finally:
        if proc.poll() is None:
            _kill(proc)
        out.close()
    text = (tmp_path / "job.out").read_text()
    assert "stop marker found" in text and "final sync" in text, text[-3000:]
    # the waves; a capped wave makes no change and no line
    waves_log = _jsonl(wlog)
    assert waves_log[0]["total"] == totals[0][1]  # the first tick never merges two waves
    assert "waves clock started at" in text
    assert [(sum(x["before"].values()), x["total"]) for x in waves_log] == totals
    assert all(x["lines"] and "error" not in x for x in waves_log), waves_log
    n1 = totals[0][1]  # wave 1 staggered: one agent per step, the last step n1 - 1 -> n1
    assert (waves_log[0]["steps"], waves_log[0]["stagger_s"]) == (n1, 0.2)
    assert f"types.agent.replicas: {n1 - 1} -> {n1}" in waves_log[0]["lines"]
    assert waves_log[1]["steps"] == 1  # later waves: all at once
    # the bundle in the bucket, with every agent's events
    (run,) = [p for p in (bucket / "flock" / ".flock").iterdir() if p.is_dir()]
    assert "run3-t1" in (run / "run.json").read_text()
    doc = json.loads((run / "run.json").read_text())
    # naming: words: readable names for the agents, `iv-` for the interviewers
    from flock.controller.naming import HANDLE_RE

    agents = sorted(x["name"] for x in doc["agents"] if x["type"] == "agent")
    assert len(agents) == cap and all(HANDLE_RE.match(a) and not a.startswith("iv-")
                                      for a in agents), agents
    ivs = [r["name"] for r in doc["removed_agents"]]
    assert ivs and all(n.startswith("iv-") and HANDLE_RE.match(n) for n in ivs), ivs
    posts = [m for m in board.msgs if m["body"].startswith("INTERVIEW ")]  # on the board
    assert posts and all(m["sender"]["external_id"] in ivs for m in posts)
    assert any(a in m["body"] for m in posts for a in agents)  # booking ids carry the name
    if variant == "smoke":  # the server switch: turns moved to server 2, named in turn_ended
        ends = [e["data"]["data"] for a in agents
                for e in _bundle_events(run, a) if e["type"] == "turn_ended"]
        host1, host2 = model_url.split("/")[2], f"127.0.0.1:{srv2.server_port}"
        assert {d.get("endpoint") for d in ends} == {host1, host2}
        moved = [d for d in ends if d.get("endpoint_from")]
        assert moved and all((d["endpoint_from"], d["endpoint"]) == (host1, host2) for d in moved)
        assert not (bucket / "control" / "endpoints").exists()
        assert "endpoints 1 ok" in (bucket / "logs" / "bootstrap.log").read_text()
        srv2.shutdown()
    gone = doc["removed_agents"]  # removed interviewers carry their cost
    assert gone and all(r["type"] == "interviewer" and r["cost_usd"] is not None for r in gone)
    if variant == "standin":  # priced: live agents + removed ones = the swarm's total
        assert all(r["cost_usd"] > 0 for r in gone)
        live = sum(x["cost_usd"] for x in doc["agents"])
        assert abs(live + sum(r["cost_usd"] for r in gone) - doc["totals"]["cost_usd"]) < 1e-3
    local = yaml.safe_load((home / "repo/experiments/run3/spec.local.yaml").read_text())
    assert local["types"]["agent"]["loop"]["while"] == loop_while
    looped = 0
    for a in agents:  # a newcomer: the newcomer prompt first, then rounds
        ev = _bundle_events(run, a)
        kinds = [e["data"]["data"]["kind"] for e in ev if e["type"] == "loop_prompt"]
        assert kinds[0] == "first" and "first" not in kinds[1:], (a, kinds)
        looped += "loop" in kinds
        first_user = next(e["data"]["message"] for e in ev if e["type"] == "message"
                          and e["data"]["message"].get("role") == "user")
        assert "You just joined a running effort" in json.dumps(first_user)
    if variant == "smoke":  # standin's rounds are 60 s apart: none in this short rehearsal
        assert looped >= 5, looped  # rounds follow the first turn (few under a loaded machine)
    # promotion from the start: the first bot promoted at once, identical ones refused as ties
    hist = _jsonl(ws / "best" / "history.jsonl")
    assert len(hist) == 1 and hist[0]["agent"] in agents
    log = _jsonl(ws / "log.jsonl")
    assert sum(x["action"] == "promote_refused" and "a tie" in x.get("why", "") for x in log) == cap - 1
    # interviews: every agent booked one, none refused for quota, verdicts on frozen copies
    books = json.loads((ws / "interviews.json").read_text())
    assert len(books) == cap and not [x for x in log if x.get("why") == "quota"]
    vs = _jsonl(ws / "verdicts.jsonl")
    assert len(vs) >= cap // 4 and all(v["level"] == "strong" and v["counts"] for v in vs)
    assert all(v["snapshot"] == f"interviews/{v['booking']}/bot" for v in vs)
    assert (bucket / "ws" / "best" / "current.json").is_file()  # the mirror and the ws sync
    _no_secrets(bucket, board)


def test_world_space_entry_point(tmp_path):
    """What the Space's start.sh runs: `flock world serve chatbot3:Chatbot3World` from
    worlds/chatbot3 serves the manifest behind the world token; the scripts parse."""
    import socket

    for f in ("start.sh", "prepare.sh"):
        assert subprocess.run(["bash", "-n", str(EXP / "world-space" / f)], check=False).returncode == 0
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**_env(tmp_path), "FLOCK_WORLD_TOKEN": "w0rld-t0ken", "CHATBOT_WS": str(tmp_path / "ws")}
    proc = subprocess.Popen([sys.executable, "-m", "flock.cli", "world", "serve",
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
    import contextlib

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
    monkeypatch.setattr(cli, "client", lambda: contextlib.nullcontext(object()))
    try:
        assert waves.main(["run", "--spec", str(spec), "--waves", str(sched)]) == 0
        state = json.loads(Path(str(spec) + ".waves.json").read_text())
        assert sleeps == [5, 5] and state["start"] == 1010.0  # after two failed contacts
        out = capsys.readouterr().out
        assert "waiting for the controller" in out and "waves clock started at" in out
    finally:
        for f in (spec, Path(str(spec) + ".waves.json")):
            f.unlink(missing_ok=True)
