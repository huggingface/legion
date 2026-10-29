"""experiments/eliza end to end under `flock run` (offline): the real CLI in a subprocess runs the
experiment's own spec, prompts, protocols and kickoff with worlds/eliza, against
tests/fake_openai.py and tests/fake_msgboard.py. Only the provider, the board, the loop prompts
(scripted tool calls) and the loop timings are swapped: 5 agents, round 0, 2 rounds, a final.

The world starts as a child process and stops with the run; write_file outside the agent's own
dir is refused; run_bot talks to the seeded bot; promote_best puts it at best/bot.py; the
bundle has run.json with the loops and budget summaries."""

import json
import shutil
import subprocess

import httpx
import pytest
import yaml

from flock.models import load_spec_yaml

from . import fake_msgboard, fake_openai
from .test_local_run_e2e import _bundle_events, _env, _flock, _run_dir
from .test_world import ROOT

AGENTS = ["ds-0000", "ds-0001", "ds-0002", "glm-0000", "glm-0001"]
BOT = ("import sys\nfor line in sys.stdin:\n    t = line.strip().lower()\n"
       "    print('Hello, you said: ' + t + '. Why do you feel that way? Tell me about your mother'"
       " ' and family. sam hiking tired nobody care. bye', flush=True)\n")


def script(*steps: tuple[str, dict]) -> str:
    return "script: " + json.dumps([list(s) for s in steps])


EDITED = BOT.replace("import sys\n", "import sys\n# tuned by {me}\n")
FIRST = script(("write_file", {"path": "agents/{me}/bot.py", "content": BOT}),
               ("write_file", {"path": "best/bot.py", "content": "print('mine now')"}),
               ("insert_after", {"path": "agents/{me}/bot.py", "anchor": "import sys",
                                 "text": "# tuned by {me}"}),
               ("edit_file", {"path": "agents/{me}/bot.py", "old": "line.strip()",
                              "new": "line.strip("}),  # does not compile: refused
               ("read_file", {"path": "agents/{me}/bot.py", "start": 1, "end": 2}),
               ("read", {}))
ROUND = script(("run_bot", {"path": "agents/{me}/bot.py", "lines": ["hello", "I feel sad"]}),
               ("promote_best", {"path": "agents/{me}/bot.py", "reason": "GREEN on run_bot"}),
               ("best_status", {}))
FINAL = script(("run_bot", {"path": "best/bot.py", "lines": ["hi", "My mother never calls"]}),
               ("post", {"text": "VERDICT GREEN best/bot.py: dry run", "thread": "new"}))


@pytest.fixture
def fakes(tmp_path):
    srv = fake_openai.serve(0, str(tmp_path / "requests.jsonl"))
    board, bsrv, burl = fake_msgboard.serve(fake_msgboard.Board(channels={12: "eliza-test"}))
    yield f"http://127.0.0.1:{srv.server_port}/v1", board, burl
    srv.shutdown()
    bsrv.shutdown()


def _results(run, agent: str) -> list[str]:
    return [json.dumps(e["data"]["message"]["content"]) for e in _bundle_events(run, agent)
            if e["type"] == "message" and e["data"]["message"].get("role") == "toolResult"]


def test_eliza_experiment_dry_run(tmp_path, fakes):
    model_url, board, board_url = fakes
    exp, world = tmp_path / "exp", tmp_path / "world"
    shutil.copytree(ROOT / "experiments" / "eliza", exp, ignore=shutil.ignore_patterns(".flock"))
    shutil.copytree(ROOT / "worlds" / "eliza", world, ignore=shutil.ignore_patterns("ws", "__py*"))
    spec = load_spec_yaml((exp / "spec.yaml").read_text())  # YAML 1.2: `on:` stays a key
    spec["providers"] = {"fake": {"type": "openai_compat", "base_url": model_url,
                                  "api_key_env": "FAKE_KEY",
                                  "pricing": {"fake-model": [1.0, 2.0, 0.5]}}}
    for m in spec["models"].values():
        m["model"] = "fake:fake-model"
    spec["world"]["dir"] = str(world)
    spec["comms"].update(url=board_url, channel="eliza-test", invite_env="FAKE_INVITE")
    for t in spec["types"].values():  # after `<<: *dev` both types hold this same dict
        loop = t["loop"]
        loop.update(first=FIRST, prompt=ROUND, every="1s", stagger="100ms",
                    final={"prompt": FINAL, "reserve_usd": 0.01},
                    **{"while": {"until": "5m", "max_rounds": 2}})
        loop.pop("prompt_file", None)
    spec["limits"]["budget_usd"] = 5
    (exp / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    p = subprocess.run(_flock("run", "spec.yaml", "--until", "idle", "--skip-preflight"),
                       cwd=exp, env={**_env(tmp_path), "FAKE_INVITE": board.invite},
                       capture_output=True, text=True, timeout=180, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "stopping (idle)" in p.stdout, p.stdout
    run = _run_dir(exp)

    # the world ran as a child process and stopped with the run
    log = (run / "logs" / "world-eliza.log").read_text()
    url = log.split("world eliza 0.1.0 on ", 1)[1].split()[0]
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{url}/manifest", timeout=2)

    ws = world / "ws"
    wlog = [json.loads(x) for x in (ws / "log.jsonl").read_text().splitlines()]
    for a in AGENTS:
        res = _results(run, a)
        # write_file outside its own dir: refused, as a result the agent reads
        assert any(f"refused: you ({a}) may write only under agents/{a}/" in r for r in res), res
        # the insert went in; the edit that broke the syntax was refused, the file kept
        assert (ws / "agents" / a / "bot.py").read_text() == EDITED.replace("{me}", a)
        assert any("NOT changed" in r and "syntax error" in r for r in res), res
        assert any("lines 1-2 of 5" in r and f"tuned by {a}" in r for r in res), res
        # run_bot talked to the seeded bot; promote_best answered; the final verdict was posted
        assert any("Hello, you said: i feel sad." in r for r in res), res
        assert any('\\"promoted\\": true' in r for r in res), res
        assert {(w["action"], w["path"]) for w in wlog if w["agent"] == a} >= {
            ("write_file", f"agents/{a}/bot.py"), ("refused", "best/bot.py"),
            ("insert_after", f"agents/{a}/bot.py"), ("syntax_error", f"agents/{a}/bot.py"),
            ("promote", f"agents/{a}/bot.py")}
        assert all(w["turn"] and w["req"] for w in wlog if w["agent"] == a)
    assert (ws / "best" / "bot.py").read_text() in {EDITED.replace("{me}", a) for a in AGENTS}
    hist = [json.loads(x) for x in (ws / "best" / "history.jsonl").read_text().splitlines()]
    assert len(hist) == 2 * len(AGENTS) and all(h["score"] == h["of"] for h in hist), hist
    verdicts = [m for m in board.msgs if "VERDICT GREEN" in m["body"]]
    assert len(verdicts) == len(AGENTS)

    # the agents had the world and board tools, no bash, and the world's rules
    reqs = [json.loads(x)["body"] for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    offered = {t["function"]["name"] for t in reqs[0]["tools"]}
    assert offered == {"read_file", "write_file", "copy_file", "edit_file", "insert_after",
                       "list_files", "run_bot", "promote_best", "best_status", "post", "read",
                       "search", "done"}, offered
    assert "## World: eliza (version 0.1.0)" in reqs[0]["messages"][0]["content"]

    # the bundle: run.json with each type's loop (2 rounds + final) and the budget summary
    doc = json.loads((run / "run.json").read_text())
    for t, n in (("ds", 3), ("glm", 2)):
        lp = doc["loops"][t]
        assert (lp["state"], lp["reason"], lp["final_sent"], lp["final_done"]) == (
            "done", "max_rounds", n, n), lp
        assert lp["rounds_total"] == 2 * n, lp
    assert doc["budget"]["budget_usd"] == 5 and 0 < doc["budget"]["cost_usd"] < 5, doc["budget"]
    assert doc["budget"]["stop"] is None
    assert (run / "events.jsonl").is_file()
