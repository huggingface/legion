"""experiments/eliza end to end under `flock run` (offline): the real CLI in a subprocess runs the
experiment's own spec, prompts, protocols and kickoff with worlds/eliza, against
tests/fake_openai.py and tests/fake_msgboard.py. Only the provider, the board, the loop prompts
(scripted tool calls) and the loop timings are swapped: 5 agents, round 0, 2 rounds, a final.

The world starts as a child process and stops with the run; write_file outside the agent's own
dir is refused; run_bot talks to the seeded bot; promote_best puts it at best/bot.py; the
bundle has run.json with the loops and budget summaries."""

from __future__ import annotations

import json

import pytest

from . import fake_msgboard, fake_openai
from .test_local_run_e2e import _bundle_events

AGENTS = ["ds-0000", "ds-0001", "ds-0002", "glm-0000", "glm-0001"]
BOT = ("import sys\nfor line in sys.stdin:\n    t = line.strip().lower()\n"
       "    print('Hello, you said: ' + t + '. Why do you feel that way? Tell me about your mother'"
       " ' and family. sam hiking tired nobody care. bye', flush=True)\n")


def script(*steps: tuple[str, dict]) -> str:
    return "script: " + json.dumps([list(s) for s in steps])


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
