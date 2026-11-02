"""worlds/eliza in process: the own-dir rule, patch-style edits (edit_file, insert_after,
copy_file), line-range reads, the compile check with rollback, run_bot and promote_best."""

from __future__ import annotations

import asyncio
import importlib.util
import json

import pytest

from flock.world import Agent, ToolError

from .test_world import ROOT

BOT = "import sys\n\nfor line in sys.stdin:\n    print('hello: ' + line.strip(), flush=True)\n"
ME = Agent("ds-0000", type="ds", turn="t-3", request_id="req-1")
PATH = "agents/ds-0000/bot.py"


@pytest.fixture
def world(tmp_path):
    s = importlib.util.spec_from_file_location("eliza", ROOT / "worlds" / "eliza" / "eliza.py")
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    w = mod.ElizaWorld(str(tmp_path / "ws"))
    w.write_file(ME, PATH, BOT)
    return w


def _log(w) -> list[dict]:
    return [json.loads(x) for x in (w.ws / "log.jsonl").read_text().splitlines()]


def test_manifest_lists_the_edit_tools(world):
    m = world.manifest()
    assert [t["name"] for t in m["tools"]] == [
        "read_file", "write_file", "copy_file", "edit_file", "insert_after", "list_files",
        "run_bot", "promote_best", "best_status"]
    assert "Edit, do not rewrite" in m["description"]
    params = {t["name"]: t["parameters"] for t in m["tools"]}
    assert params["read_file"]["required"] == ["path"]
    assert set(params["edit_file"]["properties"]) == {"path", "old", "new"}  # no `agent`


def test_edit_file_replaces_a_unique_match_and_logs_it(world):
    r = world.edit_file(ME, PATH, "'hello: '", "'hi there: '")
    assert r["written"] == PATH
    assert (world.ws / PATH).read_text() == BOT.replace("'hello: '", "'hi there: '")
    rec = _log(world)[-1]
    assert (rec["action"], rec["path"], rec["agent"], rec["turn"], rec["req"]) == (
        "edit_file", PATH, "ds-0000", "t-3", "req-1")
    assert (rec["bytes_before"], rec["bytes_after"]) == (len(BOT), len(BOT) + 3)
    world.edit_file(ME, PATH, "\n\n", "\n")  # drops the blank line
    assert "\n\n" not in (world.ws / PATH).read_text()
    world.edit_file(ME, PATH, ", flush=True", "")  # `new` may be empty
    assert "flush" not in (world.ws / PATH).read_text()


@pytest.mark.parametrize(("old", "count"), [("nowhere", 0), ("line", 2)])
def test_edit_file_refuses_zero_or_several_matches(world, old, count):
    with pytest.raises(ToolError) as e:
        world.edit_file(ME, PATH, old, "x")
    assert f"occurs {count} times" in e.value.message and e.value.code == f"matches_{count}"
    assert (world.ws / PATH).read_text() == BOT


def test_insert_after_adds_lines_after_the_anchor_line(world):
    world.insert_after(ME, PATH, "import sys", "import re\nimport random")
    assert (world.ws / PATH).read_text().startswith("import sys\nimport re\nimport random\n\nfor")
    world.insert_after(ME, PATH, "flush=True)", "# end")  # the last line, at EOF
    assert (world.ws / PATH).read_text().endswith("flush=True)\n# end\n")
    with pytest.raises(ToolError, match="occurs 3 times"):
        world.insert_after(ME, PATH, "import", "x = 1")


def test_read_file_line_ranges(world):
    assert world.read_file(PATH) == f"# {PATH}: lines 1-4 of 4\n{BOT}"
    assert world.read_file(PATH, start=3, end=3) == (
        f"# {PATH}: lines 3-3 of 4\nfor line in sys.stdin:\n")
    assert world.read_file(PATH, start=4).startswith(f"# {PATH}: lines 4-4 of 4\n    print(")
    assert world.read_file(PATH, end=1) == f"# {PATH}: lines 1-1 of 4\nimport sys\n"


def test_a_change_that_does_not_compile_keeps_the_previous_content(world):
    with pytest.raises(ToolError) as e:
        world.edit_file(ME, PATH, "for line in sys.stdin:", "for line in sys.stdin")
    msg = e.value.message
    assert e.value.code == "syntax_error" and "NOT changed" in msg
    assert "line 3" in msg and PATH in msg, msg
    with pytest.raises(ToolError, match="syntax error"):
        world.write_file(ME, PATH, "def (:\n")
    with pytest.raises(ToolError, match="syntax error"):
        world.insert_after(ME, PATH, "import sys", "  bad indent")
    assert (world.ws / PATH).read_text() == BOT
    assert [r["action"] for r in _log(world)[-3:]] == ["syntax_error"] * 3
    assert not list((world.ws / "agents" / "ds-0000").glob(".*"))  # no temp files left
    world.write_file(ME, "agents/ds-0000/notes.md", "def (:\n")  # only .py files are compiled


def test_only_the_own_dir_is_writable(world):
    other = Agent("glm-0001")
    for call in (lambda: world.edit_file(other, PATH, "sys", "os"),
                 lambda: world.insert_after(other, PATH, "import sys", "x = 1"),
                 lambda: world.write_file(other, PATH, BOT),
                 lambda: world.copy_file(ME, PATH, "best/bot.py"),
                 lambda: world.write_file(ME, "agents/ds-0000/../glm-0001/bot.py", BOT)):
        with pytest.raises(ToolError, match="may write only under agents/") as e:
            call()
        assert e.value.code == "not_yours"
    assert (world.ws / PATH).read_text() == BOT
    assert world.copy_file(other, PATH, "agents/glm-0001/bot.py")["written"] == (
        "agents/glm-0001/bot.py")
    with pytest.raises(ToolError, match="create it with write_file or copy_file"):
        world.edit_file(ME, "agents/ds-0000/missing.py", "a", "b")


def test_run_bot_and_promote_best(world):
    r = asyncio.run(world.run_bot(PATH, ["hi", "I feel sad"]))
    assert r["status"] == "ok" and r["turns"][1] == {"user": "I feel sad", "bot": "hello: I feel sad"}
    p = asyncio.run(world.promote_best(ME, PATH, "seed"))
    assert p["promoted"] and (world.ws / "best" / "bot.py").read_text() == BOT
    world.edit_file(ME, PATH, "'hello: ' + line.strip()", "''")  # empty replies: a regression
    p = asyncio.run(world.promote_best(ME, PATH, "worse"))
    assert not p["promoted"] and "the current best" in p["why"]
