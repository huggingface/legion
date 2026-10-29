"""Workspace tools for a chatbot-writing swarm (the shape of the Eliza experiment's tools).

Every public function with a docstring is a tool (docs/tools.md). `ctx` is filled in by flock and
never shown to the model. Raise ToolError for a refusal the model should read as is.
"""

import json
import re
import subprocess
import sys
import time

from flock.tools import Context, ToolError

MAX_FILE_CHARS = 20_000
MAX_MESSAGES = 20
BOT_TIMEOUT_S = 20
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def _ws(ctx: Context):
    d = ctx.workdir / "ws"
    d.mkdir(exist_ok=True)
    return d


def ws_write(filename: str, content: str, reason: str, ctx: Context) -> dict:
    """Write a file to your workspace (ws/), replacing it if it exists.

    Python files are checked for syntax errors before they are written.

    Args:
        filename: a plain file name such as bot.py (no directories)
        content: the whole file
        reason: one line: why this change
    """
    if not _NAME.match(filename) or filename.startswith("."):
        raise ToolError(f"bad filename {filename!r}: use a plain name like bot.py")
    if len(content) > MAX_FILE_CHARS:
        raise ToolError(f"file is {len(content)} chars; the limit is {MAX_FILE_CHARS}")
    if filename.endswith(".py"):
        try:
            compile(content, filename, "exec")
        except SyntaxError as e:
            raise ToolError(f"not written: syntax error line {e.lineno}: {e.msg}") from None
    (_ws(ctx) / filename).write_text(content)
    # tool state outside the agent's workdir: shared by the agents of this worker
    with (ctx.data_dir / "writes.jsonl").open("a") as f:
        f.write(json.dumps({"ts": time.time(), "agent": ctx.agent, "file": filename,
                            "reason": reason, "tools_version": ctx.tools_version}) + "\n")
    return {"written": f"ws/{filename}", "chars": len(content)}


def run_bot(path: str, messages: list[str], ctx: Context) -> dict:
    """Run a bot script on a conversation and return its replies.

    The script gets one user message per line on stdin and must print one reply line per
    message.

    Args:
        path: the script, relative to your workdir, e.g. ws/bot.py
        messages: the user messages, in order (at most 20)
    """
    if not messages:
        raise ToolError("messages must be a non-empty list of strings")
    if len(messages) > MAX_MESSAGES:
        raise ToolError(f"at most {MAX_MESSAGES} messages per conversation")
    script = (ctx.workdir / path).resolve()
    if not script.is_relative_to(ctx.workdir.resolve()) or not script.is_file():
        raise ToolError(f"no such file in your workdir: {path}")
    stdin = "".join(m.replace("\n", " ") + "\n" for m in messages)
    try:
        p = subprocess.run([sys.executable, "-I", str(script)], input=stdin, capture_output=True,
                           text=True, timeout=BOT_TIMEOUT_S, cwd=script.parent, check=False)
    except subprocess.TimeoutExpired:
        raise ToolError(f"the bot did not answer within {BOT_TIMEOUT_S}s") from None
    replies = p.stdout.splitlines()
    out: dict = {"conversation": [{"user": m, "bot": replies[i] if i < len(replies) else None}
                                  for i, m in enumerate(messages)]}
    if p.returncode != 0:
        out["error"] = p.stderr.strip()[-500:] or f"exit code {p.returncode}"
    return out
