"""The Eliza world (experiments/eliza): agents write ELIZA-style bots in a shared workspace
`$ELIZA_WS` (default ws/ here): agents/<agent>/ (writable only by that agent, by X-Flock-Agent),
best/ (bot.py, v<n>.py, history.jsonl), probes/ (seeded from probes/ here), log.jsonl. Bots run
as `python -I bot.py` in an empty temp dir under rlimits and a timeout: not a sandbox."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from flock.world import Agent, ToolError, World, tool

HERE = Path(__file__).resolve().parent
TIMEOUT_S, MEM, OUT_CAP, READ_CAP, WRITE_CAP = 5, 256 << 20, 64 << 10, 60_000, 200_000
LIMITS = f"((r.RLIMIT_AS,{MEM}),(r.RLIMIT_FSIZE,{OUT_CAP}),(r.RLIMIT_CPU,{TIMEOUT_S}))"
SHIM = (f"import resource as r,runpy,sys\nfor k,v in {LIMITS}: r.setrlimit(k,(v,v))\n"
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")


class ElizaWorld(World):
    name, version, description = "eliza", "0.1.0", (HERE / "rules.md").read_text()

    def __init__(self, root: str | None = None) -> None:
        self.ws = Path(root or os.environ.get("ELIZA_WS", "ws")).resolve()
        for d in ("agents", "best", "probes"):
            (self.ws / d).mkdir(parents=True, exist_ok=True)
        if not any((self.ws / "probes").glob("*.txt")):
            shutil.copytree(HERE / "probes", self.ws / "probes", dirs_exist_ok=True)
        self._promote = asyncio.Lock()

    def _path(self, path: str, must_exist: bool = True) -> Path:
        p = (self.ws / path.strip().lstrip("/")).resolve()
        if p != self.ws and self.ws not in p.parents:
            raise ToolError(f"{path!r} is outside the workspace", "bad_path")
        if must_exist and not p.is_file():
            raise ToolError(f"no file {path!r}; list_files shows what exists", "not_found")
        return p

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.ws).as_posix()

    def _append(self, rel: str, rec: dict) -> None:
        with open(self.ws / rel, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _log(self, agent: Agent, action: str, **data: object) -> None:
        self._append("log.jsonl", {"ts": round(time.time(), 3), "agent": agent.name, "turn":
                                   agent.turn, "req": agent.request_id, "action": action, **data})

    def _run(self, path: Path, lines: list[str]) -> dict:
        """One conversation: each line with its reply line, the exit status, stderr's tail."""
        with tempfile.TemporaryDirectory() as tmp:
            out, err = Path(tmp) / ".out", Path(tmp) / ".err"
            with open(out, "wb") as fo, open(err, "wb") as fe:
                proc = subprocess.Popen([sys.executable, "-I", "-c", SHIM, str(path)], cwd=tmp,
                                        env={}, stdin=subprocess.PIPE, stdout=fo, stderr=fe,
                                        start_new_session=True)
                try:
                    proc.communicate("".join(x + "\n" for x in lines).encode(), TIMEOUT_S)
                    status = f"exit {proc.returncode}" if proc.returncode else "ok"
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, 9)
                    proc.wait()
                    status = f"timeout after {TIMEOUT_S} s"
            replies = out.read_text(errors="replace").splitlines()
            bots = replies + [None] * len(lines)
            return {"status": status, "stderr": err.read_text(errors="replace")[-400:],
                    "extra_lines": max(0, len(replies) - len(lines)),
                    "turns": [{"user": u, "bot": r} for u, r in zip(lines, bots)]}

    def _probes(self) -> dict[str, list[tuple[str, list[str]]]]:
        """probes/*.txt: a user line per line, optionally ` => kw1|kw2` (the reply holds one)."""
        return {p.name: [(m.strip(), [k.strip().lower() for k in kws.split("|") if k.strip()])
                         for m, _, kws in (x.partition(" => ") for x in p.read_text().splitlines())
                         if m.strip() and not m.startswith("#")]
                for p in sorted((self.ws / "probes").glob("*.txt"))}

    def _score(self, path: Path) -> dict:
        """1 point per probe file run without a crash, per non-empty reply, per keyword hit."""
        got, total, report, crashed = 0, 0, {}, False
        for name, rows in self._probes().items():
            r = self._run(path, [m for m, _ in rows])
            crashed |= r["status"] != "ok"
            fails = [] if r["status"] == "ok" else [r["status"]]
            for (msg, kws), t in zip(rows, r["turns"]):
                reply = (t["bot"] or "").strip().lower()
                hit = bool(kws) and any(k in reply for k in kws)
                got += bool(reply) + hit
                if not reply or (kws and not hit):
                    fails.append(f"{msg!r}: no reply" if not reply else f"{msg!r}: none of {kws}")
            got += r["status"] == "ok"
            total += 1 + len(rows) + sum(1 for _, k in rows if k)
            report[name] = "GREEN" if not fails else "RED: " + "; ".join(fails[:3])
        return {"score": got, "of": total, "probes": report, "crashed": crashed}

    def _history(self) -> list[dict]:
        hist = self.ws / "best" / "history.jsonl"
        return [json.loads(x) for x in hist.read_text().splitlines()] if hist.is_file() else []

    @tool
    def read_file(self, path: str) -> str:
        """Read a workspace file (any agent's, best/bot.py, probes/*.txt).

        Args:
            path: e.g. agents/ds-0001/bot.py or best/bot.py
        """
        text = self._path(path).read_text(errors="replace")
        return text if len(text) <= READ_CAP else text[:READ_CAP] + "\n[... truncated]"

    @tool
    def write_file(self, agent: Agent, path: str, content: str) -> dict:
        """Write a whole file under agents/<your name>/ (replaces it if it exists).

        Args:
            path: e.g. agents/<your name>/bot.py or agents/<your name>/notes.md
            content: the complete file content
        """
        p, size = self._path(path, must_exist=False), len(content.encode())
        if self.ws / "agents" / agent.name not in p.parents:
            self._log(agent, "write_refused", path=path)
            raise ToolError(f"refused: you ({agent.name}) may write only under agents/"
                            f"{agent.name}/, not {path!r}. best/bot.py changes by promote_best.",
                            "not_yours")
        if size > WRITE_CAP:
            raise ToolError(f"refused: {size} bytes, the limit is {WRITE_CAP}", "too_big")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self._log(agent, "write", path=self._rel(p), bytes=size)
        return {"written": self._rel(p), "bytes": size}

    @tool
    def list_files(self, prefix: str = "") -> dict:
        """List workspace files, newest first, with size and age in seconds.

        Args:
            prefix: only paths starting with this, e.g. agents/glm-0000/ or best/
        """
        files = sorted((p for p in self.ws.rglob("*") if p.is_file() and p.name != "log.jsonl"
                        and self._rel(p).startswith(prefix)), key=lambda p: -p.stat().st_mtime)
        return {"total": len(files), "files": [
            {"path": self._rel(p), "bytes": p.stat().st_size,
             "age_s": int(time.time() - p.stat().st_mtime)} for p in files[:200]]}

    @tool
    async def run_bot(self, path: str, lines: list[str]) -> dict:
        """Talk to a bot: run it on one fresh conversation and return the transcript.

        Args:
            path: the bot file, e.g. agents/ds-0002/bot.py or best/bot.py
            lines: the user's messages in order (1 to 20, one line each)
        """
        p = self._path(path)
        if not 0 < len(lines) <= 20:
            raise ToolError("give 1 to 20 lines", "bad_lines")
        return await asyncio.to_thread(self._run, p, [" ".join(x.split())[:500] for x in lines])

    @tool
    async def promote_best(self, agent: Agent, path: str, reason: str) -> dict:
        """Make a bot the team's best (best/bot.py) if it scores at least as well on the probes.

        Args:
            path: the candidate bot file, e.g. agents/ds-0001/bot.py
            reason: one line: what it does better, and your run_bot evidence
        """
        p = self._path(path)
        async with self._promote:
            cand, hist = await asyncio.to_thread(self._score, p), self._history()
            best = self.ws / "best" / "bot.py"
            cur = await asyncio.to_thread(self._score, best) if hist else None
            why = ("it crashed or timed out on a probe" if cand["crashed"] else
                   f"it scores {cand['score']}, the current best {cur['score']}"
                   if cur and cand["score"] < cur["score"] else "")
            if why:
                self._log(agent, "promote_refused", path=path, score=cand["score"], why=why)
                return {"promoted": False, "why": why, "candidate": cand, "best": cur}
            text, n = p.read_text(errors="replace"), len(hist) + 1
            rec = {"n": n, "path": self._rel(p), "agent": agent.name, "reason": reason[:300],
                   "score": cand["score"], "of": cand["of"], "ts": round(time.time()),
                   "sha": hashlib.sha256(text.encode()).hexdigest()[:12]}
            for dst in (best, best.with_name(f"v{n}.py")):
                dst.write_text(text)
            self._append("best/history.jsonl", rec)
            self._log(agent, "promote", path=rec["path"], score=cand["score"], n=n)
            return {"promoted": True, "best": rec, "probes": cand["probes"]}

    @tool
    def best_status(self) -> dict:
        """Show best/bot.py: where it came from, its probe score, and the last 5 promotions."""
        recs = self._history()
        return {"best": recs[-1] if recs else None, "earlier": recs[-6:-1][::-1],
                "probes": sorted(self._probes())}
