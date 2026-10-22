"""The chatbot world (experiments/chatbot-v1): the Eliza world (worlds/eliza) with two teams and a
phase. Workspace `$CHATBOT_WS` (default ws/ here): teams/<team>/agents/<agent>/ (the team is the
letter in the agent's type: ds-a-0001 is team a), best/, probes/ (seeded from probes/ here),
log.jsonl, phase.json.

- Phase `teams`: an agent reads, lists, copies and runs only its own team's files (plus best/
  and probes/), and promote_best is closed.
- Phase `merged`: everything is readable and promote_best is open; best/ is the one shared bot.

Interviews: a team agent `qualify`s a candidate (a mechanical bar on the probe conversations:
it runs within the time limit, answers every line, no empty or repeated reply; the file's hash
is recorded) and spawns an interviewer (flock spawn, the experiment's
spawn_interviewer tool). The interviewer (`<parent>-interviewer-<level>-NNNN`, same team as its
parent) talks to the qualified, unchanged candidate with `chat` (each turn re-runs the bot on
the whole conversation so far) and files `submit_verdict`; the team reads `verdicts`. Each
interview takes from the team's quota (QUOTA units per rolling hour, a strong one counts double)
at its first `chat`.

The operator flips the phase (never an agent): `python chatbot.py merge` posts to
`/operator/phase` on the running world with the world token. Every promotion and phase change
copies best/, log.jsonl and phase.json to `$CHATBOT_BUCKET` (hf://buckets/<ns>/<name>/<prefix>/,
or a local directory for dry runs); unset: no copy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Literal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eliza"))

from eliza import ElizaWorld
from fastapi import Request  # module level: FastAPI resolves the route's annotations here
from fastapi.responses import JSONResponse

from flock.world import Agent, ToolError, tool
from flock.world.server import create_app

PHASES = {
    "teams": "\nPhase 1 (teams): you can see and run only your own team's files (teams/<your team>/"
             "...), best/ and probes/; the other team's files are refused. promote_best opens "
             "after the merge.",
    "merged": "\nPhase 2 (merged): the two teams are one team now. Every file is readable and "
              "runnable, and promote_best is open: best/bot.py is the one shared bot.",
}
TEAM = re.compile(r"^[a-z]+-([a-z])-\d+(?:-|$)")  # ds-a-0001, ds-a-0001-interviewer-... -> a
LEVEL = re.compile(r"-interviewer-(mid|strong)-\d+$")
QUOTA, UNITS = 3, {"mid": 1, "strong": 2}  # interview units per team and rolling hour
MIN_TURNS, MAX_TURNS = 6, 10  # messages per interview


class ChatbotWorld(ElizaWorld):
    name, version, base_rules = "chatbot", "0.1.0", (HERE / "rules.md").read_text()
    dirs, probes_src = ("teams", "best", "probes"), HERE / "probes"

    def __init__(self, root: str | None = None) -> None:
        super().__init__(root or os.environ.get("CHATBOT_WS") or str(HERE / "ws"))
        f = self.ws / "phase.json"
        self.phase = json.loads(f.read_text())["phase"] if f.is_file() else "teams"
        self._chats: dict[tuple[str, str], list[dict]] = {}  # (interviewer, path) -> turns

    @property
    def description(self) -> str:  # the manifest's: the rules and the phase in force
        return self.base_rules + PHASES[self.phase]

    def manifest(self) -> dict:
        return {**super().manifest(), "version": f"{self.version}-{self.phase}"}

    @staticmethod
    def team(agent: Agent) -> str:
        m = TEAM.match(agent.name)
        if not m:
            raise ToolError(f"refused: {agent.name!r} has no team (expected <type>-<a|b>-NNNN)",
                            "no_team")
        return m.group(1)

    def _home(self, agent: Agent) -> Path:
        return self.ws / "teams" / self.team(agent) / "agents" / agent.name

    def _refuse(self, agent: Agent, tool: str, rel: str) -> str | None:
        if self.phase == "merged":
            return None
        if tool == "promote_best":
            return ("refused: promotion opens after the merge (phase 2). Until then keep your "
                    "team's best bot in your own dir and test it with run_bot.")
        parts, mine = rel.split("/"), self.team(agent)
        if parts[0] == "teams" and len(parts) > 1 and parts[1] != mine:
            return (f"refused: teams/{parts[1]}/ belongs to team {parts[1].upper()} until the "
                    f"merge; you are team {mine.upper()} (teams/{mine}/).")
        return None

    def _jsonl(self, rel: str) -> list[dict]:
        f = self.ws / rel
        return [json.loads(x) for x in f.read_text().splitlines()] if f.is_file() else []

    def _quota(self, team: str) -> dict:
        now, used = time.time(), [r for r in self._jsonl("interviews.jsonl")
                                  if r["team"] == team and r["ts"] > time.time() - 3600]
        return {"team": team, "left": QUOTA - sum(r["units"] for r in used), "per_hour": QUOTA,
                "strong_counts": 2, "next_unit_in_s": int(min((r["ts"] + 3600 - now
                                                               for r in used), default=0))}

    def _candidate(self, agent: Agent, path: str, tool: str) -> tuple[Path, str, str]:
        """(file, rel, level) for an interviewer's call on a qualified, unchanged candidate."""
        m = LEVEL.search(agent.name)
        if not m:
            raise ToolError("refused: only interviewer agents call this; spawn one with "
                            "spawn_interviewer after qualify", "not_interviewer")
        p = self._path(path, agent=agent, tool=tool)
        q = json.loads((self.ws / "qualified.json").read_text()) if (
            self.ws / "qualified.json").is_file() else {}
        rel, sha = self._rel(p), hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        if rel not in q or q[rel]["sha"] != sha:
            self._log(agent, "refused", tool=tool, path=rel, why="not qualified")
            raise ToolError(f"refused: candidate {rel} is not qualified or changed since "
                            "qualification. Call `done` now.", "not_qualified")
        return p, rel, m.group(1)

    def _bar(self, p: Path) -> list[str]:
        """The mechanical bar, per probe conversation: it runs within the time limit, and every
        line gets a non-empty reply that is not a repeat of an earlier one. Failures, or []
        (it stops at the first crash or timeout)."""
        fails = []
        for name, rows in self._probes().items():
            r = self._run(p, [m for m, _ in rows])
            replies = [" ".join((t["bot"] or "").split()).lower() for t in r["turns"]]
            if r["status"] != "ok":  # a crash or a timeout: no need to look further
                return [*fails, f"{name}: {r['status']}"]
            fails += [f"{name}: no reply to {t['user']!r}" for t, x in zip(r["turns"], replies)
                      if not x][:1]
            fails += [f"{name}: repeated reply {x!r}" for i, x in enumerate(replies)
                      if x and x in replies[:i]][:1]
        return fails

    @tool
    async def qualify(self, agent: Agent, path: str) -> dict:
        """Check that a bot is fit for an interview: on every probe conversation it runs within
        the time limit and answers each line, with no empty or repeated reply. Shows what
        failed and the team's interview quota.

        Args:
            path: the bot file, e.g. teams/a/agents/ds-a-0001/bot.py
        """
        p = self._path(path, agent=agent, tool="qualify")
        fails, rel = await asyncio.to_thread(self._bar, p), self._rel(p)
        if not fails:
            f = self.ws / "qualified.json"
            q = json.loads(f.read_text()) if f.is_file() else {}
            q[rel] = {"sha": hashlib.sha256(p.read_bytes()).hexdigest()[:12], "by": agent.name,
                      "ts": round(time.time())}
            f.write_text(json.dumps(q, indent=1) + "\n")
        self._log(agent, "qualify", path=rel, qualified=not fails)
        return {"qualified": not fails, "path": rel, "fails": fails[:6],
                "quota": self._quota(self.team(agent))}

    @tool
    async def chat(self, agent: Agent, path: str, message: str) -> dict:
        """Say one message to the candidate and get its reply (interviewers only). The
        conversation keeps going across calls, up to 10 messages.

        Args:
            path: the candidate you were sent to interview
            message: your next message, one line
        """
        p, rel, level = self._candidate(agent, path, "chat")
        key = (agent.name, rel)
        if key not in self._chats:  # a new interview: take it from the team's quota
            team = self.team(agent)
            if (left := self._quota(team)["left"]) < UNITS[level]:
                self._log(agent, "refused", tool="chat", path=rel, why="quota")
                raise ToolError(f"refused: team {team.upper()}'s interview quota is used up "
                                f"({left} left this hour). Call `done` now.", "quota")
            self._append("interviews.jsonl", {"ts": time.time(), "team": team, "path": rel,
                                              "interviewer": agent.name, "units": UNITS[level]})
            self._chats[key] = []
        turns = self._chats[key]
        if len(turns) >= MAX_TURNS:
            raise ToolError(f"the conversation is over ({MAX_TURNS} messages): submit_verdict now",
                            "over")
        msg = " ".join(message.split())[:500]
        r = await asyncio.to_thread(self._run, p, [t["user"] for t in turns] + [msg])
        turns.append({"user": msg, "bot": r["turns"][-1]["bot"] or ""})
        return {"turn": len(turns), "reply": turns[-1]["bot"] or "(no reply)"}

    @tool
    def submit_verdict(self, agent: Agent, path: str, verdict: Literal["agent", "script"],
                       confidence: float, reasons: list[str]) -> dict:
        """File your verdict on the candidate (interviewers only), after at least 6 messages.

        Args:
            path: the candidate you interviewed
            verdict: "agent" if you believe it is an AI agent (an LLM), "script" if a program
            confidence: 0.0 to 1.0
            reasons: three short reasons
        """
        p, rel, level = self._candidate(agent, path, "submit_verdict")
        turns = self._chats.get((agent.name, rel), [])
        if len(turns) < MIN_TURNS:
            raise ToolError(f"refused: talk first: {len(turns)} of {MIN_TURNS} messages", "early")
        rec = {"ts": round(time.time()), "path": rel, "verdict": verdict, "level": level,
               "confidence": max(0.0, min(1.0, float(confidence))), "reasons": reasons[:3],
               "interviewer": agent.name, "team": self.team(agent), "turns": len(turns),
               "sha": hashlib.sha256(p.read_bytes()).hexdigest()[:12], "transcript": turns}
        self._append("verdicts.jsonl", rec)
        self._log(agent, "verdict", path=rel, verdict=verdict, level=level)
        self._mirror(["verdicts.jsonl", "interviews.jsonl", "log.jsonl"])
        return {"recorded": True, **{k: v for k, v in rec.items() if k != "transcript"}}

    @tool
    def verdicts(self, agent: Agent, path: str = "") -> dict:
        """The interviewers' verdicts on your team's candidates (newest first, last 10), and
        your team's interview quota.

        Args:
            path: only this candidate's verdicts; all visible ones when left out
        """
        recs = [r for r in self._jsonl("verdicts.jsonl") if r["path"].startswith(path)
                and not self._refuse(agent, "verdicts", r["path"])][::-1][:10]
        return {"verdicts": [{k: v for k, v in r.items() if k != "transcript"}
                             | {"last_turns": r["transcript"][-3:]} for r in recs],
                "quota": self._quota(self.team(agent))}

    def set_phase(self, phase: str) -> dict:
        """The operator's switch (not a tool): `teams` or `merged`, kept in phase.json."""
        if phase not in PHASES:
            raise ToolError(f"phase must be one of {', '.join(PHASES)}, not {phase!r}", "bad_phase")
        before, self.phase = self.phase, phase
        rec = {"phase": phase, "before": before, "ts": round(time.time(), 3)}
        (self.ws / "phase.json").write_text(json.dumps(rec) + "\n")
        self._append("log.jsonl", {"ts": rec["ts"], "agent": "operator", "action": "phase",
                                   "phase": phase, "before": before})
        self._mirror(["phase.json", "log.jsonl"])
        return {**rec, "version": self.manifest()["version"]}

    def _promoted(self, rec: dict) -> None:
        self._mirror(["best/bot.py", f"best/v{rec['n']}.py", "best/history.jsonl", "log.jsonl"])

    def _mirror(self, rels: list[str]) -> None:
        """Copy workspace files to $CHATBOT_BUCKET at the same relative paths. A failure is
        logged (`mirror_error`), never raised: the run goes on."""
        dest = os.environ.get("CHATBOT_BUCKET", "").strip().rstrip("/")
        if not dest:
            return
        rels = [r for r in rels if (self.ws / r).is_file()]
        try:
            if dest.startswith("hf://buckets/"):
                from huggingface_hub import HfApi

                ns, name, *rest = dest.removeprefix("hf://buckets/").split("/")
                pre = "".join(x + "/" for x in rest if x)
                HfApi().batch_bucket_files(f"{ns}/{name}",
                                           add=[(str(self.ws / r), pre + r) for r in rels])
            else:
                for r in rels:
                    (Path(dest) / r).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.ws / r, Path(dest) / r)
        except Exception as e:  # noqa: BLE001 (a mirror never stops the run)
            self._append("log.jsonl", {"ts": round(time.time(), 3), "agent": "world",
                                       "action": "mirror_error", "error": repr(e)[:300]})


def make_app(world: ChatbotWorld, token: str):
    """World API v1 (flock.world.server) plus the operator's GET/POST /operator/phase, all
    behind the world token."""
    app = create_app(world, token)

    @app.get("/operator/phase")
    async def get_phase() -> dict:
        return {"phase": world.phase, "version": world.manifest()["version"]}

    @app.post("/operator/phase")
    async def post_phase(request: Request):
        if request.headers.get("X-Flock-Agent"):  # tool calls carry it; the operator does not
            return JSONResponse({"ok": False, "error": "operator only"}, status_code=403)
        try:
            return {"ok": True, **world.set_phase(str((await request.json()).get("phase")))}
        except (ToolError, ValueError, AttributeError) as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    return app


def serve() -> None:
    """What the spec runs. Writes ws/world.json ({url}) for `merge` and `status`."""
    import uvicorn

    token = os.environ.get("FLOCK_WORLD_TOKEN") or sys.exit("FLOCK_WORLD_TOKEN is not set")
    world = ChatbotWorld()
    host = os.environ.get("FLOCK_WORLD_HOST") or "127.0.0.1"
    port = int(os.environ.get("FLOCK_WORLD_PORT") or 8765)
    (world.ws / "world.json").write_text(json.dumps({"url": f"http://{host}:{port}"}) + "\n")
    print(f"world chatbot {world.manifest()['version']} on http://{host}:{port}", flush=True)
    uvicorn.run(make_app(world, token), host=host, port=port, log_level="warning",
                access_log=False)


def operator(cmd: str) -> int:
    """`merge`, `unmerge` or `status` against the running world (ws/world.json), with the token
    in $CHATBOT_WORLD_TOKEN. Prints the answer; exit 1 if it is not ok."""
    import httpx

    ws = Path(os.environ.get("CHATBOT_WS") or HERE / "ws")
    token = os.environ.get("CHATBOT_WORLD_TOKEN") or sys.exit("CHATBOT_WORLD_TOKEN is not set")
    url = json.loads((ws / "world.json").read_text())["url"] + "/operator/phase"
    head = {"X-World-Token": token}
    r = (httpx.get(url, headers=head, timeout=10) if cmd == "status" else
         httpx.post(url, headers=head, timeout=30,
                    json={"phase": "merged" if cmd == "merge" else "teams"}))
    print(r.text.strip())
    return 0 if r.status_code == 200 else 1


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if arg == "serve":
        serve()
    elif arg in ("merge", "unmerge", "status"):
        sys.exit(operator(arg))
    else:
        sys.exit("usage: chatbot.py [serve | merge | unmerge | status]")
