"""Identity-bound sessions in the unmodified AgentEscapeBench HTTP server."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from flock.world import Agent, ToolError, World, tool


class EscapeWorld(World):
    name = "agent-escape"
    version = "0.1.0"
    description = (
        "Solve your independent copy of an escape-room puzzle. Start with get_observation. "
        "Share discoveries through the message board; other workers' actions do not change "
        "your environment. Only submit_answer can confirm victory."
    )

    def __init__(self, client=None, case_file=None, output=None, run_id=None, agents=None):
        self.client = client or httpx.Client(base_url=os.environ["ESCAPE_SERVER_URL"], timeout=20)
        self.case_file = case_file or os.environ["ESCAPE_CASE_FILE"]
        self.output = Path(output or os.environ["ESCAPE_OUTPUT"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or os.environ["ESCAPE_RUN_ID"]
        self.agents = int(agents if agents is not None else os.environ.get("ESCAPE_AGENTS", "4"))
        self.sessions: dict[str, dict] = {}
        self.winner: str | None = None
        self.action_counts: dict[str, int] = {}
        # Flock serializes synchronous world calls, including session initialization.

    def _record(self, agent, action, args, result):
        event = {
            "time": time.time(),
            "agent": agent.name,
            "action": action,
            "args": args,
            "result": result,
            "request_id": agent.request_id,
            "turn": agent.turn,
        }
        with (self.output / "trajectory.jsonl").open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()

    def _request(self, path, **kwargs):
        try:
            response = self.client.post(path, **kwargs)
            if response.status_code >= 500:
                raise ToolError("Benchmark server failed; this attempt cannot continue.")
            if response.status_code >= 400:
                return {"ok": False, "error": response.json().get("detail", "Invalid action")}
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolError("Benchmark server unavailable or returned invalid JSON.") from exc

    def _session(self, agent):
        if agent.name not in {f"worker-{i:04d}" for i in range(self.agents)}:
            raise ToolError("Unknown experiment worker", "unknown_agent")
        if agent.name not in self.sessions:
            result = self._request(
                "/env/reset",
                params={
                    "case_file_path": self.case_file,
                    "session_id": f"{self.run_id}-{agent.name}",
                },
            )
            if not result.get("ok") or not result.get("case_id"):
                raise ToolError("Could not initialize benchmark session")
            # Explicit allowlist: omit file paths, administrative metadata, and stored solutions.
            observation = {
                key: result[key]
                for key in (
                    "narrative",
                    "available_entities",
                    "available_information",
                    "win_node_id",
                )
                if key in result
            }
            self.sessions[agent.name] = {"case_id": result["case_id"], "observation": observation}
            self._record(agent, "initialize", {}, observation)
        return self.sessions[agent.name]

    def _action(self, agent, action, args):
        session = self._session(agent)
        if self.winner is not None:
            return {
                "swarm_solved": True,
                "winner": self.winner,
                "message": "The swarm has a verified solution. Call done.",
            }
        result = self._request(f"/env/{session['case_id']}/{action}", json=args)
        self.action_counts[agent.name] = self.action_counts.get(agent.name, 0) + 1
        if self.action_counts[agent.name] % 3 == 0 and not result.get("is_win"):
            result = {
                **result,
                "collaboration_reminder": (
                    "Collaboration checkpoint: read(n=12, since=LAST_READ_ID), updating your cursor "
                    "only from returned messages. Post only NEW verified findings, failed inputs, "
                    "or a specific request in thread 1 (max 500 characters). Preserve exact values "
                    "and acknowledge message IDs when reusing teammate findings."
                ),
            }
        self._record(agent, action, args, result)
        if action == "submit_answer" and result.get("is_win") is True:
            self.winner = agent.name
            target = self.output / "success.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"agent": agent.name, "result": result, "time": time.time()})
            )
            temporary.replace(target)
        return result

    @tool
    def get_observation(self, agent: Agent) -> dict:
        """Read the initial room description, entities, and clues; does not reset progress."""
        return self._session(agent)["observation"]

    @tool
    def investigate(self, agent: Agent, node_id: str) -> dict:
        """Inspect a visible entity to discover its interface or contents."""
        return self._action(agent, "investigate", {"node_id": node_id})

    @tool
    def tool_call(self, agent: Agent, node_id: str, arguments: dict[str, Any]) -> dict:
        """Execute an investigated Tool node with its exact named input parameters."""
        return self._action(agent, "tool_call", {"node_id": node_id, "arguments": arguments})

    @tool
    def use_item(self, agent: Agent, node_id: str, arguments: dict[str, Any]) -> dict:
        """Activate an Item node, supplying any parameters required to unlock it."""
        return self._action(agent, "use_item", {"node_id": node_id, "arguments": arguments})

    @tool
    def submit_answer(self, agent: Agent, answer: str) -> dict:
        """Submit the victory string. Only is_win=true establishes success."""
        return self._action(agent, "submit_answer", {"answer": answer})
