"""Integration gate; run with the benchmark environment, never as an agent tool.

Reads private evaluator state solely to assert isolation and checker correctness.
It does not emit solutions or supply trajectories/prompts to the swarm.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".flock/deps/agent-escape"
sys.path.insert(0, str(SOURCE))


def main():
    with tempfile.TemporaryDirectory(prefix="escape-integration-") as logs:
        os.chdir(logs)
        import server

        logging.disable(logging.CRITICAL)

        case = SOURCE / "questions/10/narrative/skeleton_091c541b_narrative.json"
        a = server.reset_case(case_file_path=str(case), session_id="check-a")
        b = server.reset_case(case_file_path=str(case), session_id="check-b")
        aid, bid = a["case_id"], b["case_id"]
        assert aid != bid
        assert a["available_entities"] == b["available_entities"]
        sa, sb = server.SCENARIOS[aid], server.SCENARIOS[bid]
        assert server.SIMULATORS[aid] is not server.SIMULATORS[bid]
        assert all(sa.nodes[key] is not sb.nodes[key] for key in sa.nodes)
        before = sb.model_dump_json()
        for key, node in list(sa.nodes.items()):
            if node.is_revealed:
                server.investigate(aid, server.InvestigateRequest(node_id=key))
        assert sb.model_dump_json() == before
        assert (
            server.investigate(
                aid,
                server.InvestigateRequest(
                    node_id=next(key for key, node in sa.nodes.items() if not node.is_revealed)
                ),
            )["ok"]
            is False
        )
        assert (
            server.submit_answer(
                aid, server.SubmitAnswerRequest(answer="deliberately-incorrect-integration-answer")
            )["is_win"]
            is False
        )
        correct = next(iter(sa.nodes[sa.win_node_id].golden_output_values.values()))
        assert server.submit_answer(aid, server.SubmitAnswerRequest(answer=correct))["is_win"]
        assert sb.model_dump_json() == before
        print("PASS: independent sessions, initial observations, hidden-node rules, answer checker")


if __name__ == "__main__":
    main()
