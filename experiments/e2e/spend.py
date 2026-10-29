"""Total spend of a run from the controller's event streams (works after agents are removed).

    python spend.py <agent-name> ...    (FLOCK_URL/TOKEN)
Sums turn_ended tokens per model, prices them with the controller's table, and counts turns.
"""

import json
import os
import sys

import httpx

from flock.controller.prices import cost_usd
from flock.models import TokenUsage


def main() -> None:
    c = httpx.Client(base_url=os.environ["FLOCK_URL"], timeout=60,
                     headers={"Authorization": f"Bearer {os.environ['FLOCK_TOKEN']}"})
    tokens: dict[str, TokenUsage] = {}
    turns = interrupted = 0
    for name in sys.argv[1:]:
        r = c.get(f"/agents/{name}/events", params={"limit": 5000})
        if r.status_code != 200:
            continue
        for e in r.json()["events"]:
            if e["type"] == "turn_interrupted":
                interrupted += 1
            if e["type"] != "turn_ended":
                continue
            turns += 1
            for model, u in e["data"]["data"].get("tokens", {}).items():
                tokens[model] = tokens.get(model, TokenUsage()) + TokenUsage(**u)
    print(json.dumps({"agents": len(sys.argv) - 1, "turns": turns, "interrupted": interrupted,
                      "tokens": {m: u.model_dump() for m, u in tokens.items()},
                      "cost_usd": cost_usd(tokens)}, indent=1))


if __name__ == "__main__":
    main()
