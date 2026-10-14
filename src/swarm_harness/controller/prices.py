"""Price table and cost derivation.

Claude rows are Anthropic first-party list prices (cached model table, 2026-10-06): input and
output as listed, cache read as listed (0.1x input unless noted), 5-minute cache write 1.25x
input. Haiku 5.5 matches Claude Code's own `total_cost_usd` (tests/test_prices.py). The Codex row
is still a placeholder. Override any row with a JSON file whose path is in SWARM_HARNESS_PRICES:

    {"claude-opus-5-5": {"input": 15, "output": 75, "cache_read": 1.5, "cache_write": 18.75}}

All prices are USD per 1M tokens. Reasoning tokens are assumed to be included in the
output count the harness reports, so they are not billed separately.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache

from swarm_harness.models import TokenUsage


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float


DEFAULT_MODEL = "default"

# See module docstring.
PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.0, 20.0, 0.2, 5.0),  # cache read listed at $0.20
    "claude-sonnet-5-5": Price(2.0, 10.0, 0.2, 2.5),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125),
    "gpt-5.5-codex": Price(1.25, 10.0, 0.125, 0.0),
    DEFAULT_MODEL: Price(15.0, 75.0, 1.5, 18.75),  # unknown models priced high, not free
}


@lru_cache(maxsize=1)
def price_table() -> dict[str, Price]:
    table = dict(PRICES)
    path = os.environ.get("SWARM_HARNESS_PRICES")
    if path:
        with open(path, encoding="utf-8") as f:
            for model, p in json.load(f).items():
                table[model] = Price(**p)
    return table


def price_for(model: str, table: dict[str, Price] | None = None) -> Price:
    """Exact match, else the longest table key that prefixes the model id, else default."""
    table = table or price_table()
    if model in table:
        return table[model]
    prefixes = [k for k in table if k != DEFAULT_MODEL and model.startswith(k)]
    return table[max(prefixes, key=len)] if prefixes else table[DEFAULT_MODEL]


def cost_usd(tokens: dict[str, TokenUsage], table: dict[str, Price] | None = None) -> float:
    total = 0.0
    for model, u in tokens.items():
        p = price_for(model, table)
        total += (
            u.input * p.input
            + u.output * p.output
            + u.cache_read * p.cache_read
            + u.cache_write * p.cache_write
        ) / 1_000_000
    return round(total, 6)
