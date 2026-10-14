"""Price table and cost derivation.

PLACEHOLDER NUMBERS: the prices below are rough placeholders, not researched or verified
list prices. Override them with a JSON file whose path is in SWARM_HARNESS_PRICES:

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

# Placeholders; see module docstring.
PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(15.0, 75.0, 1.5, 18.75),
    "claude-sonnet-5-5": Price(3.0, 15.0, 0.3, 3.75),
    "claude-haiku-5-5": Price(1.0, 5.0, 0.1, 1.25),
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
