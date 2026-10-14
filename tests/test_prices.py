from __future__ import annotations

import json

import pytest

from swarm_harness.controller import prices
from swarm_harness.controller.store import agent_tokens
from swarm_harness.models import TokenUsage


def test_cost_by_model():
    tokens = {
        "claude-opus-5-5": TokenUsage(
            input=1_000_000, output=1_000_000, cache_read=1_000_000, cache_write=1_000_000
        ),
        "claude-haiku-5-5": TokenUsage(input=2_000_000),
    }
    p = prices.PRICES["claude-opus-5-5"]
    expected = (
        p.input
        + p.output
        + p.cache_read
        + p.cache_write
        + 2 * prices.PRICES["claude-haiku-5-5"].input
    )
    assert prices.cost_usd(tokens) == pytest.approx(expected)


def test_prefix_and_default():
    assert prices.price_for("claude-sonnet-5-5-20261001") == prices.PRICES["claude-sonnet-5-5"]
    assert prices.price_for("mystery-model") == prices.PRICES["default"]


def test_override_file(tmp_path, monkeypatch):
    path = tmp_path / "prices.json"
    path.write_text(
        json.dumps({"tiny": {"input": 1, "output": 2, "cache_read": 0, "cache_write": 0}})
    )
    monkeypatch.setenv("SWARM_HARNESS_PRICES", str(path))
    prices.price_table.cache_clear()
    try:
        assert prices.cost_usd({"tiny": TokenUsage(input=1_000_000, output=500_000)}) == 2.0
    finally:
        monkeypatch.delenv("SWARM_HARNESS_PRICES")
        prices.price_table.cache_clear()


def test_token_counter_reset_is_folded_into_base(store):
    store.insert_agent("a-0000", "s", "a", 0, 0)
    m = "claude-opus-5-5"
    store.record_agent_heartbeat(
        "a-0000", {}, {m: TokenUsage(input=100, output=10)}, prices.cost_usd
    )
    store.record_agent_heartbeat(
        "a-0000", {}, {m: TokenUsage(input=150, output=20)}, prices.cost_usd
    )
    # New process started counting from zero.
    store.record_agent_heartbeat("a-0000", {}, {m: TokenUsage(input=5, output=1)}, prices.cost_usd)
    row = store.get_agent("a-0000")
    assert agent_tokens(row)[m] == TokenUsage(input=155, output=21)
    assert row["cost"] == prices.cost_usd({m: TokenUsage(input=155, output=21)})
