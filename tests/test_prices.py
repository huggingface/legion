from __future__ import annotations

import json

import pytest

from legion.controller import prices
from legion.controller.store import agent_tokens
from legion.models import TokenUsage

BIG, MID, SMALL = prices.Price(4.0, 20.0, 0.2, 5.0), prices.Price(2.0, 10.0, 0.2, 2.5), \
    prices.Price(0.10, 0.50, 0.01, 0.125)
TABLE = {**prices.PRICES, "big-model": BIG, "mid-model": MID, "small-model": SMALL}


def test_cost_by_model():
    tokens = {
        "big-model": TokenUsage(
            input=1_000_000, output=1_000_000, cache_read=1_000_000, cache_write=1_000_000
        ),
        "small-model": TokenUsage(input=2_000_000),
    }
    expected = BIG.input + BIG.output + BIG.cache_read + BIG.cache_write + 2 * SMALL.input
    assert prices.cost_usd(tokens, TABLE) == pytest.approx(expected)


def test_override_file(tmp_path, monkeypatch):
    path = tmp_path / "prices.json"
    path.write_text(
        json.dumps({"tiny": {"input": 1, "output": 2, "cache_read": 0, "cache_write": 0}})
    )
    monkeypatch.setenv("LEGION_PRICES", str(path))
    prices.price_table.cache_clear()
    try:
        assert prices.cost_usd({"tiny": TokenUsage(input=1_000_000, output=500_000)}) == 2.0
    finally:
        monkeypatch.delenv("LEGION_PRICES")
        prices.price_table.cache_clear()


def test_token_counter_reset_is_folded_into_base(store):
    store.insert_agent("a-0000", "s", "a", 0, 0)
    m = "big-model"
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


def test_provider_qualified_ids():
    """A harness may report `provider/model`: priced as the model, not at the default row."""
    u = TokenUsage(input=1_000_000)
    assert prices.cost_usd({"acme/small-model": u}, TABLE) == prices.cost_usd(
        {"small-model": u}, TABLE) == pytest.approx(0.10)
    # prefix rows apply after the provider too; the full id is tried first
    assert prices.price_for("acme/mid-model-20261001", TABLE) == MID
    assert prices.price_for("x/mystery", TABLE) == prices.PRICES["default"]
    full = prices.Price(9, 9, 9, 9)
    assert prices.price_for("acme/small-model", {**TABLE, "acme/small-model": full}) == full


def test_pinned_price_wins_for_the_qualified_id():
    """A provider price pinned under the bare model id (type_price_table) wins over the
    global row and the default, also when the model id itself contains '/'."""
    pinned = prices.Price(0.2, 0.8, 0.02, 0.2)
    table = {**TABLE, "small-model": pinned, "Qwen/Qwen3-Coder": pinned}
    assert prices.price_for("legion/small-model", table) == pinned
    assert prices.price_for("legion/Qwen/Qwen3-Coder", table) == pinned
    assert prices.price_for("Qwen/Qwen3-Coder", table) == pinned
