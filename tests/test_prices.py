from __future__ import annotations

import json

import pytest

from flock.controller import prices
from flock.controller.store import agent_tokens
from flock.models import TokenUsage


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
    monkeypatch.setenv("FLOCK_PRICES", str(path))
    prices.price_table.cache_clear()
    try:
        assert prices.cost_usd({"tiny": TokenUsage(input=1_000_000, output=500_000)}) == 2.0
    finally:
        monkeypatch.delenv("FLOCK_PRICES")
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


def test_haiku_5_5_matches_claude_code_estimate():
    # resume-gate job A: Claude Code reported total_cost_usd 0.001143 for this usage
    u = TokenUsage(input=4, output=196, cache_read=28_030, cache_write=6_111)
    assert prices.cost_usd({"claude-haiku-5-5": u}) == pytest.approx(0.001143, abs=2e-6)
    # local e2e smoke turn: 0.001034465
    u = TokenUsage(input=1206, output=265, cache_read=25_374, cache_write=4_221)
    assert prices.cost_usd({"claude-haiku-5-5": u}) == pytest.approx(0.0010345, abs=2e-6)


def test_provider_qualified_ids():
    """A harness may report `provider/model` (review 2026-10-16 finding 3: 1M input tokens of
    anthropic/claude-haiku-5-5 were billed $15 at the default row instead of $0.10)."""
    u = TokenUsage(input=1_000_000)
    assert prices.cost_usd({"anthropic/claude-haiku-5-5": u}) == prices.cost_usd(
        {"claude-haiku-5-5": u}) == pytest.approx(0.10)
    # prefix rows apply after the provider too; the full id is tried first
    assert prices.price_for("anthropic/claude-sonnet-5-5-20261001") == prices.PRICES["claude-sonnet-5-5"]
    assert prices.price_for("x/mystery") == prices.PRICES["default"]
    full = prices.Price(9, 9, 9, 9)
    assert prices.price_for("anthropic/claude-haiku-5-5", {**prices.PRICES, "anthropic/claude-haiku-5-5": full}) == full


def test_pinned_price_wins_for_the_qualified_id():
    """A provider price pinned under the bare model id (PR A's type_price_table) wins over the
    global row and the default, also when the model id itself contains '/'."""
    pinned = prices.Price(0.2, 0.8, 0.02, 0.2)
    table = {**prices.PRICES, "claude-haiku-5-5": pinned, "Qwen/Qwen3-Coder": pinned}
    assert prices.price_for("flock/claude-haiku-5-5", table) == pinned
    assert prices.price_for("flock/Qwen/Qwen3-Coder", table) == pinned
    assert prices.price_for("Qwen/Qwen3-Coder", table) == pinned
