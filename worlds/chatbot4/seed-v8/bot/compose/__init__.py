"""Phrasing helpers for the response composer. The routing lane owns the composer: what to do
this turn (bot/modules/arbiter.py chooses among the modules' replies) and how it is phrased.
"""
import random


def pick(options, memory):
    """One of options, avoiding any reply the bot gave in its last five turns."""
    turns = memory.get("turns") or []
    recent = {t.get("bot") for t in turns[-5:]}
    fresh = [o for o in options if o not in recent] or list(options)
    return random.Random(len(turns)).choice(fresh)
