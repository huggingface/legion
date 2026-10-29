"""Runs the understanding engine (bot/understanding) before any module proposes a reply."""
from ..understanding import understand

PRIORITY = 10000  # observes first, so every other module reads memory["understood"]


def observe(text, memory):
    memory["understood"] = understand(text, memory)
