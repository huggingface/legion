"""The memory store: what the user and the bot have said. The conversation lane owns it.

Shared keys of the conversation's memory dict (list new ones in README.md):
- memory["turns"]: every earlier turn as {"user", "bot", "module"}; the runtime writes it.
- memory["understood"]: this turn's reading of the user's line (bot/understanding).
- memory["user"]: facts the user told about themselves, e.g. {"name": "Sam", "city": "Leeds"}.
- memory["topic"]: what the conversation is about now.
Write facts in observe(), so they are kept whichever reply is chosen.
"""


def user(memory):
    """The user's facts (a dict), created on first use."""
    facts = memory.get("user")
    if not isinstance(facts, dict):
        facts = memory["user"] = {}
    return facts
