"""Chooses each turn's reply among the modules' candidates. The routing lane owns this file.

The seed rule: the highest PRIORITY wins (candidates arrive in that order), and a fallback
(PRIORITY below 0) answers only when nothing else does. Improve the rule with
tools/compare.py evidence (README.md, "Routing"); it may read memory but not change it.
"""


def choose(text, memory, candidates):
    """Return the index of the chosen candidate.

    candidates: [{"module": name, "priority": PRIORITY, "reply": text}], highest PRIORITY
    first. memory["turns"] holds the earlier turns of the conversation.
    """
    for index, candidate in enumerate(candidates):
        if candidate["priority"] >= 0:
            return index
    return 0
