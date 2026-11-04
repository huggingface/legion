"""The understanding engine: what the user said and meant. The understanding lane owns it.

bot/modules/understand.py stores understand(text, memory) in memory["understood"] before any
module proposes a reply, so every module reads one reading of the line instead of matching its
own keywords. Grow it with general mechanisms (spelling and chat-speak, what kind of line it
is, the topic and the things named, what "it" or "that" refers to). Keep the keys below
stable: add new ones and list them in README.md.
"""
import re

WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
GREETINGS = ("hi", "hello", "hey", "hiya", "morning", "good morning", "good afternoon",
             "good evening", "evening")
THANKS = ("thanks", "thank you", "thx", "cheers", "ty")
QUESTION_STARTS = {"what", "why", "how", "when", "where", "who", "which", "can", "could", "should",
                   "would", "is", "are", "do", "does", "did", "will", "have", "has"}


def understand(text, memory):
    """{"text": the line, "clean": lower case with single spaces, "words": its words,
    "kind": "empty" | "greeting" | "thanks" | "question" | "statement"}."""
    clean = " ".join(text.lower().split())
    words = WORD.findall(clean)
    bare = " ".join(words)
    if not words:
        kind = "empty"
    elif len(words) <= 4 and any(bare == g or bare.startswith(g + " ") for g in GREETINGS):
        kind = "greeting"
    elif any(bare == t or bare.startswith(t + " ") for t in THANKS):
        kind = "thanks"
    elif clean.endswith("?") or words[0] in QUESTION_STARTS:
        kind = "question"
    else:
        kind = "statement"
    return {"text": text, "clean": clean, "words": words, "kind": kind}
