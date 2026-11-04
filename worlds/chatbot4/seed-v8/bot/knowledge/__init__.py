"""The knowledge base: facts about the world, kept as data. The answers lane owns it.

One module per domain (bot/knowledge/cooking.py, bot/knowledge/travel.py, ...) holds plain data
(dicts, lists, tuples) keyed by meaning (a dish, a city, a symptom), not by how a question is
worded, and small functions to look it up. Answer modules in bot/modules/ find what was asked
in memory["understood"], look it up here and compose the reply. The bot cannot read files, so
the data lives in .py files.
"""
