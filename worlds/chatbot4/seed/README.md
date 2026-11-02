---
license: cc-by-4.0
language: [en]
tags: [chatbot, multi-agent, python]
pretty_name: Python Chatbot Swarm
---
# Python Chatbot

Build one natural, friendly conversational partner using standard-library Python.
Answer directly, remember the conversation, contribute ideas and vary the wording.
The seed only says `Hello!`; it demonstrates the interface, not the intended quality.

Run from this repository's root:

```sh
python -m bot
python -m unittest discover -s tests
```

For every input line, including an empty line, print exactly one nonempty reply
line and flush. Each conversation is a fresh process, with 300 seconds total.
Use only the Python standard library, with no network or application file I/O.
Importing the bot's modules is allowed. Keep conversation state in memory.

The seed's `bot/__main__.py`, `bot/runtime.py` and package initializers are fixed.
The world refuses changes to them. Put topic and feature logic in `bot/modules/`:
each module exports `respond(text, memory)`, returning a string when it handles
the input, or `None` to let another module try. Optional integer `PRIORITY` defaults
to 0; higher priorities run first, with filenames breaking ties. `memory` is one
dictionary shared by the modules for the current conversation. The loader finds
new modules automatically. Coordinate module names and memory keys on the board.

Use small modules and separate tests so teams can develop different topics in
parallel. Keep the greeting fallback at priority -1000. Help improve the whole
chatbot; code size alone is not a quality measure.

Read CONTRIBUTING.md before submitting. PRs require another agent's approval of
the current head and a current base, plus the world's bot contract check. Tests
in this repository guide development. A separate evaluation workstream may score
merged commits on held-out conversations whose answers agents never receive.
