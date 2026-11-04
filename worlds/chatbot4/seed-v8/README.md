---
license: cc-by-4.0
language: [en]
tags: [chatbot, multi-agent, python]
pretty_name: SwarmChat
---
# Python Chatbot

One natural, helpful conversation partner written in standard-library Python, with no
machine learning. People bring it everyday questions and plans, and chat with it over many
turns; it should answer them the way a knowledgeable friend would. STYLE.md defines a good
reply and shows examples. The seed only says `Hello!`: it demonstrates the interface, not
the intended quality.

Run from this repository's root:

```sh
python -m bot                                  # talk to it: one reply per line
python -m unittest discover -s tests           # the tests
python tools/compare.py ../base . --seed 7     # compare with another checkout (CONTRIBUTING.md)
```

## Interface

For every input line, including an empty line, the bot prints exactly one nonempty reply line
and flushes. Each conversation is a fresh process, with 300 seconds in total. Use only the
Python standard library, with no network or file I/O beyond importing the bot's modules.

The seed's `bot/__main__.py`, `bot/runtime.py` and `bot/__init__.py`/`bot/modules/__init__.py`
are fixed: the world refuses changes to them. The runtime loads every module in `bot/modules/` by
itself; the shared parts of the bot live in packages next to it ("Architecture"). A module may
define any of these, all optional:

| name | what it does |
|---|---|
| `PRIORITY` | an integer (default 0): the order in which modules run, highest first |
| `observe(text, memory)` | remember something from the user's line (a name, a plan, the topic); never replies |
| `respond(text, memory)` | propose a reply (a string), or `None` to pass |
| `after(text, reply, memory)` | see the reply that was chosen, e.g. to avoid repeating it |

Every turn, all modules observe, then all modules propose a reply, each on its own copy of
`memory`. The arbiter (`bot/modules/arbiter.py`, `choose(text, memory, candidates)`) picks one
reply, and only the chosen module's changes to `memory` are kept, so write facts you always
want kept in `observe`. The runtime appends each turn to `memory["turns"]` as
`{"user", "bot", "module"}`: read it, don't change it. `runtime.turn(text, memory)` returns the
reply, the module that gave it and every candidate, for tools and tests.

Coordinate module names and `memory` keys on the board. Keep the greeting fallback at
PRIORITY -1000.

## Architecture

The bot has four parts. Each has a home and an owning lane, and modules build on them instead
of each doing everything itself:

| part | what it does | where | lane |
|---|---|---|---|
| understanding engine | what the user said and meant: clean text, the kind of line, the topic, what "it" refers to | `bot/understanding/`, stored in `memory["understood"]` by `bot/modules/understand.py` before any module replies | understanding |
| memory store | what the user and the bot have said: the user's facts, the topic, the turns | `bot/memory/` and the `memory` keys below | conversation |
| knowledge base | facts about the world, as data keyed by meaning | `bot/knowledge/`, one module per domain | answers |
| response composer | what to do this turn, and how to phrase it | the modules' `respond`, `bot/modules/arbiter.py`, `bot/compose/` | routing (every lane writes replies) |

Shared `memory` keys (add new ones here):

| key | what | written by |
|---|---|---|
| `memory["turns"]` | every earlier turn: `{"user", "bot", "module"}` | the runtime |
| `memory["understood"]` | this turn's reading of the line: `text`, `clean`, `words`, `kind` | `bot/modules/understand.py` |
| `memory["user"]` | facts the user told about themselves, e.g. `{"name": "Sam"}` (`bot.memory.user`) | the conversation lane |
| `memory["topic"]` | what the conversation is about now | the conversation lane |

The project runs for a long time and will produce a lot of code. Prefer general mechanisms that
improve many conversations at once (better understanding, a richer knowledge base, a better
composer) over rules or content written for one phrasing. When a reply falls short, find which
part failed and improve that part.

## Routing

Many modules can propose a reply to the same line; the arbiter decides which one the user
gets. The seed's arbiter takes the highest PRIORITY. Raising a module's PRIORITY to win more
turns takes them from other modules, so don't: changes to the arbiter and to priorities belong
to the routing lane, and need comparison evidence like any other change.

## The deck and comparisons

`deck/` holds realistic conversations, one JSON object per line:
`{"id", "area", "turns": [user lines]}`. `deck/single.jsonl` has one-line requests and
`deck/multi.jsonl` has conversations of several turns. The deck is how the swarm sees the
whole bot: add conversations people would really have, across all everyday topics, and keep
it free of lines written to make one module look good.

`tools/compare.py` runs two checkouts of the bot on a random sample of the deck and shows each
changed reply side by side, in random order. You judge each one by STYLE.md, then score your
verdicts against the key. Every PR shows this evidence and every reviewer checks it on their
own sample; CONTRIBUTING.md has the steps.

## Layout

| path | what |
|---|---|
| `bot/` | the bot: the fixed runtime, `bot/modules/` and the shared parts ("Architecture") |
| `deck/` | the conversations comparisons are run on |
| `tools/compare.py` | the comparison tool |
| `tests/` | unit tests; add one with every module |
| `STYLE.md` | what a good reply is, with examples |
| `CONTRIBUTING.md` | how to propose, review, merge and revert changes |
