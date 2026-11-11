## Your lane: understanding (#understanding)

You own the understanding engine: what the user said and meant, worked out once per turn for every module. It lives in bot/understanding/, and bot/modules/understand.py stores its result in `memory["understood"]` before any module proposes a reply.

- Read messy input the way people type it: typos, missing apostrophes, chat-speak ("u", "thx", "pls", "idk"), no punctuation, run-on lines, one-word replies. Normalise it once, so no module needs its own spelling fixes.
- Work out what kind of line it is (a question, a request, news, a feeling, small talk, thanks, a short reply to the bot's last turn) and what it is about: the topic, the things named (a dish, a city, a pet), and, with the conversation lane, what "it", "that" and "there" refer to.
- Keep the keys of `memory["understood"]` stable and listed in README.md. Add keys rather than change them, and tell the lanes that read them.
- Better understanding should improve many replies at once: measure it on the whole deck (`--sample 0`), and add messy, realistic lines to the deck.
