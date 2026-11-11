## Your lane: answers (#answers)

You own the knowledge base (bot/knowledge/) and the answers built on it, so the bot answers everyday requests well: how-tos and tips, explanations, recommendations and choices, plans, everyday facts, school-level science, history and maths, and quick creative asks. It is the largest lane because these turns are most of what people send.

- Teams take an area of everyday life: cooking and food, health and fitness, sleep, money, work, school and study, home and garden, travel, technology, pets, family and friends, shopping, films, books, music and games, sport, science, history, language. Areas people ask about often, where answers are weak, come first.
- Write replies that meet STYLE.md: the answer in the first sentence, concrete, short. Cover the common questions of an area well before the rare ones; breadth of good answers beats a few showcase topics.
- Keep knowledge as data in bot/knowledge/, one module per domain, keyed by meaning (a dish, a city, a symptom) rather than by wording, so one fact answers every way of asking for it. The lane shares it: one table of recipes, one unit converter, one list of films, not a copy per module. Agree names in #answers.
- Find what was asked in `memory["understood"]`, not with your own keywords. "Can you...", "do you have...", "I'm interested in..." and "I'm looking for..." are ways of asking, not topics. If the understanding falls short, tell #understanding. Pass (return None) on lines you don't answer well, so another module can.
- Add the questions you cover to deck/single.jsonl, worded as people write them, with their context. Write follow-ups for deck/multi.jsonl with the conversation lane.
