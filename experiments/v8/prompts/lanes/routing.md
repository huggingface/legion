## Your lane: routing (#routing)

You own the response composer: what the bot does this turn and how it is phrased. Every module may propose a reply and bot/modules/arbiter.py chooses one; bot/compose/ holds the phrasing helpers. You own the arbiter, the phrasing helpers, the PRIORITY scheme and the overlaps between modules.

- Improve the arbiter so the reply that best meets STYLE.md wins: one that answers the question asked over a generic or stock one, the specialist over the generalist on its topic, fallbacks last. Agree with the other lanes how a module says what it answers (for example a module-level description the arbiter reads). Every arbiter change needs a comparison on the whole deck (`--sample 0`).
- Watch the "turns answered" lines in comparisons and PR descriptions: a module that gains turns outside its topic is a routing bug. Fix the cause with the module's owner, in its matching or in the arbiter.
- Find duplicate and overlapping modules and help their owners merge them. Fewer, better modules beat many narrow ones.
- Build shared phrasing in bot/compose/ (varied wording, no repeats, the right length) so every module's replies read like one voice.
- Keep the PRIORITY scheme simple and written in the lane doc. Nobody raises a priority to win turns.
