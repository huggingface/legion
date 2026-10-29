## Your lane: routing (#routing)

You decide which candidate reply the user gets. Every module may propose a reply; bot/modules/arbiter.py chooses one. You own the arbiter, the PRIORITY scheme and the overlaps between modules.

- Improve the arbiter so the reply that best meets STYLE.md wins: one that answers the question asked over a generic or stock one, the specialist over the generalist on its topic, fallbacks last. Agree with the other lanes how a module says what it answers (for example a module-level description the arbiter reads). Every arbiter change needs a comparison on the whole deck (`--sample 0`).
- Watch the "turns answered" lines in comparisons and PR descriptions: a module that gains turns outside its topic is a routing bug. Fix the cause with the module's owner, in its matching or in the arbiter.
- Find duplicate and overlapping modules and help their owners merge them. Fewer, better modules beat many narrow ones.
- Keep the PRIORITY scheme simple and written in the lane doc. Nobody raises a priority to win turns.
