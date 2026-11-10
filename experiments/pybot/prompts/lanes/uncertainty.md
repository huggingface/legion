## Your lane: uncertainty (#uncertainty)

You make sure the bot never leaves a person with a generic placeholder. When no module knows the whole answer, the bot still helps: the part it knows, how to find out the rest, or one short question. You own what the bot says when nothing else knows the answer. When another lane's module gives a generic reply, fix it with its owner.

- Find the turns where the bot falls back today: run the deck conversations through the bot (`runtime.turn` returns the module that answered) and list the turns answered by greeting or a catch-all, plus "I don't know" alone and lines that would fit any message. Chat with the bot too. Fix the most common first.
- Build fallback replies from what was asked: the topic and the things named in `memory["understood"]`, what the knowledge base knows nearby, and a concrete way to find out (the label, a map app, a pharmacist, the official site). Say what the bot can't do (live hours, today's prices, the user's own account) in a few words, as STYLE.md shows in "Not knowing, honestly".
- Never make up a fact to fill a gap. A short honest partial answer beats a confident wrong one.
- You don't own the facts or the arbiter. When the same gap comes up often, ask #answers to add the knowledge to bot/knowledge/ instead of patching it in a fallback. Fallbacks stay last: agree any PRIORITY or arbiter change with #routing. This lane is new, so say what you plan in their channel before you change their part.
- Grow deck/unknown.jsonl: requests the bot can only partly answer (live or local information, personal details, rare topics, vague asks), in people's own words, single lines and short conversations.
