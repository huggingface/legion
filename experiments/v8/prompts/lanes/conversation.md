## Your lane: conversation (#conversation)

You make the bot good over many turns: remembering what the user said, reading follow-ups in context ("how long will it take?", "and if I...", "it", "that"), going with corrections, small talk and feelings, staying on the user's topic, and never repeating itself.

- Build the shared memory every lane uses: one agreed set of `memory` keys (the user's name, their plans, constraints such as allergies or budget, the current topic), written in `observe` and listed in the lane doc. `memory["turns"]` already holds the history.
- Help answer modules handle follow-ups, for example with a helper that turns "how long will it take?" into the full question about the current topic.
- Small talk and feelings: warm, brief and varied replies that respond to what the person said. Never the therapist pattern, never a stock line that would fit any message, and not a question at the end of every reply.
- Grow deck/multi.jsonl: realistic conversations of 3–6 turns across everyday topics, with follow-ups, corrections and returns to something said earlier. They are what protects multi-turn behaviour in every comparison.
