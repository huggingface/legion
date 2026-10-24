Round {round} ({why}; {elapsed} of {wall} min). Your last summary: {last_summary}
ONE tool call per reply; `done` after at most 18 calls.
- Lost track (or new)? `search` the channel for your thread or the best, `read` once; otherwise continue your approach.
- RED first: add_probe (lines, check, the failing reply), post its line in your thread, then fix it with edit_file and run_bot again. When you add a probe, paste the bot's failing reply exactly as `run_bot` returned it.
- You beat best on your own runs? promote_best now.
- best or your bot looks promising and has no recent verdict? book_interview, then spawn_interviewer in your thread.
- One thread per approach, not per agent: reply inside the thread of the approach you are extending; open a new thread only for a genuinely new approach.
- When you copy or build on someone's bot, say so in their thread in one line (what you changed).
- Then `done` with one line: what you changed and the evidence.
