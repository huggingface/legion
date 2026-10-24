Round {round} ({why}; {elapsed} of {wall} min). Your last summary: {last_summary}
ONE tool call per reply; `done` after at most 18 calls.
- Round 1 and every round that is a multiple of 3 (3, 6, 9, ...): first call `read` your own thread (n: 10); if another agent posted there, answer them in one line there before your own work. Other rounds: skip that read, unless you were @mentioned: then read and answer them first.
- Lost track (or new)? `search` the channel for your thread or the best, `read` once; otherwise continue your approach.
- RED first: add_probe (lines, check, the failing reply), post its line in your thread, then fix it with edit_file and run_bot again. When you add a probe, paste the bot's failing reply exactly as `run_bot` returned it.
- You beat best on your own runs? promote_best now.
- best or your bot looks promising and has no recent verdict? book_interview, then spawn_interviewer in your thread.
- Post a RED or GREEN in the thread of the approach it belongs to: `search` its [tag], reply there (thread: <id>). No new root for a status update; a new root only for a genuinely new approach, titled [tag] what it is.
- When you `copy_file` another agent's best or module, post one line in THEIR thread: what you took and what you changed.
- Stuck on a probe? Ask in the thread of the agent whose promotion touched it, @their-name, one line. If someone @mentions you, answer them in your next round.
- Then `done` with one line: what you changed and the evidence.
