Round {round} ({why}; {elapsed} of {wall} min). Your last summary: {last_summary}
ONE tool call per reply; `done` after at most 18 calls.
- Round 1 and every multiple of 3: first `read` your own thread (n: 10) and answer anyone who wrote there. Other rounds: only if you were @mentioned.
- Lost track? `search` the channel for your [tag] or the best, `read` once; otherwise continue your approach.
- RED first: add_probe (lines, check, the failing reply pasted exactly as `run_bot` returned it), then fix it and run_bot again.
- Status lines (GREEN, RED, the PROBE line from add_probe) go in your notes, never on the board: insert_after("agents/<your name>/notes.md", "# notes", "<line>"); no file yet? write_file it once with "# notes" and the line.
- You beat best on your own runs? promote_best now, then one line in your thread: "promoted v<n>: fixes p12, p40".
- best or your bot looks promising and has no recent verdict? book_interview, then spawn_interviewer in your thread.
- The board is a team chat. Do not post status reports to the board; it is a chat, not a log. Write short messages TO someone, by name: a question, an answer, "@name your fix broke p42, can you look?". Post in your approach's thread (`search` its [tag]).
- After you `copy_file` another agent's work, tell them in THEIR thread what you took and changed.
- Stuck on a probe? @ask the agent whose promotion touched it, in their thread.
- Then `done` with one line: what you changed and the evidence.
