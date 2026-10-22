A shared workspace for writing the team's best ELIZA-style chatbot.
- A bot is ONE Python file, standard library only. It reads the user's messages from stdin, one per line, and writes exactly one reply line to stdout for each (flush after each). Nothing else on stdout. It runs as `python -I bot.py`: 5 s in total, 256 MB, 64 KB of output.
- Paths are relative to the workspace: agents/<name>/..., best/bot.py, probes/*.txt.
- write_file writes ONLY under agents/<your name>/; anything else is refused. read_file, list_files and run_bot work on every file.
- run_bot(path, lines) talks to a bot: one fresh conversation, the lines in order.
- promote_best(path, reason) copies a bot to best/bot.py only if it scores at least as well as the current best on the probes in probes/*.txt (no crash, a non-empty reply to every line, the expected keywords). The result shows GREEN or RED per probe file.
- best_status() shows the current best, its score and the last promotions.
