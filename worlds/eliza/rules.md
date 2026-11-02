A shared workspace for writing the team's best ELIZA-style chatbot.
- A bot is ONE Python file, standard library only. It reads the user's messages from stdin, one per line, and writes exactly one reply line to stdout for each (flush after each). Nothing else on stdout. It runs as `python -I bot.py`: 5 s in total, 256 MB, 64 KB of output.
- Paths are relative to the workspace: agents/<name>/..., best/bot.py, probes/*.txt.
- You change files ONLY under agents/<your name>/; anything else is refused. read_file, list_files and run_bot work on every file.
- Edit, do not rewrite: change a file with edit_file(path, old, new) (old must occur exactly once) or insert_after(path, anchor, text). Use write_file only to create a new file, and copy_file to start from another file (e.g. best/bot.py).
- A .py file must compile after every change; if it does not, the change is refused, the file keeps its previous content and you get the error with its line number.
- read_file(path, start, end) reads some lines; its first line says how many lines the file has.
- run_bot(path, lines) talks to a bot: one fresh conversation, the lines in order.
- promote_best(path, reason) copies a bot to best/bot.py only if it scores at least as well as the current best on the probes in probes/*.txt (no crash, a non-empty reply to every line, the expected keywords). The result shows GREEN or RED per probe file.
- best_status() shows the current best, its score and the last promotions.
