Round {round} ({why}; {elapsed} of {wall} min; you spent ${spend} of ${cap}). Your last summary: {last_summary}
Do the steps below in order, skipping the ones already done, ONE tool call per reply. Stop with `done` after at most 7 calls.
A. Call `read` (no arguments). Find the [plan] thread and any message that mentions you.
B. If you are not ds-0000 and you have not posted `agree` in [plan]: `post` with thread = the [plan] root id and text "agree <root id>". If there is no [plan] thread yet: `done`.
C. If a message asks you to check a bot: `run_bot` that path with 4 to 8 realistic lines about your part, then `post` in that thread: "GREEN <path>: <what worked>" or "RED <path>: '<your line>' -> '<its reply>'".
D. Your part: `read_file` agents/<your name>/bot.py (if you have none: best/bot.py; if that does not exist either, start a new file). Improve YOUR part. `write_file` agents/<your name>/bot.py with the whole file. Then `run_bot` it with 4 to 8 lines that test your part. If a reply is wrong or empty, fix it and write again.
E. When your bot is GREEN on your own run: `post` in [plan] "@ds-0000 ready: agents/<your name>/bot.py: <one line on your part>".
F. ds-0000 only: when a teammate is ready, `read_file` their bot, merge their part into agents/ds-0000/bot.py, `write_file` it, `run_bot` it, then `promote_best` it with your run_bot evidence as the reason. `post` the result in [plan] with @<author>.
G. Call `done` with a one-line summary of what you did.
