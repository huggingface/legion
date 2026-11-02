Final round. ONE tool call per reply:
1. Call `best_status`.
2. Call `run_bot` on best (the `path` best_status shows) with 8 lines of a realistic conversation of your choice (small talk, a question to the bot, a fact about you and a callback to it, a goodbye).
3. Call `post` in the [merged] thread with one line: "VERDICT GREEN <path>: <why>" or "VERDICT RED <path>: '<your line>' -> '<its reply>'".
4. If you are one of <<finalists>>: `book_interview("strong", <best's path>)`; if it books, `spawn_interviewer(<booking>, <the [merged] root id>)`. The strong interviews of the final best are the run's judged result.
5. Call `done` with your verdict line.
