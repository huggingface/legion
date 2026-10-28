Eliza team rules:
- Announce: before you work on something, say it in one line in its thread ("[memory] I take name recall in agents/ds-0002/bot.py").
- Review by talking: judge a bot only after `run_bot` on it with a realistic conversation. Report one line: `GREEN <path>: <what worked>` or `RED <path>: '<your line>' -> '<its reply>'`.
- Promote only with evidence: call `promote_best` only for a bot you ran with `run_bot` in this turn, and put that evidence in `reason`. Then post the result (promoted or refused, and the probe lines) in the [plan] thread.
- Never rewrite a teammate's file; copy what you need into your own dir and credit them in your post.
- Final round: everyone runs best/bot.py with `run_bot` and posts one verdict line in [plan].
