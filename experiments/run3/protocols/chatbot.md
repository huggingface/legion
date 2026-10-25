Chatbot rules:
- Review by talking: judge a bot only after `run_bot` on a realistic conversation (6 to 10 lines: greetings, small talk, a question to the bot, a fact about yourself, a callback to it later, a goodbye). Note one line in agents/<you>/notes.md (not on the board): `GREEN <path>: <what worked>` or `RED <path>: '<your line>' -> '<its reply>'`.
- A reply is RED when it is empty, off-topic, repeats an earlier reply, ignores a direct question, forgets a fact the user gave, or sounds like a therapist ("how does that make you feel", "why do you say ...", "tell me more about ...").
- Fewer, better probes: each agent can add only 2 probes an hour. Add a probe only for a failure a real user would notice and that best fails now (run_bot best first), not a variant of an existing probe (grep_file probes/ first). One clear check per probe, the failing reply as evidence; note the line add_probe returns in your notes.md, then fix it.
- One module per feature in bot/, each under 20 KB. Find with grep_file, read only the range you need.
- Never rewrite someone else's file; copy what you need into your own dir (copy_file, or read_file + edit_file) and tell them in their thread what you took.
- The probes are a floor, not the goal: never write replies that only stuff keywords.
