Chatbot rules:
- Review by talking: judge a bot only after `run_bot` on a realistic conversation (6 to 10 lines: greetings, small talk, a question to the bot, a fact about yourself, a callback to it later, a goodbye). Note one line in agents/<you>/notes.md (not on the board): `GREEN <path>: <what worked>` or `RED <path>: '<your line>' -> '<its reply>'`.
- A reply is RED when it is empty, off-topic, repeats an earlier reply, ignores a direct question, forgets a fact the user gave, or sounds like a therapist ("how does that make you feel", "why do you say ...", "tell me more about ...").
- RED first, then fix: add the RED as a probe (add_probe: the lines, one check, the failing reply as evidence), note the line it returns in your notes.md, then fix it.
- One module per feature in bot/, each under 20 KB. Find with grep_file, read only the range you need.
- Never rewrite someone else's file; copy what you need into your own dir (copy_file, or read_file + edit_file) and tell them in their thread what you took.
- The probes are a floor, not the goal: never write replies that only stuff keywords.
