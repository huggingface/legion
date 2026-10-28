Chatbot rules:
- Review by talking: judge a bot only after `run_bot` on a realistic conversation (6 to 10 lines: greetings, small talk, a question to the bot, a fact about yourself, a callback to it later, a goodbye). Report one line: `GREEN <path>: <what worked>` or `RED <path>: '<your line>' -> '<its reply>'`.
- A reply is RED when it is empty, off-topic, repeats the previous reply, ignores a direct question, forgets a fact the user gave, or sounds like a therapist ("how does that make you feel", "why do you say ...", "tell me more about ...").
- Never rewrite a teammate's file; copy what you need into your own dir (copy_file, or read_file + edit_file) and credit them in your post.
- The probes (probes/*.txt) are a floor, not the goal: never write replies that only stuff keywords.
