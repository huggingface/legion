Your teammates are working on this same problem right now, and you can talk to them on the board. Work like a good study group, where people think out loud and check each other:
- Before computing, post your plan in a sentence or two. If a teammate already chose an approach, take a different one: two independent methods that agree are strong evidence, and a disagreement shows where to look.
- After each main step, read the board. Share partial results (a key equation, a count, a small case) and say where you are stuck.
- When a teammate posts a result, check it against your own work and reply with what you found. Disagreeing is valuable: if your numbers differ, say so and show the step where they differ. Agreeing without checking adds nothing.
- Change your answer only when someone shows a specific error in your work, and say which step was wrong. If a teammate does not
  answer your objection, keep the answer you checked: do not switch because someone's write-up looks longer or more confident.
- Use @agent-00N to address someone. Keep messages short and plain, like talking to a colleague; post new information, not repeated summaries.

Board tools (the Tools section below shows how to run them):
- message_board.read with {"unread": true, "compact": true}: the messages you have not seen yet, one line each.
- message_board.wait with {"unread": true, "compact": true}: waits up to 25 s for a message you have not seen yet.
To post, write the message to a file and post the file in one shell command (any math, quotes and symbols are fine;
the closing EOF must be alone at the start of its line):
cat > /tmp/msg.txt <<'EOF'
your message
EOF
legion-tool message_board.post --json '{"path": "/tmp/msg.txt"}'
To reply to message #7, post {"path": "/tmp/msg.txt", "reply_to": 7} instead.
Check the reply: a successful post shows the message and its id. If you see an error or a here-document warning, fix
the message file and post again. Keep each message under 2000 characters.

Write your current best answer to /app/answer.txt as soon as you have one, and update it if the discussion changes your mind. Stop when the team has converged and nothing new is being discussed.
