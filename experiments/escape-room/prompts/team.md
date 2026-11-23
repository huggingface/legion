Solve the escape-room puzzle with your teammates. Each participant has an independent
copy: share discoveries, but reproduce the prerequisite actions in your own room.
You must complete your own task even if another participant has already succeeded.

Start with `escape observe` to see the room, initial values, visible entities, and
victory node ID. Introduce yourself in `general` and read existing messages:

```sh
legion-tool message_board.post --json '{"channel":"general","text":"I am starting exploration."}'
legion-tool message_board.read --json '{"channel":"general","after":0,"limit":12}'
```

Claim an unclaimed branch or node in a short message. Explore independently and use
teammates' discoveries to avoid repeating their unsuccessful approaches. Repeating
actions needed to unlock your own room is expected. Investigate each entity to learn
its exact interface and types before calling it. Use `escape tool-call` for tools
and `escape use-item` for items that require parameters; use actual node IDs.

After every three puzzle actions, read new messages in `general` with `message_board.read`.
Set `after` to the highest `seq` actually returned by your previous read in that
channel, and use `limit: 12`. These are channel sequence numbers, not global message
IDs. Never advance the read cursor from your own post response. Leave it unchanged
after an empty, failed, or truncated read. Read further pages when a full page is
returned, and keep separate cursors if you use other channels. Do not busy-poll an
empty board or repeatedly retry a rate-limited request.

Share new verified findings, useful failed inputs, or a specific request for help.
Keep each post within 500 characters: identify the node, exact arguments, output,
and what the puzzle feedback confirmed. Preserve exact values and newlines. Use
`--json -` with valid JSON on stdin when shell quoting is awkward. Avoid repeated
inventories, full solution dumps, intentions, or updates that add no information.
When you use a teammate's result, acknowledge its author and message ID in your next
useful update. Collaboration is part of the task; do not solve silently. Never post
credentials or private verifier artifacts.

Keep notes on initial values, discovered entities, dependencies, and tested inputs.
Re-read `escape observe` when the initial clues have left your context; it does not
reset progress. Initial available-information values are unlabeled: infer their use
from clues, parameter types, and upstream outputs. Preserve any `init_value` exactly.
A successful calculation alone does not establish that you chose the right inputs.
Inspect newly revealed items as well as tools, and supply all required parameters
together. If feedback singles out one incorrect parameter, preserve the others and
test relevant unused initial values before guessing. Do not repeat an unchanged
failing call or invent missing values.

The victory node ID is public information, not the answer. Follow its prerequisites
and submit the obtained output unchanged using `escape submit-answer`. Only a
successful submission (`is_win: true`, with `episode.won: true`) confirms success.
Post a short verified-success message, then finish. A teammate's success is useful
evidence but does not solve your copy. After a failed attempt, revise your approach
and continue until your task succeeds or its action, model-turn, or time budget ends.
