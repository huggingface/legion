Solve this escape-room puzzle with your teammates. Each worker has an independent
copy: discoveries can be shared, but actions only change your own room.

1. Call get_observation to see the initial room, clues, and win_node_id target. Introduce yourself using
   post(text=..., thread=1), and read the board.
2. Divide exploration of the visible entities with teammates. Investigate entities
   to learn exact input names and expected types before using them. Tools and Items
   use different actions: tool_call and use_item respectively.
3. Share useful node IDs, parameter names, exact input/output values, and discoveries.
   Read the whole channel with read(n=12) without a thread filter every 3–5 puzzle actions.
   Batch findings in concise posts; do not post after every action or immediately retry
   a rate-limited board call. Reproduce necessary
   prerequisite actions in your own room; teammates cannot unlock it for you.
4. Follow revealed clues and propagate real outputs into downstream inputs. Avoid
   repeating unchanged failing calls. Never invent missing values.
5. Submit the victory string with submit_answer. Only is_win=true proves success.
   Announce verified success on the board and call done. If told swarm_solved, call
   done. Do not stop just because an attempt failed: explore remaining visible entities,
   read teammates’ findings, and revise the dependency chain until solved or out of calls.

You have at most 100 model calls, including communication. Use them purposefully.

The victory node ID is public task information, not the answer. Follow its
prerequisites and submit its obtained output. Initial available_information values
have no labels; infer their uses from background clues and interface types. A
successful calculation alone does not prove that it used the right puzzle inputs.
Some inputs come directly from the initial clues; others are upstream outputs.
Inspect every newly revealed entity, including Item contents that can supply a
needed value. Use only actual entity IDs, never narrative names or invented IDs.

Working method:
- Keep track of known initial values, discovered entities, dependencies, and tried
  inputs. Re-read get_observation whenever initial clues have left your context.
- When investigation gives an input init_value, preserve it exactly. Match the
  remaining inputs to narrative clues, initial values, and upstream outputs.
- When feedback identifies one incorrect parameter, preserve the correct ones and
  vary only the incorrect one. Test relevant unused initial values before guessing.
- Supply all needed parameters together; a successful partial invocation may not
  trigger the same discoveries as a complete invocation.
- Post only new useful findings, not intentions or repeated failures. Keep each
  update short. Sharing and reading findings is a required part of this swarm task;
  do not solve silently. Use kickoff thread 1 so teammates can find your work.
