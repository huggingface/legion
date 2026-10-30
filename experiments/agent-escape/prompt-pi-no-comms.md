Solve your independent copy of this escape-room puzzle using the available puzzle tools.

1. Call get_observation to see the initial room, clues, and win_node_id target.
2. Investigate entities to learn exact input names and expected types before using them.
   Tools and Items use different actions: tool_call and use_item respectively.
3. Follow revealed clues and propagate real outputs into downstream inputs. Avoid
   repeating unchanged failing calls. Never invent missing values. Reproduce necessary
   prerequisite actions in your own room.
4. Submit the victory string with submit_answer. Only is_win=true proves success.
   If told swarm_solved, finish your response. Do not stop just because an attempt
   failed: explore remaining visible entities and revise the dependency chain until
   solved or out of calls.

You have at most 200 model calls. Use them purposefully.

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
