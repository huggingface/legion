# AgentEscapeBench experiment results

Run date: 2026-10-28. All attempts used the existing local Qwen3-14B server and four
workers with separate sessions of the same 10-node puzzle. No paid model APIs or
additional compute jobs were launched.

## Final result

The final configuration (`2b31423c3eb44471800429190469dfda`) solved the instance:
`worker-0002` received upstream `is_win: true` **150.0 seconds** after launch.
All four independent sessions initialized. The run made 54 benchmark actions
across the swarm, and Flock exited cleanly after stopping on the first success.

The persisted board contains 10 messages: one host kickoff, four introductions,
and five substantive findings covering tool outputs and newly revealed entities.
The agents read the shared thread. This verifies that communication works during
a successful run; it does not prove that sharing caused or accelerated success.

The final board is at
`hf://buckets/HuggingFaceSwarms/Experiments/flock/runs/agent-escape/76018d0122784e628040c925297c8a81/board`.
Its remote WAL was empty, a fresh downloaded database passed SQLite integrity,
and shutdown verification completed at `2026-10-28T17:16:58.928284+00:00`.
The owned mount and mounter were absent afterward.

## Development attempts

| Attempt ID | Configuration / finding | Result | Time to verified win |
|---|---|---|---|
| `465f8939bcd34739aca5dab0514565dc` | Non-reasoning; adapter accidentally omitted the public victory-node ID | Unsolved; invalid integration baseline | — |
| `18cd54ff5ffb47028c4de698274d1761` | Public clue restored; non-reasoning, 1,024 output tokens | Unsolved; workers repeated inputs and stopped | — |
| `db8bd14e1cf34c588e2289327afd14ac` | Reasoning enabled, 4,096 output tokens, shorter board reads | Solved by `worker-0003` | 148.4 s |
| `ff14498ebdc3481590821260966dc12d` | Fresh repeat; explicit wake disabling and richer result summary | Solved by `worker-0001` | 179.3 s |
| `5efcd29f80ef4c0c84cdedf6f4088b45` | Explicit initial post/read instruction; frozen adapter | Solved by `worker-0003`; board contained introductions only | 137.7 s |
| `2b31423c3eb44471800429190469dfda` | Final configuration with periodic collaboration reminders | Solved by `worker-0002`; substantive board exchanges | 150.0 s |

The first winning attempt had only the host kickoff in its independently
downloaded board. It demonstrates successful parallel attempts, **not a benefit
from collaboration**. The final launch instruction therefore explicitly requests
posting and reading between puzzle actions, and the adapter adds a generic
collaboration reminder after every three puzzle actions. Failed attempts remain available;
these development runs are not an unbiased success-rate estimate.

The verified wins come from the upstream answer checker, not from `done`, natural
language claims, or the separate integration check's synthetic submission.

## Isolation and persistence

The offline tests cover identity routing, independent progress, preserving public
clues without administrative fields, wrong answers, server failures, authentication,
rejected session overrides, and graceful/forced shutdown. The live upstream check
confirms distinct scenario and simulator objects and unchanged second-session
state after actions in the first session.

The successful `db8bd14...` board was independently downloaded from
`hf://buckets/HuggingFaceSwarms/Experiments/flock/runs/agent-escape/a0160e566ee64683b078d50b9944b1b0/board`.
The remote WAL was empty, SQLite integrity was `ok`, and the Flock shutdown
recorded verification at `2026-10-28T17:05:17.923824+00:00`.

A separate five-second deadline check (`9201477d35984ceeb0b25ac608e5f9c3`)
stopped during board startup, returned failure as intended, and left neither its
owned mount nor its mounter process running. It launched no workers and is not a
puzzle attempt. Startup cancellation did not claim a verified bucket snapshot.

## Artifacts

Local attempt directories are `.flock/agent-escape-attempts/<attempt-id>/`.
They retain frozen configuration/prompts, benchmark actions, model transcripts,
token totals, run results, and board locations. New runs also freeze the adapter
and record its hash. Raw evaluator logs and downloaded databases remain ignored
and are not committed.

The winning attempts used the same model and puzzle, but prompt and reasoning
settings changed during development. This experiment does not establish a swarm
advantage over a single agent or an official AgentEscapeBench score.

## Validation results

- Focused regression tests: **7 passed**, including per-agent reminder cadence.
- Pinned upstream engine integration check: passed.
- Repository-wide Ruff check, shell syntax, and diff whitespace checks: passed.
- Full suite: **1,050 passed, 14 skipped, 2 failed** in 19m 9s. The failures are
  the previously reproduced baseline failures
  `tests/test_chatbot_experiment.py::test_chatbot_experiment_dry_run`
  (five interview posts versus six expected) and
  `tests/test_run3_experiment.py::test_single_job_three_waves_end_to_end[standin]`
  (not all expected gate events arrived). No existing source files were changed.

The full-suite log is `.flock/escape-validation/pytest-full.log`. The focused tests
were rerun after the final adapter/launcher edits; the broader suite ran alongside
live validation. Work completed within the approved four-hour budget.
