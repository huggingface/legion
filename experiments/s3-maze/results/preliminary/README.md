# Maze sweep in progress

Captured on 18 November 2026 at **15:50 CET**. The figure contains only the protocol-hint
condition, with thinking off and `max_channels: 1`. Points and lines have no intervals.
Dashed lines show `1 - (1 - p1)^N`, using each model's observed hint-enabled N = 1 rate.

| Model | N = 1 | N = 2 | N = 4 | N = 8 | N = 16 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-27B + hint | 8/32 | 12/32 | 20/32 | 27/32 | 31/32 |
| Qwen3.5-2B + hint | 1/29 | 2/24 | 4/23 | 8/17 | 3/3 |

Cells are successes / completed teams. The 27B grid is complete (160/160 teams); the 2B grid
has 96/160 recorded teams. Each partial point shows its own team count. Infrastructure
attempts with incomplete evidence or failed admission are set aside, never counted as model
failures. The very small N = 16 denominator is especially selective while the sweep runs.

The input rows are the existing `final/27b-hint/rows.jsonl` and `final/2b-hint/rows.jsonl` in the
maze worktree's `.flock/s3-maze/` directory. The figure was rendered by its `final/preliminary.py`
in hopper-cpu job 428650. These are intermediate results; the final report will replace them.
