#!/bin/bash
# The scaling figure and tables: 27B thinking off (main), 27B thinking on (lighter), 2B thinking
# off (its template's default), into $1 (default off/results)
W=/fsx/lewis/git/hf/flock/.claude/worktrees/agent-aa6ab6d95c0f6cdad
S=$W/.flock/s3-maze
OUT=${1:-$S/off/results}
cd "$W" || exit 2
.venv/bin/ruff check .flock/s3-maze/ds/flock/sweep/plot.py || exit 1
uv run --quiet .flock/s3-maze/ds/flock/sweep/plot.py \
  --series "27B, thinking off=$S/off/sweep" \
  --series "27B, thinking on=$S/sweep/27b@light" \
  --series "2B, thinking off (template default)=$S/sweep/2b" \
  --out "$OUT"
