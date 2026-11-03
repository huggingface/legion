#!/usr/bin/env bash
# The phase switch (README.md "Phase switch"): the world first, then the phase-2 prompts.
# Runs inside the job (bootstrap.sh calls it on the control marker or at CHATBOT_SWITCH_AT) or on
# the box that runs `flock.controller.localrun`, from any directory. Needs CHATBOT_WORLD_TOKEN (and CHATBOT_WS if
# the run set it), the rendered *.local.yaml specs, and `flock` on PATH.
set -euo pipefail
EXP=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
"$PY" "$EXP/../../worlds/chatbot/chatbot.py" merge          # 1. world: phase -> merged
if [ "${CHATBOT_VARIANT:-one}" = two-channels ]; then        # 2. prompts: phase 2
  cd "$EXP/variant-two-channels"
  FLOCK_CURRENT=.flock-a/current flock update phase2/spec-a.local.yaml
  FLOCK_CURRENT=.flock-b/current flock update phase2/spec-b.local.yaml
else
  cd "$EXP"
  flock update phase2/spec.local.yaml
fi
