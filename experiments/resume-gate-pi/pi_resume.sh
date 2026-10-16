#!/usr/bin/env bash
# Pi cross-machine resume gate (docs/pi.md; flock-v2 note milestone 1). No inference spend: Pi
# talks to stub_openai.py inside the same container.
#   pi_resume.sh A   two sessions in WORKDIR (one cut off mid-tool by SIGKILL), push workdir +
#                    ONLY sessions/<agent>/*_<sid>.jsonl + out to DEST
#   pi_resume.sh B   fresh container: restore workdir + session files at the same paths, resume
#                    both, check history, same file appended, stats continue
# Env: SID_MAIN, SID_CUT (required), PI_CODING_AGENT_DIR (default /state/pi), WORKDIR (default
#      /work/agents/probe-0001), DEST (default hf://buckets/cmpatino/flock/resume-gate/pi),
#      OUT (default ./out). Needs pi and hf on PATH and hf auth.
set -euo pipefail
PHASE=${1:?phase A or B}
: "${SID_MAIN:?}" "${SID_CUT:?}"
export PI_CODING_AGENT_DIR=${PI_CODING_AGENT_DIR:-/state/pi}
export WORKDIR=${WORKDIR:-/work/agents/probe-0001}
export OUT=${OUT:-$PWD/out}
DEST=${DEST:-hf://buckets/cmpatino/flock/resume-gate/pi}
AGENT=$(basename "$WORKDIR")
SESS="$PI_CODING_AGENT_DIR/sessions/$AGENT"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
echo "pi=$(pi --version) node=$(node --version) user=$(id -un)"

if [ "$PHASE" = A ]; then
  [ -e "$WORKDIR" ] && { echo "WORKDIR already exists; A must start clean" >&2; exit 2; }
  python3 "$HERE/gate.py" A
  echo "== files under PI_CODING_AGENT_DIR after A"; (cd "$PI_CODING_AGENT_DIR" && find . -type f | sort) | tee "$OUT/A.home-files.txt"
  hf buckets sync "$WORKDIR" "$DEST/workdir" --delete -q
  hf buckets sync "$SESS" "$DEST/sessions/$AGENT" --include "*_$SID_MAIN.jsonl" --include "*_$SID_CUT.jsonl" --delete -q
  hf buckets sync "$OUT" "$DEST/out-A" -q
  hf buckets list "$DEST" -R 2>/dev/null || hf buckets list "$DEST"
elif [ "$PHASE" = B ]; then
  [ -e "$WORKDIR" ] && { echo "WORKDIR already exists; B must run on a clean machine" >&2; exit 2; }
  [ -e "$PI_CODING_AGENT_DIR/sessions" ] && { echo "$PI_CODING_AGENT_DIR/sessions exists; B needs a clean agent dir" >&2; exit 2; }
  mkdir -p "$WORKDIR" "$SESS"
  hf buckets sync "$DEST/workdir" "$WORKDIR" -q
  hf buckets sync "$DEST/sessions/$AGENT" "$SESS" -q
  hf buckets sync "$DEST/out-A" "$OUT" -q   # A.json: phase A's stats, for the comparison
  echo "== restored files"; (cd "$PI_CODING_AGENT_DIR" && find . -type f | sort; cd "$WORKDIR" && find . -type f | sort) | tee "$OUT/B.restored-files.txt"
  python3 "$HERE/gate.py" B | tee "$OUT/B.verdict.json"
  hf buckets sync "$OUT" "$DEST/out-B" -q
else
  echo "phase must be A or B" >&2; exit 2
fi
