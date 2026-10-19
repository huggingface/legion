#!/usr/bin/env bash
# Local dry run of gate.py without jobs or buckets: phase A in agent dir A, copy ONLY the two
# session files into a fresh agent dir B (workdir kept at the same path, as a restore would put
# it back), delete A, phase B. Usage: ROOT=<scratch dir> PI=<pi binary> local_dry_run.sh
set -euo pipefail
ROOT=${ROOT:?scratch dir}; HERE=$(cd "$(dirname "$0")" && pwd)
export SID_MAIN=${SID_MAIN:-$(python3 -c 'import uuid;print(uuid.uuid4())')}
export SID_CUT=${SID_CUT:-$(python3 -c 'import uuid;print(uuid.uuid4())')}
export WORKDIR=$ROOT/work/agents/probe-0001 OUT=$ROOT/out
rm -rf "$ROOT"; mkdir -p "$ROOT"
PI_CODING_AGENT_DIR=$ROOT/pi-A python3 "$HERE/gate.py" A >/dev/null
mkdir -p "$ROOT/pi-B/sessions/probe-0001"
cp "$ROOT"/pi-A/sessions/probe-0001/*_"$SID_MAIN".jsonl "$ROOT"/pi-A/sessions/probe-0001/*_"$SID_CUT".jsonl \
   "$ROOT/pi-B/sessions/probe-0001/"
rm -rf "$ROOT/pi-A"
PI_CODING_AGENT_DIR=$ROOT/pi-B python3 "$HERE/gate.py" B
