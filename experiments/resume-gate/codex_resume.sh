#!/usr/bin/env bash
# Codex cross-machine resume gate, run locally (no OPENAI_API_KEY for jobs; the OAuth auth.json stays here).
# Two fresh CODEX_HOMEs (only auth.json + config.toml copied in) and two distinct workdirs simulate two machines.
#   codex_resume.sh A                 codeword turn in HOME_A / WD_A
#   codex_resume.sh B rollout         copy only the rollout jsonl into HOME_B, resume from WD_B
#   codex_resume.sh B rollout+sqlite  also copy HOME_A/*.sqlite* (only if the first B fails)
# Env: ROOT (scratch dir), MODEL (default gpt-6-luna), OUT (default ./out), SANDBOX (default danger-full-access:
#      workspace-write needs bwrap user namespaces, which this container and likely HF Jobs lack).
set -euo pipefail
PHASE=${1:?}; SET=${2:-rollout}
ROOT=${ROOT:?scratch dir}; OUT=${OUT:-$PWD/out}; MODEL=${MODEL:-gpt-6-luna}
SRC_HOME=${SRC_CODEX_HOME:-/home/node/local/codex-home}
HOME_A=$ROOT/codex-home-A; HOME_B=$ROOT/codex-home-B; WD_A=$ROOT/machineA/work/agents/probe-0001; WD_B=$ROOT/machineB/work/agents/probe-0001
mkdir -p "$OUT"
codex_run() { codex exec --json -m "$MODEL" -c 'model_reasoning_effort="low"' --skip-git-repo-check \
                --sandbox "${SANDBOX:-danger-full-access}" -c approval_policy="never" "$@" < /dev/null; }
fresh_home() { rm -rf "$1"; mkdir -p "$1"; cp "$SRC_HOME/auth.json" "$SRC_HOME/config.toml" "$1/"; }
summarize() { python3 - "$1" <<'PY'
import json, sys
out = {"thread_id": None, "reply": None, "usage": None, "commands": [], "errors": []}
for line in open(sys.argv[1]):
    try: ev = json.loads(line)
    except ValueError: continue
    t = ev.get("type")
    if t == "thread.started": out["thread_id"] = ev.get("thread_id")
    elif t == "turn.completed": out["usage"] = ev.get("usage")
    elif t in ("error", "turn.failed"): out["errors"].append(ev)
    elif t == "item.completed":
        it = ev.get("item", {})
        if it.get("type") == "agent_message": out["reply"] = it.get("text")
        if it.get("type") == "command_execution": out["commands"].append(it.get("command"))
print(json.dumps(out))
PY
}

if [ "$PHASE" = A ]; then
  fresh_home "$HOME_A"; rm -rf "$ROOT/machineA"; mkdir -p "$WD_A"; cd "$WD_A"
  t0=$(date +%s.%N)
  CODEX_HOME=$HOME_A codex_run "The codeword is PELICAN-7. Write it to notes.txt and reply with only the codeword." \
    > "$OUT/codex-A.stream.jsonl" 2> "$OUT/codex-A.stderr"
  echo "wall_s=$(python3 -c "import time,sys;print(round(time.time()-float(sys.argv[1]),2))" "$t0")"
  summarize "$OUT/codex-A.stream.jsonl" | tee "$OUT/codex-A.result.json"
  echo "== files in CODEX_HOME A after turn 1"; (cd "$HOME_A" && find . -type f | sort) | tee "$OUT/codex-A.home-files.txt"
else
  TID=$(python3 -c "import json;print(json.load(open('$OUT/codex-A.result.json'))['thread_id'])")
  fresh_home "$HOME_B"; rm -rf "$ROOT/machineB"; mkdir -p "$WD_B"
  cp -a "$WD_A/." "$WD_B/"            # restore the workdir (at a different path, on purpose)
  rel=$(cd "$HOME_A" && find sessions -name "rollout-*$TID.jsonl")
  mkdir -p "$HOME_B/$(dirname "$rel")"; cp "$HOME_A/$rel" "$HOME_B/$rel"
  [ "$SET" = rollout+sqlite ] && cp "$HOME_A"/*.sqlite* "$HOME_A"/session_index.jsonl "$HOME_B/" 2>/dev/null || true
  mv "$ROOT/machineA" "$ROOT/machineA.offline"   # machine A is gone: its workdir must not be readable
  echo "== restored into CODEX_HOME B ($SET)"; (cd "$HOME_B" && find . -type f | sort) | tee "$OUT/codex-B-$SET.home-files-before.txt"
  cd "$WD_B"; t0=$(date +%s.%N); rc=0
  CODEX_HOME=$HOME_B codex_run resume "$TID" "What is the codeword? Answer from memory without running any commands; reply with only the codeword." \
    > "$OUT/codex-B-$SET.stream.jsonl" 2> "$OUT/codex-B-$SET.stderr" || rc=$?
  echo "rc=$rc wall_s=$(python3 -c "import time,sys;print(round(time.time()-float(sys.argv[1]),2))" "$t0")"
  mv "$ROOT/machineA.offline" "$ROOT/machineA"
  summarize "$OUT/codex-B-$SET.stream.jsonl" | tee "$OUT/codex-B-$SET.result.json"
  (cd "$HOME_B" && find . -type f | sort) > "$OUT/codex-B-$SET.home-files-after.txt"
  python3 - "$OUT/codex-B-$SET.result.json" "$WD_B/notes.txt" "$TID" <<'PY' | tee "$OUT/codex-B-$SET.verdict.json"
import json, os, sys
r = json.load(open(sys.argv[1]))
v = {"reply_has_codeword": "PELICAN-7" in (r.get("reply") or ""), "notes_txt_exists": os.path.exists(sys.argv[2]),
     "same_thread_id": r.get("thread_id") == sys.argv[3], "no_commands": not r.get("commands")}
v["pass"] = v["reply_has_codeword"] and v["notes_txt_exists"] and v["no_commands"]
print(json.dumps(v))
PY
fi
