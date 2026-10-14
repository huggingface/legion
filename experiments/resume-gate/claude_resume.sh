#!/usr/bin/env bash
# Claude Code cross-machine resume gate.
#   claude_resume.sh A   start a session, write the codeword, push workdir + session file to DEST
#   claude_resume.sh B   restore workdir + session file from DEST, resume with no tools, check codeword
# Env: SESSION_ID (required), WORKDIR (default /work/agents/probe-0001), CLAUDE_CONFIG_DIR (required),
#      DEST (default hf://buckets/cmpatino/swarm-harness/resume-gate/claude), OUT (default ./out),
#      MODEL (default claude-haiku-5-5). Needs ANTHROPIC_API_KEY (or a logged-in CLI) and hf auth.
set -euo pipefail
PHASE=${1:?phase A or B}
: "${SESSION_ID:?}" "${CLAUDE_CONFIG_DIR:?}"
WORKDIR=${WORKDIR:-/work/agents/probe-0001}
DEST=${DEST:-hf://buckets/cmpatino/swarm-harness/resume-gate/claude}
OUT=${OUT:-$PWD/out}
MODEL=${MODEL:-claude-haiku-5-5}
ENC=$(printf '%s' "$WORKDIR" | sed 's/[^a-zA-Z0-9]/-/g')
PROJ="$CLAUDE_CONFIG_DIR/projects/$ENC"
mkdir -p "$OUT"
export IS_SANDBOX=1   # lets bypassPermissions run as root inside the job container

summarize() {  # print the result event of a stream-json file as one JSON line
  python3 - "$1" <<'PY'
import json, sys
res = None; tools = []
for line in open(sys.argv[1]):
    try: ev = json.loads(line)
    except ValueError: continue
    if ev.get("type") == "result": res = ev
    if ev.get("type") == "assistant":
        tools += [c["name"] for c in ev["message"].get("content", []) if c.get("type") == "tool_use"]
keys = ("subtype", "is_error", "result", "session_id", "num_turns", "duration_ms",
        "duration_api_ms", "total_cost_usd", "usage", "modelUsage")
print(json.dumps({**{k: (res or {}).get(k) for k in keys}, "tool_uses": tools}))
PY
}

if [ "$PHASE" = A ]; then
  mkdir -p "$WORKDIR"; cd "$WORKDIR"
  claude -p --session-id "$SESSION_ID" --model "$MODEL" --permission-mode bypassPermissions \
    --max-budget-usd 0.25 --output-format stream-json --verbose \
    "The codeword is PELICAN-7. Write it to notes.txt and reply with only the codeword." \
    > "$OUT/claude-A.stream.jsonl"
  summarize "$OUT/claude-A.stream.jsonl" | tee "$OUT/claude-A.result.json"
  echo "== files under CLAUDE_CONFIG_DIR after turn 1"; (cd "$CLAUDE_CONFIG_DIR" && find . -type f | sort) | tee "$OUT/claude-A.config-files.txt"
  # Sync set: the workdir and ONLY this session's files in projects/<enc>/ (jsonl + optional <id>/ dir).
  hf buckets sync "$WORKDIR" "$DEST/workdir" --delete -q
  hf buckets sync "$PROJ" "$DEST/projects/$ENC" --include "$SESSION_ID.jsonl" --include "$SESSION_ID/*" --delete -q
  hf buckets sync "$OUT" "$DEST/out-A" -q
  hf buckets list "$DEST" -R 2>/dev/null || hf buckets list "$DEST"
elif [ "$PHASE" = B ]; then
  [ -e "$WORKDIR" ] && { echo "WORKDIR already exists; B must run on a clean machine" >&2; exit 2; }
  [ -e "$PROJ" ] && { echo "$PROJ already exists; B must run on a clean config dir" >&2; exit 2; }
  mkdir -p "$WORKDIR" "$PROJ"
  hf buckets sync "$DEST/workdir" "$WORKDIR" -q
  hf buckets sync "$DEST/projects/$ENC" "$PROJ" -q
  echo "== restored files"; (cd "$CLAUDE_CONFIG_DIR" && find . -type f | sort; cd "$WORKDIR" && find . -type f | sort) | tee "$OUT/claude-B.restored-files.txt"
  cd "$WORKDIR"
  # --tools "" so the answer can only come from the restored transcript, not from reading notes.txt.
  claude -p --resume "$SESSION_ID" --model "$MODEL" --tools "" --max-budget-usd 0.25 \
    --output-format stream-json --verbose "What is the codeword? Reply with only the codeword." \
    > "$OUT/claude-B.stream.jsonl"
  summarize "$OUT/claude-B.stream.jsonl" | tee "$OUT/claude-B.result.json"
  python3 - "$OUT/claude-B.result.json" "$WORKDIR/notes.txt" "$SESSION_ID" <<'PY' | tee "$OUT/claude-B.verdict.json"
import json, os, sys
r = json.load(open(sys.argv[1]))
v = {"reply_has_codeword": "PELICAN-7" in (r.get("result") or ""),
     "notes_txt_exists": os.path.exists(sys.argv[2]),
     "notes_txt": open(sys.argv[2]).read().strip() if os.path.exists(sys.argv[2]) else None,
     "same_session_id": r.get("session_id") == sys.argv[3],
     "no_tools_used": not r.get("tool_uses")}
v["pass"] = all(v[k] for k in ("reply_has_codeword", "notes_txt_exists", "no_tools_used"))
print(json.dumps(v))
PY
  hf buckets sync "$OUT" "$DEST/out-B" -q
else
  echo "phase must be A or B" >&2; exit 2
fi
