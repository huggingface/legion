#!/usr/bin/env bash
# Prompt every agent of a swarm from the CLI, 25 CLI processes at a time.
#   prompt_all.sh <swarm> <out.jsonl>
# Each output line: {"agent", "codeword", "sent": epoch s, "ack": {...}}.
# Needs FLOCK_URL and FLOCK_TOKEN.
set -euo pipefail
SWARM=${1:?swarm}; OUT=${2:?out.jsonl}
: > "$OUT"
python3 -c "import sys; from flock.client import Client
print('\\n'.join(a['name'] for a in Client().agents(sys.argv[1])))" "$SWARM" |
  xargs -P 25 -I{} bash -c '
    cw="{}-$(python3 -c "import secrets;print(secrets.token_hex(3))")"
    t=$(date +%s.%N)
    ack=$(flock prompt {} "Your codeword is $cw. Record it in your file and reply with only the codeword.")
    printf "{\"agent\":\"%s\",\"codeword\":\"%s\",\"sent\":%s,\"ack\":%s}\n" {} "$cw" "$t" "$ack" >> "$0"
  ' "$OUT"
wc -l < "$OUT"
