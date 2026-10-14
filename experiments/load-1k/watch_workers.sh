#!/usr/bin/env bash
# watch_workers.sh <swarm> <out.tsv> [every s]: one line per worker per poll (time, id, state, job,
# agents, crashes, last_heartbeat). Needs FLOCK_URL/TOKEN.
SWARM=${1:?}; OUT=${2:?}; EVERY=${3:-5}
while true; do
  curl -s -H "Authorization: Bearer $FLOCK_TOKEN" "$FLOCK_URL/workers?swarm=$SWARM" |
    python3 -c 'import json,sys,time
t=time.strftime("%H:%M:%S", time.gmtime())
try: ws=json.load(sys.stdin)
except Exception: print(t, "error"); sys.exit()
for w in ws: print(t, w["id"], w["state"], w["job_id"], len(w["agents"]), w["crashes"], (w["last_heartbeat"] or "")[11:19], sep="\t")' >> "$OUT"
  sleep "$EVERY"
done
