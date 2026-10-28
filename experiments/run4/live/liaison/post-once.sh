#!/bin/bash
# post-once.sh <marker phrase> <file> [thread]: until success (max ~50 min), post <file> to thread 33745 (run-4 status thread; Carlos: post there) unless a message
# of mine containing <marker> already exists (board search), so retries never duplicate.
D=$(dirname "$0"); T=$(cat "$D/token"); B=https://lvwerra-msgboard-demo.hf.space/v1
M="$1"; F="$2"; TH="${3:-33745}"; end=$((SECONDS+3000))
while [ $SECONDS -lt $end ]; do
  q=$(python3 -I -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]+' from:@science-cluster-liaison'))" "\"$M\"")
  r=$(curl -s -m 30 -w '\n%{http_code}' -H "Authorization: Bearer $T" "$B/search?q=$q&limit=1"); code=$(echo "$r"|tail -1)
  if [ "$code" = 200 ]; then
    n=$(echo "$r"|head -n -1|python3 -I -c "import json,sys;print(json.load(sys.stdin).get('total',0))")
    if [ "${n:-0}" -gt 0 ]; then echo "already posted ($(date -u +%H:%M:%S))"; exit 0; fi
    out=$(timeout 200 "$D/post.sh" 20 "$TH" < "$F"); echo "$out"|tail -1
    echo "$out"|grep -q "posted id" && exit 0
  else echo "board http $code at $(date -u +%H:%M:%S)"; fi
  sleep 60
done; echo "gave up"
