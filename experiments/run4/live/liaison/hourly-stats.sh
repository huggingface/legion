#!/bin/bash
# hourly-stats.sh: cluster-side numbers for the run-4 status thread (no secrets printed; Hub read via hf discussions list).
R=/fsx/carlos_patino/prj-eliza/runs/run4; H=/fsx/carlos_patino/prj-eliza/venv-run4-ctl/bin/hf
echo "== $(date -u +%FT%TZ)"
echo "-- waves (last 4):"; tail -4 $R/logs/waves-run4.jsonl | python3 -I -c 'import sys,json,time
for l in sys.stdin:
    d=json.loads(l); print(" ", time.strftime("%H:%M:%S", time.gmtime(d["ts"])), d["why"], d["before"], "->", d["after"])'
grep -i "hold\|ACTION" $R/logs/waves-run4.out | tail -2 | cut -c1-160
echo "-- jobs:"; squeue -u carlos_patino -h -o "%j %T" | awk '{n[$1" "$2]++} END{for(k in n) if (k ~ /run4-agent/) a[$0]=0; } {print}' >/dev/null; squeue -u carlos_patino -h -o "%j %T" | awk '/run4-agent/{w[$2]++} /flock-world|flock-ctl|run4-vllm/{print "  "$1" "$2} END{for(s in w) print "  agent workers "s": "w[s]}'
echo "-- board 429 total: $(cat $R/logs/run4-agent-w*.out | grep -c 'board 429'); budget refusals last hour: $(grep -h 'board busy' $R/logs/run4-agent-w*.out | awk -v t="$(date -u -d '-60 min' +%H:%M)" '{ if (substr($2,1,5) >= t) c++ } END{print c+0}')"
echo "-- vLLM:"; for h in ip-10-53-196-166 ip-10-53-197-39; do curl -s -m 10 http://$h:8000/metrics | awk -v h=$h '/^vllm:num_requests_running\{/{r+=$2} /^vllm:num_requests_waiting\{/{w+=$2} /^vllm:request_success_total\{/{s+=$2} /^vllm:prefix_cache_queries_total\{/{q+=$2} /^vllm:prefix_cache_hits_total\{/{hh+=$2} END{printf "  %s running=%d waiting=%d success_total=%d prefix_hit=%.3f\n", h, r, w, s, (q>0?hh/q:0)}'; done
echo "-- Hub PRs (status counts over all discussions):"; $H discussions list cmpatino/chatbot-swarm --type dataset --status all --limit 5000 --format json 2>/dev/null | python3 -I -c 'import sys,json
d=json.load(sys.stdin); items=d if isinstance(d,list) else d.get("discussions",d.get("items",[]))
from collections import Counter; c=Counter(i.get("status") for i in items); print("  total", len(items), dict(c), "newest", max((i.get("num",0) for i in items), default=0))' 2>/dev/null || echo "  (hf json listing unavailable)"
echo "-- OOM-killed worker steps today: $(sacct -u carlos_patino -S $(date -u +%Y-%m-%d) -n -o State%14 | grep -c OUT_OF_ME); FAILED worker jobs last hour: $(sacct -u carlos_patino -S $(date -u -d "-60 min" +%H:%M) -X -n -s F -o JobName%24 | grep -c run4-agent)"
echo "-- controller status:"; ( set +u; set -a; . /fsx/carlos_patino/prj-eliza/secrets/secrets.env >/dev/null 2>&1; set +a; export FLOCK_URL=http://$(cat $R/hosts/controller.host):7860; /fsx/carlos_patino/prj-eliza/venv-run4-ctl/bin/flock status run4 2>&1 | head -4 | sed 's/^/  /' )
echo "-- worker RSS (top 3):"; for j in $(squeue -u carlos_patino -h -o "%i %j" | awk '/run4-agent/{print $1}'); do sstat -j $j.0 -n -o MaxRSS 2>/dev/null | awk -v j=$j '{print $1, j}'; done | sort -h | tail -3 | sed 's/^/  /'
