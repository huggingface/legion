#!/bin/bash
# world-watch.sh: one line per actionable world/relaunch event (for the Monitor tool; re-arm every 30 min).
# Every 5 min: /manifest probe + conns/cpu/rss appended to world-health.log; emits on slow (>=${SLOW_S:-20} s) or non-401 answers.
# Every 60 s: vote-failure trip-wire (>=3 new in 15 min), world .out lifecycle lines, worker relaunches finishing setup.
R=/fsx/carlos_patino/prj-eliza/runs/run4; L=$R/logs/world-run4.log; O=$R/logs/flock-world-run4-267838.out
H=$R/logs/liaison/world-health.log; declare -A seen vf; n=0
osz=$(stat -c %s $O); for j in $(squeue -u carlos_patino -h -n run4-agent -o "%i" 2>/dev/null; squeue -u carlos_patino -h -o "%i %j" | awk '/run4-agent/{print $1}'); do seen[$j]=1; done
while true; do
  t=$(date -u +%H:%M); now=$(date +%s)
  vf[$now]=$(grep -ac 'vote failed' $L); for k in "${!vf[@]}"; do [ $((now-k)) -gt 900 ] && unset "vf[$k]"; done
  old=$(printf "%s\n" "${!vf[@]}" | sort -n | head -1); d=$((vf[$now]-vf[$old])); [ $d -ge 3 ] && echo "$t TRIP-WIRE: $d new vote failures in 15 min (total ${vf[$now]})"
  s=$(stat -c %s $O); [ $s -gt $osz ] && { tail -c $((s-osz)) $O | grep -aE 'SIGUSR|SIGTERM|requeue|final ws sync|world answers|restored|Node failure|CANCELLED|exited' | cut -c1-200 | sed "s|^|$t world.out: |"; osz=$s; }
  for jn in $(squeue -u carlos_patino -h -o "%i:%j" | grep run4-agent); do j=${jn%%:*}; [ -n "${seen[$j]}" ] && continue
    f=$(ls $R/logs/*-$j.out 2>/dev/null | head -1); [ -n "$f" ] || continue
    l=$(grep -aE 'set up in|failed=\[[^]]' $f | tail -1); [ -n "$l" ] && { echo "$t relaunch ${jn#*:} ($j): $(echo "$l" | cut -c1-160)"; seen[$j]=1; }; done
  if [ $((n%5)) -eq 0 ]; then
    h=$(cat $R/hosts/world.host); m=$(curl -s -m 20 -o /dev/null -w "%{http_code} %{time_total}" http://$h:7860/manifest || true)
    st=$(timeout 60 srun --jobid=267838 --overlap -N1 -n1 bash -c 'p=$(pgrep -f "flock world serve"|head -1); echo $(ss -tn state established "( sport = :7860 )"|tail -n +2|wc -l) $(ps -o %cpu=,rss= -p $p)' 2>/dev/null)
    echo "$t world $h manifest=${m/ / } conns/cpu/rssKB=[$st ]" >> $H
    c=${m%% *}; tt=${m##* }; { [ "$c" != 401 ] || awk "BEGIN{exit !($tt>=${SLOW_S:-20})}"; } && echo "$t world probe slow/bad: manifest=$m conns/cpu/rssKB=[$st]"
  fi
  n=$((n+1)); sleep 60
done
