#!/bin/bash
# mem-watch.sh [once]: per-worker cgroup memory (anon+shmem = non-reclaimable, file = cache, current, max) via one
# `srun --overlap` per node. Prints every worker with `once`; otherwise loops every 15 min and prints only workers
# over the thresholds (ANON_WARN MB of anon+shmem, CUR_WARN MB of memory.current).
ANON_WARN=${ANON_WARN:-9500}; CUR_WARN=${CUR_WARN:-10500}
snapshot() {
  local mine; mine=$(squeue -u carlos_patino -h -t R -o "%i %j" 2>/dev/null | grep run4-agent | awk '{print $1}' | paste -sd'|')
  squeue -u carlos_patino -h -t R -o "%N %i %j" 2>/dev/null | grep run4-agent | sort -k1,1 -u | while read -r node jid _; do
    MINE="$mine" timeout 60 srun --jobid=$jid --overlap -N1 -n1 --quiet --export=ALL,MINE bash -c '
      for d in /sys/fs/cgroup/system.slice/slurmstepd.scope/job_*; do
        [ -r "$d/memory.current" ] || continue; j=${d##*job_}; echo "$j" | grep -qE "^($MINE)$" || continue
        read -r a < <(awk "/^anon /{print \$2}" $d/memory.stat); read -r s < <(awk "/^shmem /{print \$2}" $d/memory.stat); read -r f < <(awk "/^file /{print \$2}" $d/memory.stat)
        printf "%s %s anon+shmem=%d file=%d current=%d max=%s\n" "$(hostname)" "$j" $(((a+s)/1048576)) $((f/1048576)) $(($(cat $d/memory.current)/1048576)) "$(awk "{print (\$1==\"max\")?\"max\":int(\$1/1048576)}" $d/memory.max)"
      done' 2>/dev/null < /dev/null | grep -v "Auto Resume"
  done
}
if [ "${1:-}" = once ]; then snapshot | sort -t= -k2 -rn; exit 0; fi
while true; do
  snapshot | while read -r line; do
    a=$(echo "$line" | sed -n 's/.*anon+shmem=\([0-9]*\).*/\1/p'); c=$(echo "$line" | sed -n 's/.*current=\([0-9]*\).*/\1/p'); f=$(echo "$line" | sed -n 's/.*file=\([0-9]*\).*/\1/p')
    [ "${a:-0}" -ge "$ANON_WARN" ] || { [ "${c:-0}" -ge "$CUR_WARN" ] && [ "${f:-99999}" -le "${FILE_WARN:-1200}" ]; } && echo "$(date -u +%H:%M) MEM $line"
  done
  sleep 900
done
