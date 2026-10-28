#!/bin/bash
# job-monitor.sh: one line per actionable run-4 event (for the Monitor tool; re-arm every 30 min).
R=/fsx/carlos_patino/prj-eliza/runs/run4; prevw=""; prevf=""; prev429=$(cat $R/logs/run4-agent-w*.out 2>/dev/null | grep -c 'board 429'); declare -A vsz
for f in $R/logs/run4-vllm-267644_*.log; do vsz[$f]=$(stat -c %s "$f"); done
while true; do
  st=$(squeue -j 267644 -h -o "%T" 2>/dev/null | sort | tr '\n' ';'); case "$st" in "RUNNING;RUNNING;") ;; *) echo "$(date -u +%H:%M) vllm job state: [$st]";; esac
  wc=$(squeue -u carlos_patino -h -n flock-ctl-run4,flock-world-run4 -o "%j %T" 2>/dev/null | sort | tr '\n' ';'); [ "$wc" != "$prevw" ] && { echo "$(date -u +%H:%M) world/controller: [$wc]"; prevw=$wc; }
  f=$(sacct -u carlos_patino -S "$(date -u -d '-3 min' +%H:%M)" -s NF,TO,OOM,F -X -n -o JobID,JobName,State 2>/dev/null | tr -s ' ' | tr '\n' ';'); [ -n "$f" ] && [ "$f" != "$prevf" ] && { echo "$(date -u +%H:%M) failed jobs: $f"; prevf=$f; }
  p=$(squeue -u carlos_patino -h -t PD -n run4-agent -o "%j" 2>/dev/null | wc -l); pw=$(squeue -u carlos_patino -h -t PD -o "%j" 2>/dev/null | grep -c run4-agent); [ "$pw" -ge 3 ] && echo "$(date -u +%H:%M) $pw workers pending"
  n429=$(cat $R/logs/run4-agent-w*.out 2>/dev/null | grep -c 'board 429'); [ "$n429" -gt "$prev429" ] && { echo "$(date -u +%H:%M) board 429 count $prev429 -> $n429"; prev429=$n429; }
  for f in $R/logs/run4-vllm-267644_*.log; do s=$(stat -c %s "$f"); [ "$s" -gt "${vsz[$f]:-0}" ] && { tail -c $((s-${vsz[$f]:-0})) "$f" | grep -a "ERROR\|Traceback\|CUDA\|OOM" | cut -c1-200 | sed "s|^|$(date -u +%H:%M) $(basename $f): |"; vsz[$f]=$s; }; done
  sleep 60
done
