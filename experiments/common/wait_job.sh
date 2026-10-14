#!/usr/bin/env bash
# wait_job.sh <job_id> <log_file>: poll until the job reaches a terminal stage, then save its logs.
J=${1:?}; LOG=${2:?}
until st=$(hf jobs inspect --namespace cmpatino --json "$J" 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print(d["status"]["stage"], d["status"].get("message") or "")' 2>/dev/null) \
      && [[ "$st" =~ ^(COMPLETED|ERROR|CANCELED|DELETED) ]]; do sleep 10; done
echo "stage=$st"
hf jobs logs --namespace cmpatino "$J" > "$LOG" 2>&1
tail -40 "$LOG"
