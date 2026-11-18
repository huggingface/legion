Attempts set aside: 35 (rows-discarded.jsonl); teams that needed a rerun: 28 of 160

- launch failed or was cancelled (the controller ended before it came up): 8
- controller's node reclaimed (spot): 7
- worker started more than 2 min after the launch (the loop counts from it): 7
- worker stopped answering (e.g. while grading its Trials) and was relaunched: 5
- worker relaunched after a node failure (flock#121's case): 3
- worker ran out of memory (preempted or killed, not relaunched): 3
- worker never started: 2
