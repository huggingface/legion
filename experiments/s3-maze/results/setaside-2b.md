Attempts set aside: 104 (rows-discarded.jsonl); teams that needed a rerun: 66 of 160

- worker stopped answering (e.g. while grading its Trials) and was relaunched: 25
- worker relaunched after a node failure (flock#121's case): 24
- worker started more than 2 min after the launch (the loop counts from it): 23
- controller's node reclaimed (spot): 18
- worker ran out of memory (preempted or killed, not relaunched): 6
- worker relaunched after running out of memory (flock#121's case): 6
- launch failed or was cancelled (the controller ended before it came up): 2

Teams with no attempt that counted (5; missing from the results, not failures): tree 07 at N = 16, tree 12 at N = 16, tree 15 at N = 16, tree 18 at N = 16, tree 23 at N = 16
