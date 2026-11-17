Attempts set aside: 47 (rows-discarded.jsonl); teams that needed a rerun: 35 of 160

- an agent's model server thrashed (prefix-cache hits under 50%) for 5+ min: 25
- ran on an overloaded model server (one 27B GPU, KV-cache thrash): 8
- worker relaunched after a node failure (flock#121's case): 6
- controller's node reclaimed (spot): 5
- launch failed or was cancelled (the controller ended before it came up): 2
- worker ran out of memory (preempted or killed, not relaunched): 1
