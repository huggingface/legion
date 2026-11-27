# Carry the load

You control one ant in a team of ten. Together the ants must carry a rigid I-shaped load east,
through the gaps in two walls. The load is heavy: it moves only while at least five ants are
attached to it and lifting at the same time, and it must be turned to fit through the gaps.

The world runs on its own clock, in real time, and does not wait for you. Your ant keeps
doing its latest command until you replace it, also while you think. The clock starts once
every ant is ready; then the world runs for 4,000 ticks of 0.1 s (about 7 minutes), or until
the load is through.

What your ant senses is local: 24 rays that stop at walls and the load, your own velocity,
attachment and odometry (your displacement from where you started), a compass cue, and the
ants close to you (walls and the load hide the others), each with the text it shows. All of it
is in your ant's own frame: +x is forward, +y is left, and ray 0 points forward, the others
every 15 degrees counterclockwise. `compass_goal` is the direction of east in that frame.
There is no map: build your own from what you see. Your ant's senses are recorded every
0.5 s, and you can read back its latest 128 records.

## Tools

You act in the world through its tools, called with `world.call` (`world.tools` lists them):

- `observe`: with no arguments, your latest record and the episode's `run_id`. With `run_id`
  and `after_seq` (`-1` at first, then the answer's `next_cursor`), your records since then.
- `ready` (`run_id`): call it once, after your first `observe`.
- `action` (`run_id`, `action_seq`, `direction`, `mode`): your ant's new command. `direction`
  is a compass direction, not one in your ant's frame: `[x, y]`, x toward east and y to the
  left of east, each `-1`, `0` or `1`. `[1, 0]` is east whichever way your ant faces, and
  `compass_goal` shows where that is for your ant; `[0, 0]` stands still. `walk` moves and lets go of the
  load, `lift` attaches to the load within reach and carries it, `push` pushes on contact and
  cannot pull. `action_seq` grows with every new command: 1, 2, 3, ...
- `advertise` (`run_id`, `signal_seq`, `advertisement`): a short text, up to 256 characters,
  that ants near you see until you replace it. It is the only way to reach the other ants in
  the world. `signal_seq` grows like `action_seq`.

An `accepted` answer means your command or text is queued: it takes effect at the next world
tick, not at once. If a call fails or times out, send it again with the same sequence number
and the same content; only a new command or text takes a larger one. If you lose track (a
restart, for example), `observe` shows the last numbers accepted (`command.accepted_seq`,
`signal.accepted_seq`): continue above them, never from 1 again.

For example:

```sh
legion-tool world.call --json '{"tool": "observe"}'
legion-tool world.call --json '{"tool": "action", "args": {"run_id": "RUN_ID", "action_seq": 1, "direction": [1, 0], "mode": "walk"}}'
```

Keep going until `observe` says `done`, then stop.
