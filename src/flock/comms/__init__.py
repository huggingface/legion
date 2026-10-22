"""Comms (docs/comms.md): the team's message board behind the worker.

`board` is a small msgboard client (sync for the controller, async for the worker); `tools`
holds the built-in `comms:*` tool definitions, the rule checks and search cut 1. The worker side
(join at first prompt, the one activity poller, fan-out, the tool gateway) is
`flock.worker.comms`; the controller side (channel resolution, kickoff) `flock.controller.comms`.
"""
