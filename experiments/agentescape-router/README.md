# AgentEscapeBench on a cluster router

Sixteen Pi agents solve difficulty 25, scenario 0 in independent Podman rooms and
share discoveries on one existing message board. This ports the shepherd branch's
pinned task and team instructions to Flock's current spec and message_board tools.
Eight Slurm workers run two agents each, requesting 4 CPUs and 7600 MB per worker.

Provide RUN_NAME, RUN_BUCKET, ROUTER_URL and BOARD_URL, with ROUTER_API_KEY,
BOARD_TOKEN and HF_TOKEN in the protected run environment. Prepare a run with
`flock init` and the cluster's validated Podman settings, then launch spec.yaml.
Each agent gets one immediate solve prompt, up to 200 model steps and 60 minutes.
There is no warmup/configuration transition and no automatic first-success stop.

Monitor actual turn starts and board activity. Once all solve turns finish,
`flock stop RUN_NAME --release` drains workers and runs the native verifier.
Keep worker artifacts on shared storage. Report rewards from trial result.json,
puzzle evidence, exceptions and token usage; an agent's claim is not a verified win.
