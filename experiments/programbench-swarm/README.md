# Sixteen agents reconstruct csview

One coordinator (agent-001), fifteen generalists, sixteen isolated ProgramBench
workspaces, one shared board and private HF source repository. Use the pinned
csview task and Pi against an OpenAI-compatible router. No technical areas are
preassigned. The shared candidate and each agent's native submission are reported
separately; these are not sixteen independent benchmark samples.

The working phase is100 minutes followed by one final submission turn capped
at20 minutes. Allow setup plus the task's7,200-second native verification separately.
Eight Slurm workers each host two agents (24 CPUs, 44 GiB, five-hour allocations).
The early build milestone is a prompt instruction, not an enforced scoring rule.

## Launch gates

Do not submit the full swarm until all of these pass:

1. One-agent Podman pilot installs Pi, reaches the router, submits a committed
   buildable original program, and receives a native verifier result. Its score
   need not be high; infrastructure success is the gate.
2. Two agents on separate workers exchange a PR through the shared board and
   repository, synchronize code, and export their trial results.
3. Router tool calling and16 concurrent short requests succeed. Inspect capacity
   and confirm the router allocation outlasts the experiment.

Never expose hidden tests or oracle files to agents. A merged PR, an idle loop,
or a verifier exception is not a successful ProgramBench evaluation.

## Preparing a full run

Use a fresh timestamped run directory and name, private HF dataset initialized
with a README, board deployed once, and bucket results prefix. Each worker connects
to the existing board URL; do not configure Space deployment on every worker.

Supply RUN_NAME, RUN_BUCKET, ROUTER_URL, HF_REPO_ID and BOARD_URL. Supply HF_TOKEN,
ROUTER_API_KEY and BOARD_TOKEN through the runtime's protected secret handling.
The router key comes from its local api-key file, never from a committed YAML.
The operator explicitly approved sharing HF_TOKEN with these experiment agents;
the local hf_repo plugin implements that choice without changing Flock defaults.

Use `flock init RUN_DIR --from experiments/programbench-swarm --profile PROFILE`
with the cluster profile based on cluster.env.example, then fill the run's
mode 0600 environment using the existing deployment workflow. The run snapshot
must configure unqualified container images to resolve against docker.io and use the validated cluster's Netavark/aardvark 1.11 helpers, job-local Podman
storage, and Git LFS on PATH. Keep generated launch artifacts and secrets outside
this tracked experiment folder. Validate `flock init RUN_DIR` before launch.

The programbench_runtime resource temporarily adds only Pi installation hosts to
the task allowlist and restores the original policy before agent execution. It
also gives worker drain 7,500 seconds for native verification. These are explicit
experiment-local compatibility hooks for the pinned Harbor release.

Start `warmup.yaml` first with the same run name and resources. It initializes
all environments and asks each agent to post READY without beginning task work.
Only after all 16 are ready, update the same swarm to spec.yaml; the changed loop
starts the 100-minute working phase, excluding startup from the 120-minute budget.
Do not begin if the earliest worker has less than 250 minutes remaining. If workers
cannot all become ready within 30 minutes, stop this attempt and report capacity.

Launch with `flock --run RUN_DIR launch RUN_DIR/experiment/warmup.yaml`, then use
`flock --run RUN_DIR update RUN_DIR/experiment/spec.yaml` after the readiness gate.
Monitor every ten
minutes: startup, active agents, router failures, memory, first build, PR activity,
and token accounting. The coordinator announces the frozen candidate SHA in the
final phase. Once all final turns end, release the run with `flock --run RUN_DIR
stop RUN_NAME --release`; initiate drain while the allocation still has at least
130 minutes left. Verify results exist before declaring completion or cancelling
workers. The loop does not itself shut down a remote controller or trigger grading.

## Results

Collect native trial results, ProgramBench evaluator logs, board export, PR history,
final candidate SHA, each task submission SHA, timing, tokens, and Slurm accounting.
Run `python experiments/programbench-swarm/report.py COLLECTED_RUN_DIRECTORY` for a
basic trial/token summary. Missing verification stays missing rather than becoming
zero reward. Check build failures separately from behavioral mismatches. Report
router GPU allocation and Slurm wall time; a zero API price is not a free run.
