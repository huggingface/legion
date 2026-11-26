# AgentEscape swarm SFT collection

Collect Qwen3.5-27B demonstrations with sixteen-agent Harbor/Pi swarms and the
existing Legion messageboard. Production is one swarm per frozen SFT scenario:
25 each at difficulties 5/10/15/20, twelve at 25. Only native-verified winning
teams enter the export, including all their collaborators' completed responses.
Team success and individual success remain distinct labels.

The experiment pins the registry, scenario contents, model, and source snapshot.
Reasoning is disabled; temperature .7, top-p .8, top-k 20, context 65536, maximum
output 16384, 200 turns, and a synchronized 60-minute work budget. Stop at first
verified win. Eight B300 replicas, four on each of two nodes, serve eight concurrent swarms. Each
infrastructure failure gets at most one fresh attempt; valid puzzle failures do
not get retried. Pilots are never production examples.

## Running

From the Legion repository root, use a shared checkout and virtual environment
visible to all Slurm nodes. The supplied scripts target the HF cluster's
`hopper-b300` and `hopper-cpu` partitions. They require Pyxis, a shared vLLM
0.32.0 SquashFS image, and rootless Podman workers with git-lfs and the netavark
and aardvark-dns executables. The router installs `vllm-router==0.1.16` at
startup. See [Legion's Slurm setup](../../docs/slurm.md) for worker prerequisites.

Authenticate to the Hub on the login node. Provide a private, mode-0600
shell env file containing `HF_TOKEN` with write access to your run bucket.
Use shared absolute paths for the run directory, helper directory, HF cache,
and image. Keep the run directory private because it contains raw captures
and per-attempt credentials. Set the values below for your cluster.

```bash
uv sync --all-extras
export QWEN_IMAGE=/shared/images/vllm-openai-v0.32.0.sqsh
export SFT_EXPORT_PYTHON=/shared/export-env/bin/python  # transformers with Qwen3.5 support and pyarrow
RUN=/shared/runs/agentescape-sft-01
PIPE=experiments/agentescape-sft/pipeline.py
umask 077
.venv/bin/hf download Qwen/Qwen3.5-27B --revision fc05daec18b0a78c049392ed2e771dde82bdf654
.venv/bin/python "$PIPE" prepare "$RUN" --bucket YOUR_NAMESPACE/YOUR_BUCKET \
  --network-helpers /shared/podman/network-helpers --secrets /shared/private/secrets.env
.venv/bin/python "$RUN/pipeline.py" submit "$RUN"
```

Use `squeue --me` to see the jobs and `tail -f "$RUN/supervisor.log"` to follow
the gates. The supervisor writes the production export under `$RUN/dataset-v1/`.
Use `final-status.json` to distinguish a completed collection from an interrupted one.

After a supervisor failure with all pilots completed, use `pipeline.py resume RUN`
to revalidate and reuse the saved pilots. This requires all prior jobs to be terminal
and no outstanding attempt reservations. Previous serving state is archived; a new
router and supervisor run the export gate before admitting production.

Pi's empty final abort event at its turn limit is counted separately and never
exported. Only a null-ID, empty, exact abort after all requests are paired qualifies;
nonempty, duplicate, or otherwise unmatched responses still fail the capture gate.

Prepare downloads the pinned registry and tasks, checks all selected scenario
hashes, and freezes scripts/configuration in the new directory. Submit creates
an independent router and CPU supervisor with a 24-hour maximum allocation. The supervisor first runs a two-agent
capture smoke, then five sixteen-agent pilots (one per difficulty). It checks
capture pairing and native artifacts, exports pilot examples with the actual
teacher tokenizer, and only then admits the production array. A failed gate
stops collection and releases this study's jobs. Existing benchmark/RL jobs are
not owned or changed. `jobs.json`, `gates.json`, `progress.json`, supervisor.log,
and `final-status.json` retain progress and failure evidence.

```bash
.venv/bin/python "$RUN/pipeline.py" validate "$RUN"
"$SFT_EXPORT_PYTHON" "$RUN/export.py" "$RUN" /absolute/path/to/export-v1 \
  --tokenizer /absolute/path/to/pinned/model/snapshot
```

## Dataset contract

`examples.jsonl` defaults to one multi-turn trajectory segment per row with `id`,
`prompt`, `completion`, `tools`, and `metadata`. `prompt` is the initial observed
context; `completion` contains multiple assistant messages and their intervening
tool observations or user messages. Concatenate them to inspect the conversation.
Tool argument JSON is decoded into mappings.

Train on **assistant messages in completion only**. Mask the entire prompt and
all tool/user messages in completion. Completion-only loss alone would incorrectly
supervise tool observations; assistant-only loss alone could supervise old assistant
messages retained in a compacted prompt. Use both masks with a compatible chat
template/trainer. `metadata.assistant_message_indices` identifies the target
messages within completion; `response_ids` and `request_ids` map them to captures.

The exporter joins responses only when the next captured prompt exactly extends
the preceding prompt plus response, with unchanged tools and no intervening
unvalidated assistant. It splits at context compaction/rewrites, tool changes,
or unvalidated assistant context. Each segment
retains its actual initial context. Thus an agent can have multiple rows; segments
must remain together when partitioning data. Every accepted response is supervised
once, with its original conditioning history. No histories from different agents
are merged and no synthetic final answer or terminal tool result is added.

Incomplete/errored responses are excluded as targets. Valid unsuccessful actions
and partial collaborator trajectories from winning teams remain eligible. The
export validates each segment with the pinned teacher tokenizer and a 65536-token
context limit. A segment that cannot fit fails export rather than truncating its
context. Capture and native-success checks are unchanged.

`examples.parquet` has the same columns; prompt/completion/tools/metadata are JSON
strings because tool schemas have arbitrary nested structures. Decode these
columns with `json.loads` before training. IDs and ordering are stable across
repeat exports. `report.json` labels the format and separately counts trajectory
rows (`examples`), assistant targets (`assistant_responses`), and contributing
agents, with tokens/coverage by difficulty and rejection reasons.

For the legacy one-response-per-row format, pass `--format responses`. The default
`--format trajectories` also applies to pilot and production exports in newly
prepared runs. Existing frozen run scripts and previously published datasets are
not modified; re-export a historical run with this script and a new destination:

```bash
python experiments/agentescape-sft/export.py /absolute/path/to/run \
  /absolute/path/to/run/dataset-trajectories-v1 \
  --tokenizer /absolute/path/to/pinned/model/snapshot
```

Raw capture stays private in native artifacts. Only model-visible message-board
observations enter a trajectory. No student training or Hub publication is
performed by this pipeline.

Implementation reuses the validated Qwen benchmark runner and cluster helpers;
`task_splits.py` is the existing frozen split validator from TRL, retained with
its license. No Legion public API or messageboard behavior changes.
