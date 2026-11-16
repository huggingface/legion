# Sixteen-agent ProgramBench prompt iteration

This is a record of four exploratory runs on November 16, 2026. The strongest
candidate passed **316/335 native checks (94.3%)**, but no run solved the task.
Build quality and reliable participation improved in different variants; these
results do not establish an advantage over a single agent.

The YAML files are **archived experiment inputs, not standalone launchers**. They
retain the tested resource references and loop settings. The actual runs also
used an external phase monitor and cluster compatibility setup described below.
For a supported starting example, use the [parent experiment](../README.md).

## Shared setup

- Task: `wfxr--csview.8ac4de0`, from
  [HuggingFaceSwarms/programbench-harbor-staging](https://huggingface.co/datasets/HuggingFaceSwarms/programbench-harbor-staging),
  pinned to `71a3477240b19b966da44873be69e15364e50481`.
- Model: `Qwen/Qwen3.8-27B:deepinfra` through
  `https://router.huggingface.co/v1`, low reasoning, at most 16 working agents.
- Harness: Pi through Harbor 0.25.0; Flock checkout `d624786`.
- Topology: eight Slurm `hopper-cpu` workers, two isolated agent workspaces per
  worker, one shared message board and one fresh private HF source dataset per run.
- Allocation: 12 CPUs and 22,800 MB per worker. The baseline's first worker had
  24 CPUs and 44 GiB; remaining requests were reduced to fit available capacity.
- Model work ran sequentially across attempts; native grading could overlap the
  next attempt. No operator solution code, hidden-test feedback, or manual PR
  merges were supplied.

## Variants and actual timing

| Variant | Prompt change | Work / final window | Model calls per burst / turn timeout |
|---|---|---|---|
| [Original](01-original/spec.yaml) | Existing coordinator/generalist instructions | 20m / 5m | 60 / 20m |
| [Build-first](02-build-first/spec.yaml) | Early original build, commit, short probes, small PRs | 20m / 5m | 60 / 20m |
| [Checkpoints](03-checkpoints/spec.yaml) | One checkpoint per turn; initially no reference probes; any agent can integrate reviewed work | About 8m / 5m | 8 / 5m |
| [Preserve reference](04-preserve-reference/spec.yaml) | Copy the reference before building, one identification probe, retain the best tested local candidate | 20m / 10m | 8 / 5m |

Each folder contains the exact team prompt. Namespace was replaced with
`${HF_NAMESPACE}`; other environment placeholders were already in the run files.

The YAML loop limits remained at 100 minutes. The external monitor started its
work timer when all 16 sessions existed or the first board post appeared, called
`flock stop NAME --final` at the work deadline, and then
`flock stop NAME --interrupt --release` at the final deadline. Release allowed
native grading before worker teardown. A final-turn counter records termination,
not successful submission.

Checkpoints was ended early after agents were observed probing their own
implementations as if they were the reference. The final variant also received a
[mid-run prompt update](04-preserve-reference/spec.updated.yaml) at about 10:33 UTC:
no more probing until a buildable source commit existed. Active turns continued
using their previous prompts, so delivery was staggered. This was an adaptive
iteration, not a fixed-prompt replication.

## Native results

| Variant | Results recovered | Buildable submissions | Best checks passed | Full solves |
|---|---:|---:|---:|---:|
| Original | 9/16 | 2/9 recovered | 49/335 (14.6%) | 0 confirmed |
| Build-first | 16/16 | 10/16 | 255/335 (76.1%) | 0 |
| Checkpoints | 16/16 | 16/16 | 221/335 (66.0%) | 0 |
| Preserve reference | 16/16 | 2/16 | 316/335 (94.3%) | 0 |

[results.csv](results.csv) contains one row per agent, including unknown baseline
outcomes. Build-first had five missing `compile.sh` failures and one compile
failure. Preserve reference had fourteen missing `compile.sh` failures. A build
script observed during work was not necessarily present in the submitted commit.

The best final candidate was agent-009's original Python implementation. Its
preserved source SHA-256 exactly matched the native evaluator's executable hash:
`452603b9cc35b1a1d5241a6145de8c126e1d278534eb69d53ba933ff54e57a47`.
The two 76.1% candidates were original Rust implementations.

Grading and cleanup finished at 11:13 UTC, after the two-hour investigation window.
No experiment jobs remained after release. Scores are best-of-swarm observations;
the agents are not independent experimental replicates.

## What worked and what remains unresolved

**Build-first gave the best balance.** Original prompts led to extensive probing
and plans without a usable submission. Explicitly requiring source, executable
`compile.sh`, a clean build, and a commit produced ten valid builds.

**Small checkpoints improved participation.** That variant produced sixteen valid
builds, thirteen PRs, and one confirmed merged CLI patch. The controller recorded
11 final prompts sent/done, compared with 1/0 for Original and 1/1 for Build-first.
However, several agents overwrote the supplied reference and then validated their
own placeholders. Its shorter work window also limits score comparisons.

**Explicit reference preservation worked initially.** All sixteen final-variant
agents copied the same reference before building. The prompt preserved its
`executable` basename to avoid changing CLI usage text. The best candidate reached
94.3%, but most agents still failed to submit. This is not evidence that the final
configuration is the most reliable or that collaboration caused the higher score.

**Timing needs further investigation.** Five-minute limits were associated with
repeated interrupted/stuck turns and delayed final prompts. The final controller
recorded four final prompts sent/done. Harbor's eight-call limiter also adds a
system instruction to finish the *whole task* within that budget, unlike the
intended repeated-checkpoint workflow. A small endpoint request returned HTTP 200
in 1.09 seconds during the stalls; that does not explain long-turn behavior.

**Shared-code integration was uneven.** Original and Build-first each left two
PRs open and no implementation on main. Checkpoints eventually merged one of
thirteen PRs. The final variant opened none. Some merge calls returned opaque
`HTTP unknown` errors; two lasted about 30 seconds, matching the client's timeout.
The underlying exception was suppressed, so timeout is suspected, not proven.
Some agents also mistook the sync tool's local `head_sha` for remote `origin/main`.

**Infrastructure confounded comparisons.** A worker failed with `NODE_FAIL` in
both Original and Preserve reference, affecting agents 001 and 002. Flock replaced
it, but job-local task files/reference backups were not reported restored. Seven
baseline native results were missed by the initial artifact collector's polling
interval. They remain unknown; later collectors captured every result.

## Runtime setup needed to reproduce

These snapshots reference experiment-local resources that are not shipped here:

- `programbench_runtime` temporarily allowed Pi installer hosts
  (`raw.githubusercontent.com`, `github.com`, `objects.githubusercontent.com`,
  `nodejs.org`, `registry.npmjs.org`) during installation, then restored the task's
  network policy. It set the worker verifier wait to 7,500 seconds for this task's
  7,200-second native verifier budget.
- `message_board` used the standard resource against one separately hosted board.
- `hf_repo` used an operator-approved experiment binding to `HF_TOKEN`. For a new
  deployment, use the standard resource and its `HF_REPO_TOKEN` configuration as
  documented in the parent example; this is a credential-setup difference from
  the archived runs, not a tested replacement run.

The Slurm run snapshots also provided Git LFS and Git authentication for task
checkout, job-local Podman storage, Netavark/aardvark 1.11 helpers, explicit DNS
helper paths, a Docker Hub registry default, and a compose wrapper translating
`exec --no-TTY` to `exec -T`. Configure these for the target cluster using
[Slurm deployment](../../../docs/slurm.md); do not copy host-specific paths.

Preserve native verifier results and original source **before** job cleanup.
The tested checkout's automatic grading-artifact preservation depended on
`success_check`, which was not enabled here. External read-only collectors were
used instead. Keep missing verification distinct from a zero score. Provider
pricing was not configured, so Flock's displayed zero cost did not mean free use.

Credentials, deployment scripts with host-specific paths, raw sessions, board
exports, generated solutions, and verifier test contents are deliberately not
part of this review record.
