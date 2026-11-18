# Final maze results

Captured **2026-11-18 21:56:54 CET**. Both grids are complete: 160 teams per model, one team per tree
at each N. All 32 trees are included at every point.

![Success and token efficiency](success-efficiency.png)

| Model | N | Successes/teams | Mean team output tokens | Efficiency |
|---|---:|---:|---:|---:|
| Qwen3.5-27B | 1 | 8/32 | 11,697.688 | 0.1970 |
| Qwen3.5-27B | 2 | 12/32 | 21,763.156 | 0.3176 |
| Qwen3.5-27B | 4 | 20/32 | 39,020.188 | 0.5905 |
| Qwen3.5-27B | 8 | 27/32 | 56,355.219 | 1.1039 |
| Qwen3.5-27B | 16 | 31/32 | 78,241.250 | 1.8257 |
| Qwen3.5-2B | 1 | 2/32 | 37,394.500 | 0.0154 |
| Qwen3.5-2B | 2 | 2/32 | 77,499.344 | 0.0149 |
| Qwen3.5-2B | 4 | 5/32 | 145,589.500 | 0.0396 |
| Qwen3.5-2B | 8 | 9/32 | 254,271.500 | 0.0816 |
| Qwen3.5-2B | 16 | 16/32 | 495,642.031 | 0.1488 |

## Metric

`efficiency = success_rate * N * 9216 / mean_team_output_tokens`.

Both panels use the same valid successes and failures. Only generated/output tokens count,
including tool calls/messages; input and cached-input tokens are excluded. The arithmetic
mean includes all 32 teams, not just successful teams. This is a ratio of aggregate success
to mean usage, not a mean of team ratios. **9,216 tokens/agent is a reference scale, not an
enforced maze budget.** Dashed success lines use each model's own N=1 rate and assume
independence. No intervals are shown; a communication effect is not established.

## Setup and provenance

Qwen3.5-27B and Qwen3.5-2B run with thinking off on the same 32 fixed 391-room trees,
N=1,2,4,8,16 and 210 moves per agent. Each agent has its own maze service; the first exit
stops the team. The example's Qwen3.8 default is overridden by the evaluated models.
Actual collaboration conditions: the same read/claim-door/mark-dead-end protocol hint,
one visible `#general` channel (`max_channels: 1`), mentions and no agent DMs. Raw grid names
remain `27b-hint` and `2b-hint` in metadata.

Evaluated Flock: `03ec1dd9be659bde3751d37e393d014cdd8e5061`; task pin:
`3b6c363d92626bd4d97a7ca89964fb850982d9ab`. Task tooling publication remains pending approval;
no new task SHA is claimed. The preserved 158-team 2B prefix gained only tree19-r9
(success) and tree03-r9 (failure); the complete 27B prefix is unchanged.

[teams.csv](teams.csv), [token-audit.csv](token-audit.csv) and
[run-config-audit.csv](run-config-audit.csv) preserve exact integer usage, hashed Trial
evidence and actual per-run resources. All 2,089 Trial sources were re-read:
1,984 explicit counts plus 105 proven pre-execution setup zeros.
All 188 valid failures remain included; verified rewards and all agent/team sums match.
[capture.json](capture.json) records source/cohort hashes and prefix proof;
[artifact-hashes.json](artifact-hashes.json) covers all report files.

## Recovery and allocated GPU cost

Accepted 2B N=16 workers used 38 CPUs/72,200M (8 teams), 22 CPUs/41,800M (22 teams), then
28 CPUs/53,200M for `t03-r9` and `t19-r9`. Liveness grace rose from 180s to 600s during
recovery; those last two attempts started at 600s. Task/model/instructions, inference
placement, scoring, 20-minute turns and 45-minute loops were unchanged. These are
infrastructure recovery measures, not a demonstrated fix for the underlying stalls.

| Model | Attributed B200 GPU-hours |
|---|---:|
| Qwen3.5-27B | 15.112222 |
| Qwen3.5-2B | 16.876944 |

The private canonical allocation CSV (SHA-256
`d773ebfd5e06316005ffb566e7bd84d399dc4832b264033c181cb25cc7f5f758`) covers twelve actual single-B200 allocations, including
warm-up, idle time and infrastructure retries; CPU time and currency are excluded. Original port9601 starts
attribution at the 18 November 2026, 14:24:56 CET handover; the other original server and
both recovery lifetimes use their full relevant allocations. The 27B exclusions are
15 distinct runs. 2B has 392 discard records covering 210 runs, including one later accepted:
209 distinct excluded runs. [attempts.csv](attempts.csv) deduplicates records and marks that case.

## Limits, validation and reproduction

2B minute-delta cache data are missing before 16:17 CET on 18 November; earlier vLLM
summaries cannot reconstruct them or establish the five-minute thrashing criterion.
Rootless Podman cannot enforce container memory caps. Private original/recovery health
records remain local. Capture metadata retains their source hashes and aggregate accounting,
not raw cache, minute, admission or allocation records. The sealed private provenance manifest
has SHA-256 `11a39624aca2a8287476652fb6f69e726c2a38cb057ef617ab3e5ea5966a33da`.

Independent review re-audited all 320 teams, exact ratios and PNG/SVG. Report Ruff and
diff checks pass. Evaluated `03ec1dd`: Ruff clean, 1102 tests passed/4 skipped. Dataset
checks passed six groups, three real Podman Trials and seven metrics cases; extra strict
lint retains eight unchanged pre-existing findings. [Qualified receipts](validation-receipts.json)
name the actual tested source; these do not imply a merged-main retest.

[27B raw archive](https://huggingface.co/datasets/HuggingFaceSwarms/s3-maze-results/tree/2c61c9e0f2282d2052d8beabfdb5b8ebad2a82ca/final-27b-hint) is private. 2B raw archive publication remains pending exact approval.
The existing 27B revision is not a 2B archive revision. Run
`uv run efficiency.py teams.csv --capture capture.json --out success-efficiency.png`
to reproduce PNG, SVG and [numeric CSV](success-efficiency.csv); redraws preserve this README.

🤖 Generated with [Codex](https://openai.com/codex/)
