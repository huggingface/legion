# Resume gate: Pi (PASSED on cpu-basic, 2026-10-16)

Question: can a Pi session be saved on one machine, restored on another, and resumed with its
context intact, as root in an HF Job? This is milestone 1 of the flock v2 note and must pass before
pi types are used for real work. Pi `@earendil-works/pi-coding-agent@1.2.0`, Node 22.23.3.

**Status: passed.** Jobs A and B ran on cpu-basic as root (operator-approved, one run each, no
retries); verdict `pass: true`. Results below.

## What it checks

`pi_resume.sh` drives `pi --mode rpc` through `gate.py` against `stub_openai.py` (a copy of
`tests/fake_openai.py`) inside the same container, so no model is called and nothing is spent on
inference. Layout as on a worker: `PI_CODING_AGENT_DIR=/state/pi`, workdir
`/work/agents/probe-0001`, `--session-dir /state/pi/sessions/probe-0001`.

- Job A: session `main` gets "The codeword is PELICAN-7" (text reply) and
  `bash: echo PELICAN-7 > notes.txt` (a tool call). Session `cut` starts `sleep 60` in a tool and Pi
  is SIGKILLed mid-tool. Sync up: the workdir and ONLY `sessions/probe-0001/*_<sid>.jsonl` for the
  two ids. Also records `id -un`, Pi startup to the first RPC response, idle VmRSS and the cgroup
  memory delta, and whether the tool outlived the SIGKILL (`cut_orphans`).
- Job B, fresh container: restore both to the same absolute paths (agent dir holds nothing else;
  `models.json` is rewritten for the new stub port), resume both ids. Verdict fields:
  `history_has_codeword` and `history_has_tool_result` (the stub received the full prior history),
  `same_session_file` (`get_state.sessionFile` is the restored file), `no_new_session_files`,
  `no_stderr_warning` (Pi did not say "No project session found"), `stats_continue`
  (`get_session_stats` totals grow from job A's), `cut_session_resumed` (the cut-off session
  resumes and completes a run), `notes_txt`.

## How to run (needs approval; done 2026-10-16)

```sh
cd experiments
export SID_MAIN=$(python3 -c 'import uuid;print(uuid.uuid4())') SID_CUT=$(python3 -c 'import uuid;print(uuid.uuid4())')
COMMON=(-e SID_MAIN=$SID_MAIN -e SID_CUT=$SID_CUT -e OUT=/out -e EXTRA_NPM=@earendil-works/pi-coding-agent@1.2.0)
IMAGE=node:22.23.3-bookworm NO_ANTHROPIC=1 LABEL=experiment=flock-pi TIMEOUT=20m \
  ./common/launch.sh resume-gate-pi-A resume-gate-pi/pi_resume.sh -e JOB_ARGS=A "${COMMON[@]}"
common/wait_job.sh <job-A-id> resume-gate-pi/out/job-A.log
IMAGE=node:22.23.3-bookworm NO_ANTHROPIC=1 LABEL=experiment=flock-pi TIMEOUT=20m \
  ./common/launch.sh resume-gate-pi-B resume-gate-pi/pi_resume.sh -e JOB_ARGS=B "${COMMON[@]}"
common/wait_job.sh <job-B-id> resume-gate-pi/out/job-B.log
hf buckets sync hf://buckets/cmpatino/flock/resume-gate/pi/out-B resume-gate-pi/out/B -q   # B.json, B.verdict.json
```

`launch.sh` expands to `hf jobs run --detach --namespace cmpatino --flavor cpu-basic --timeout 20m
--name resume-gate-pi-{A,B} --label experiment=flock-pi --secrets HF_TOKEN -e SCRIPTS_REV=<rev>
-e JOB_SCRIPT=resume-gate-pi/pi_resume.sh -e JOB_ARGS={A,B} ... node:22.23.3-bookworm bash -c
"$(cat common/job_bootstrap.sh)"`. Only `HF_TOKEN` is sent (for the bucket); no model key.

Cost: cpu-basic is $0.01/h. Each job is bootstrap (Claude + Pi npm installs, uv, hf: about 40 s)
plus about 15 s of gate, so two jobs come to about 2 minutes, **under $0.001**. Worst case, both
hit the 20 min timeout: $0.007. Inference: $0 (stub).

## Local dry run (2026-10-16, not a job)

`ROOT=<scratch> PI=<pi> ./local_dry_run.sh`: phase A in agent dir A, only the two session files
copied into a fresh agent dir B (workdir kept at the same path), A deleted, phase B. Non-root user,
64-core shared dev box.

Verdict: `{"history_has_codeword": true, "history_has_tool_result": true, "same_session_file": true,
"no_new_session_files": true, "no_stderr_warning": true, "stats_continue": true,
"cut_session_resumed": true, "notes_txt": "PELICAN-7", "pass": true}`.
Phase A: startup 3.3 s (0.4 s on a quieter run), idle VmRSS 117,100 kB, `cut_orphans`: 2 (the tool's
bash and `sleep` survive a SIGKILL of Pi; the adapter kills them on stop). The cgroup delta
(69 MB) is not meaningful on a shared box.

## Results (HF Jobs, cpu-basic, 2026-10-16)

| | Job A | Job B |
|---|---|---|
| id | `6ad208545ccb47ba0f7b2058` | `6ad2096d5ccb47ba0f7b207f` |
| created | 11:19:48 | 11:24:29 |
| wall time (HF runtime) | 29 s | 29 s |
| bootstrap seconds (npm claude+pi / uv+hf / total) | 17.0 / 1.8 / 21.4 | 17.7 / 1.7 / 22.0 |
| `user` | root | root |
| Pi startup to first RPC response | 0.399 s | 0.438 s |

- Container: nproc=16, mem=126,760 MB visible (the host's; the cpu-basic limit is not visible via
  `free`), Python 3.11.2, Node 22.23.3, Pi 1.2.0.
- Idle VmRSS of Pi (job A): 107,100 kB. cgroup memory delta at idle: 57,966,592 bytes (about 58 MB).
- `cut_orphans`: [264, 265]. As in the dry run, the tool's bash and `sleep` survive a SIGKILL of Pi,
  so the adapter must kill them on stop.
- Phase A: `main_rc` 0, `cut_rc` -9, 4 stub requests. Session files pushed: exactly the two
  `*_<sid>.jsonl`; `auth.json`, `models*.json`, `settings.json` stayed behind. Job B's agent dir
  held only those two files before Pi started.
- Phase B `main_stats`: userMessages 2 -> 3, assistantMessages 3 -> 4, totalMessages 7 -> 9, tokens
  total 333 -> 448, toolCalls 1 (unchanged), `main_stderr` empty. The cut session resumed and replied.

Verdict (`out/B/B.verdict.json`):

```json
{"history_has_codeword": true, "history_has_tool_result": true, "same_session_file": true, "no_new_session_files": true, "no_stderr_warning": true, "stats_continue": true, "cut_session_resumed": true, "notes_txt": "PELICAN-7", "pass": true}
```

Cost: two 29 s jobs on cpu-basic ($0.01/h), under $0.001. Inference $0 (stub).

## Notes

- Pi as root behaved like non-root: no refusal, no extra prompts with `--no-approve`.
- Phase A's `main_stderr` has "No project session found with id ...; creating a new session". That
  is expected there (fresh id) and absent in B, which is what `no_stderr_warning` checks.
- Bootstrap was dominated by the two npm installs (about 17 s); no install or bootstrap failures.
- The Pi startup (0.4 s) matches the quiet local run; the 64-core dev box had shown 3.3 s when busy.
- `launch.sh` for job B took over 2 minutes in the shell (bucket sync of experiments/), but the job
  was created once; nothing was relaunched.
- Job logs and bucket output are in `out/` (gitignored): `out/job-A.log`, `out/job-B.log`, `out/B/`.
  Bucket: `hf://buckets/cmpatino/flock/resume-gate/pi`.
