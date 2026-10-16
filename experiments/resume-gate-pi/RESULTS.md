# Resume gate: Pi (NOT RUN YET)

Question: can a Pi session be saved on one machine, restored on another, and resumed with its
context intact, as root in an HF Job? This is milestone 1 of the flock v2 note and must pass before
pi types are used for real work. Pi `@earendil-works/pi-coding-agent@1.2.0`, Node 22.23.3.

**Status: not run. Needs operator approval for two cpu-basic jobs (below).** Only the local dry run
has been done.

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

## How to run (needs approval)

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

## Results

(to fill in after the jobs run: job ids, wall times, bootstrap seconds, `user`, startup, idle
memory on cpu-basic, the verdict)
