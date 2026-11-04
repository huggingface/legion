# Resume gate (milestone 1, step 1)

Question: can a harness session be saved on one machine, restored on another, and resumed with
its context intact? Run 2026-10-14. Claude Code 2.1.296, Codex CLI 0.162.1.

## Claude Code: two HF Jobs

| | Job | Wall (sched + run) | Image bootstrap |
|---|---|---|---|
| Phase A (start, write, sync up) | `6acf45f4c66288145223cf9e` | 2 s + 17 s | 9.3 s (npm claude 5.5 s, uv+hf 1.7 s) |
| Phase B (fresh container, restore, resume) | `6acf464a5ccb47ba0f7980e9` | 2 s + 12 s | 7.0 s |

Both: `cpu-basic`, `node:22-bookworm`, label `experiment=flock-m1`, workdir
`/work/agents/probe-0001`, `CLAUDE_CONFIG_DIR=/state/claude`, session
`61f47177-2753-4940-8bbd-e079e993d7c0`, bucket prefix
`hf://buckets/cmpatino/flock/resume-gate/claude/`.

How it was run:

```sh
cd experiments
SID=$(python3 -c 'import uuid;print(uuid.uuid4())')
./common/launch.sh resume-gate-claude-A resume-gate/claude_resume.sh -e JOB_ARGS=A -e SESSION_ID=$SID -e CLAUDE_CONFIG_DIR=/state/claude -e OUT=/out
common/wait_job.sh <job-id> resume-gate/out/job-A.log
./common/launch.sh resume-gate-claude-B resume-gate/claude_resume.sh -e JOB_ARGS=B -e SESSION_ID=$SID -e CLAUDE_CONFIG_DIR=/state/claude -e OUT=/out
```

`launch.sh` syncs `experiments/` to `scripts/<rev>/` in the bucket and runs
`common/job_bootstrap.sh` inline. The bootstrap installs `@anthropic-ai/claude-code@2.1.296` and
`huggingface_hub==2.1.1` (through uv), pulls the scripts, and execs the phase.
`claude_resume.sh` ran these commands:

```sh
# A (IS_SANDBOX=1 because the job runs as root)
claude -p --session-id $SID --model claude-haiku-5-5 --permission-mode bypassPermissions \
  --max-budget-usd 0.25 --output-format stream-json --verbose \
  "The codeword is PELICAN-7. Write it to notes.txt and reply with only the codeword."
hf buckets sync /work/agents/probe-0001 $DEST/workdir --delete
hf buckets sync $CLAUDE_CONFIG_DIR/projects/-work-agents-probe-0001 $DEST/projects/-work-agents-probe-0001 \
  --include "$SID.jsonl" --include "$SID/*" --delete
# B (fresh container, empty CLAUDE_CONFIG_DIR)
hf buckets sync $DEST/workdir /work/agents/probe-0001
hf buckets sync $DEST/projects/-work-agents-probe-0001 $CLAUDE_CONFIG_DIR/projects/-work-agents-probe-0001
claude -p --resume $SID --model claude-haiku-5-5 --tools "" --max-budget-usd 0.25 \
  --output-format stream-json --verbose "What is the codeword? Reply with only the codeword."
```

`--tools ""` on the resumed turn stops the model from reading `notes.txt`. The answer must come
from the restored transcript. The stream confirms no tool calls in turn B.

Values from the `result` events:

| Turn | result | session_id | duration_ms | duration_api_ms | total_cost_usd | usage (in / cache write / cache read / out) | tools |
|---|---|---|---|---|---|---|---|
| Job A | PELICAN-7 | 61f47177… | 2,818 | 2,425 | 0.001143 | 4 / 6,111 / 28,030 / 196 | Write |
| Job B (resumed) | PELICAN-7 | 61f47177… (same) | 1,103 | 3,315 (cumulative) | 0.001936 (cumulative) | 2 / 6,264 / 535 / 9 | none |
| Local dry run A | PELICAN-7 | d30d444e… | 2,214 | 1,949 | 0.001153 | 4 / 6,319 / 28,211 / 162 | Write |
| Local dry run B | PELICAN-7 | d30d444e… (same) | 2,431 | 4,153 (cumulative) | 0.001954 (cumulative) | 2 / 6,324 / 535 / 9 | none |

Verdict from job B: `{"reply_has_codeword": true, "notes_txt_exists": true, "same_session_id": true, "no_tools_used": true, "pass": true}`.

The local dry run used the same script with two fresh temporary `CLAUDE_CONFIG_DIR`s. The workdir
was deleted between phases and `ANTHROPIC_API_KEY` came from `ANTHROPIC_KEY` under `env -u`.
Raw outputs are in `out/` (gitignored); key fields are in `claude-summary.json`.

Facts the worker depends on:
- **Restore needs only `projects/<enc>/<session>.jsonl`** plus the workdir, at the same absolute
  workdir path. Turn A also created `.claude.json`, `backups/`, `policy-limits.json*` and
  `remote-settings.json` in the config dir. None of them were restored, and resume still worked.
  No `<session>/` subdirectory appeared for this simple session. Sessions that spawn subagents or
  write large tool results may create one, so sync `<session>/` when it exists (the include
  pattern above does this).
- `encoded cwd` = the absolute workdir with every non-alphanumeric character replaced by `-`
  (`/work/agents/probe-0001` -> `-work-agents-probe-0001`). Confirmed.
- On a resumed turn, **`total_cost_usd`, `duration_api_ms` and `modelUsage` are cumulative for the
  session**. They are not per-process and not per-turn. `usage` is per-turn. Cost accounting must
  diff `total_cost_usd` against the previous turn's value. Example: job B's own cost is
  0.001936 - 0.001143 = 0.000793 USD. The packing experiment shows the same behaviour inside one
  long-lived streaming session.
- Changing the tool set (`--tools ""`) changes the cached prefix. Turn B therefore wrote 6.2k
  cache tokens and read only 535. With an unchanged tool set, a resume reads the prefix from cache.
- Jobs run as root. `--permission-mode bypassPermissions` needs `IS_SANDBOX=1` there.
- Job bootstrap from `node:22-bookworm` took 6–9 s. `python3` 3.11 is already in the image.

## Codex: local, two simulated machines

The model ran with `-c model_reasoning_effort="low"`. Each machine had a fresh `CODEX_HOME`
holding only copies of `auth.json` and `config.toml`. Workdirs:
`$ROOT/machineA/work/agents/probe-0001` and `$ROOT/machineB/work/agents/probe-0001`. Before
resuming, the workdir was copied A->B and machine A's directory was moved out of reach.

```sh
export ROOT=$(mktemp -d) OUT=$PWD/out/codex
./codex_resume.sh A
./codex_resume.sh B rollout
# each turn: codex exec --json -m gpt-6-luna -c model_reasoning_effort="low" --skip-git-repo-check \
#   --sandbox danger-full-access -c approval_policy="never" [resume <thread_id>] "<prompt>" < /dev/null
```

| Turn | thread_id | reply | usage (input / cached / output) | commands | wall |
|---|---|---|---|---|---|
| A try 1 (`--sandbox workspace-write`) | 01a139b1-2f8c… | PELICAN-7 | 33,897 / 18,944 / 50 | `cat > notes.txt` **failed (bwrap)** | – |
| A | 01a139b1-a8f6… | PELICAN-7 | 33,322 / 17,920 / 52 | `cat > notes.txt` ok | not recorded |
| B, rollout only | 01a139b1-a8f6… (same) | PELICAN-7 | 53,773 / 34,048 / 61 (thread-cumulative) | none | 7.9 s |

Verdict: `{"reply_has_codeword": true, "notes_txt_exists": true, "same_thread_id": true, "no_commands": true, "pass": true}`.
This used 3 of the 6 allowed Codex turns. No sqlite retry was needed.

Facts the worker depends on:
- **Restore needs only `sessions/YYYY/MM/DD/rollout-<ts>-<thread_id>.jsonl`**, at the same relative
  path under `CODEX_HOME`, plus `auth.json`/`config.toml` (which the worker provisions anyway).
  `state_5.sqlite`, `thread_history_1.sqlite`, `logs_2.sqlite`, `goals_1.sqlite`,
  `memories_1.sqlite`, `queue_1.sqlite` and `session_index.jsonl` were **not** copied. Codex
  created fresh copies and indexed the thread itself (`threads` has 1 row in home B after resume).
- Resume **appends to the same rollout file**. It writes a new `turn_context` with the **new cwd**
  (machine B's path), so Codex follows the current cwd and does not need the original path. The
  worker still restores to identical paths for symmetry with Claude.
- `codex exec` waits on stdin unless it gets `< /dev/null` ("Reading additional input from
  stdin..."). The worker must close stdin.
- **`--sandbox workspace-write` fails in this container.** bwrap cannot create user namespaces.
  The command fails, and the model may still answer as if it succeeded. Use
  `--sandbox danger-full-access` (the job container is the sandbox), as the Space's own Codex
  config already does. HF Jobs very likely behave the same, but this was not tested.
- `turn.completed.usage` matches the rollout's `total_token_usage`, which is **cumulative for the
  thread**. Per-turn usage is `last_token_usage` in the rollout, or the diff between successive
  `turn.completed` events.
- The copied OAuth `auth.json` was not refreshed during the test (checksums of the original and
  both copies are unchanged). Codex refreshes about 8 days after `last_refresh`. A worker must
  never hold a copy of a rotating OAuth credential. Use an API key.

## Unverified
- Codex inside an HF Job: there is no `OPENAI_API_KEY` here, and the OAuth `auth.json` must not
  leave this machine. Cross-machine resume was only simulated locally (separate homes and
  workdirs on one host).
- Codex via the app-server (the design's per-agent process) rather than `codex exec`. The
  rollout format is shared, so restore should be the same.
- Resume after a CLI version bump, for either harness.
- Claude sessions with subagents or large tool outputs (the `<session>/` subdir).

## Verdict

| Harness | Result | Files to sync per agent (plus the workdir, restored at the same absolute path) |
|---|---|---|
| Claude Code 2.1.296 | **PASS** (two HF Jobs, fresh container) | `$CLAUDE_CONFIG_DIR/projects/<enc-cwd>/<session_id>.jsonl` and `$CLAUDE_CONFIG_DIR/projects/<enc-cwd>/<session_id>/` if present |
| Codex 0.162.1 | **PASS** (local two-home simulation; job untested) | `$CODEX_HOME/sessions/YYYY/MM/DD/rollout-*-<thread_id>.jsonl` (same relative path); no sqlite files needed |
