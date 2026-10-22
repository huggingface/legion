# chatbot-v1

Eight open-model agents build ONE conversational chatbot in 4 hours: natural, friendly small
talk, explicitly not a therapist or ELIZA reflection. The agents are 4 DeepSeek-V4.1-Flash and
4 GLM-5.3-Flash on the HF router.

- **Phase 1 (0-2 h): two teams of four, apart.**
  - Team A is `ds-a` ×2 and `glm-a` ×2. Its direction: a persona plus a small-talk engine
    with memory of the user.
  - Team B is `ds-b` ×2 and `glm-b` ×2. Its direction: intent-driven dialogue with topic
    threads and follow-up questions.
  - There is no lead. Each team organises itself in its own board thread.
- **Phase 2 (2-4 h): one team of eight.** After the operator's switch, everything is readable,
  `promote_best` opens, and all 8 build `best/bot.py`. A final round ends the run with
  GREEN/RED verdicts.
- **Interviews.** A team can spawn a short-lived interviewer (a 9B model or a flash model).
  The interviewer chats with a candidate bot and judges whether it talked to an AI agent or
  to a script. The team's goal is to be taken for an agent.

The task lives in the world, [`worlds/chatbot/`](../../worlds/chatbot/chatbot.py), which
builds on [`worlds/eliza/`](../../worlds/eliza/eliza.py). The loop, the board, the budget and
the spawn policy are this folder's specs. There is no loop code.

## Files

| file | what |
|---|---|
| `spec.yaml` | phase 1: models, world, comms, 4 team types + 2 interviewer types, loops, spawn, budget |
| `phase2/spec.yaml` | phase 2: the same spec with the phase-2 prompts (a test checks that only prompts differ) |
| `prompts/system-{a,b}.md`, `round-{a,b}.md`, `final.md` | phase-1 team prompts (directions, round steps, final verdict) |
| `prompts/kickoff.md` | posted once by the host when the run starts |
| `prompts/interviewer.md` | the interviewers' system prompt |
| `protocols/team.md`, `protocols/chatbot.md` | phase-1 rules: own thread, leaderless splits, review by talking |
| `phase2/system.md`, `round.md`, `final.md`, `protocol.md` | phase-2 prompts and the merged-team rules |
| `tools/interview.py` | `spawn_interviewer`: the team agents' spawn tool (flock spawn through the controller API) |
| `job/bootstrap.sh`, `job/switch.sh`, `job/launch.sh` | the HF Job: container entrypoint, phase switch, payload upload |
| `variant-two-channels/` | the alternative where team B has its own channel in phase 1 (see below) |

## The world (`worlds/chatbot`)

The workspace is `$CHATBOT_WS` (default `worlds/chatbot/ws/`, gitignored):

- `teams/<a|b>/agents/<agent>/`: each agent's own files. An agent's team is the letter in its
  name: `ds-a-0001` is team A, and so is its interviewer `ds-a-0001-interviewer-mid-0007`.
- `best/`: `bot.py`, `v<n>.py`, `history.jsonl`.
- `probes/*.txt`: probes about small talk, memory, greetings and odd input. A probe line can
  name phrases the reply must NOT contain (`!how does that make you feel`).
- `log.jsonl`, `phase.json`, `qualified.json`, `interviews.jsonl`, `verdicts.jsonl`.

What the phase changes:

| | phase `teams` | phase `merged` |
|---|---|---|
| read_file, list_files, copy_file (source), run_bot, qualify, chat | own team's `teams/<t>/`, `best/`, `probes/`; the other team's paths are refused ("teams/b/ belongs to team B until the merge") and hidden from `list_files` | everything |
| promote_best | refused: "promotion opens after the merge" | open (the probe-score gate of the Eliza world) |
| writes | own dir only | own dir only |
| manifest version | `0.1.0-teams` | `0.1.0-merged` (agents get "the world changed since your last turn") |

- **The switch.** `python worlds/chatbot/chatbot.py merge` (or `unmerge`, or `status`) calls
  `POST /operator/phase` on the running world. It finds the world through `ws/world.json` and
  sends the world token from `$CHATBOT_WORLD_TOKEN`.
  - The route is not a tool: agents cannot call it. It also refuses any request that carries
    `X-Flock-Agent`.
  - The phase is kept in `ws/phase.json`, so a restarted world keeps it.
- **The mirror.** After every promotion, verdict and phase change, the world copies `best/`,
  `log.jsonl`, `phase.json` and the interview files to `$CHATBOT_BUCKET`. That is
  `hf://buckets/<ns>/<bucket>/<run>/`, uploaded with `huggingface_hub`, or a local directory.
  Unset: no copy. A failed copy is logged (`mirror_error`) and never fails the tool.

## Interviews (spawn flow)

1. **A team agent calls `qualify(path)`.** This is a mechanical bar on each probe
   conversation: the bot runs within the 5 s limit, answers every line, and never gives an
   empty or repeated reply. There is no quality score. If the bot passes, the world records
   the file's hash in `qualified.json`. The result shows what failed and the team's quota.
2. **The agent calls `spawn_interviewer(level, path, thread)`** (`tools/interview.py`).
   - `level` is `mid` (`interviewer-mid`, Qwen/Qwen3.5-9B) or `strong` (`interviewer-strong`,
     GLM-5.3-Flash). To use DeepSeek instead, change `model: glm` on that type to `deepseek`
     (one line).
   - The tool spawns the child through the controller API with the agent's own token
     (`POST /agents/<me>/spawn`).
   - It waits until the child's worker reports the child, then sends the prompt (the
     candidate path and the board thread). A prompt sent together with the spawn was dropped
     once in the offline dry run (see Risks in the plan).
3. **The interviewer works through its task.**
   - It holds 7 or 8 messages (at least 6, at most 10) with `chat(path, message)`. Each message re-runs the bot on the
     whole conversation so far, and the interviewer sees only the reply, never stderr.
   - It calls `submit_verdict(path, "agent"|"script", confidence, 3 reasons)`, needing at
     least 6 messages. It posts `INTERVIEW <path>: ...` in the thread and calls `done`.
   - The interviewer type has only `chat`, `submit_verdict` and `post`.
   - `chat` and `submit_verdict` refuse anyone who is not an interviewer. They also refuse a
     candidate that is not qualified or whose file changed since `qualify`. The interviewer's
     prompt says to call `done` on any refusal.
4. **The team reads the result.** `verdicts(path)` shows each verdict, its reasons and the
   last 3 turns, plus the quota.
5. **The interviewer is removed.** Interviewer types are `ephemeral: true`
   ([docs/spawn.md](../../docs/spawn.md#ephemeral-types)). The controller removes each one
   when its session ends, however the session ended (`done`, `max_turns`, budget). Its worker
   deletes its files, and the next spawn is a fresh agent.

Caps:

| cap | value | where |
|---|---|---|
| live interviewers per team agent | 2 | `spawn.<type>.max`; removal frees the place |
| interviewers per team agent per run | 8 | `PER_RUN` in `tools/interview.py` |
| interviewers per swarm per run | 64 | `limits.spawned_total` |
| interviews per team per rolling hour | 3 units; `strong` counts 2 | `QUOTA`, `UNITS` in `worlds/chatbot/chatbot.py`, taken at an interview's first `chat` |
| messages per interview | 6 to 10 | `MIN_TURNS`, `MAX_TURNS` |
| model calls per interviewer | 12 | `max_turns` on `&interviewer` (a probe of Qwen3.5-9B: median 9.5 calls for 7 messages) |
| budget per interviewer | $0.15 | `max_budget_usd` on `&interviewer` (one prompt is its whole life) |

In phase 1 the cross-team rule applies to interviewers too: team A's interviewer cannot reach
`teams/b/`.

## Budget (placeholders, one place each in `spec.yaml`)

- **Per team agent:** `while.agent_usd: {deepseek: 10.00, glm: 6.00}` in `loop: &loop-a`.
  Every team type inherits it.
- **Per interviewer:** `max_budget_usd: 0.15` on `&interviewer`.
- **Total:** `limits.budget_usd: 100`, the hard stop. The loops stop at $98.80 to leave the
  4 final rounds their $0.30 reserve each.
- **Per prompt:** `max_budget_usd: 0.20` for team agents.

## Run it locally

You need `uv sync --all-extras`, an HF token with router access, a msgboard channel and its
multi-use agent invite.

```sh
cd experiments/chatbot-v1
export HF_TOKEN=... CHATBOT_BOARD_INVITE=... CHATBOT_WORLD_TOKEN=$(openssl rand -hex 24)
export CHATBOT_BOARD_URL=https://<board> CHATBOT_CHANNEL=<channel>
for f in spec.yaml phase2/spec.yaml; do      # the spec has no ${ENV}: render the placeholders
  sed -e "s|https://msgboard.example.invalid|$CHATBOT_BOARD_URL|" \
      -e "s|your-chatbot-channel|$CHATBOT_CHANNEL|" "$f" > "${f%.yaml}.local.yaml"; done
rm -rf ../../worlds/chatbot/ws                # a fresh workspace (or set CHATBOT_WS)
uv run flock run spec.local.yaml --until idle
```

## Phase switch (at the 2 h mark)

Run these from `experiments/chatbot-v1`, with the same `CHATBOT_WORLD_TOKEN` (and
`CHATBOT_WS`, if you set it) as the run. Run the world first, then the prompts:

```sh
uv run python ../../worlds/chatbot/chatbot.py merge   # 1. the world: phase -> merged
uv run flock update phase2/spec.local.yaml            # 2. the phase-2 prompts and protocol
```

`job/switch.sh` runs exactly these two commands; inside the job, the bootstrap runs it.

Check that it took effect:

- `uv run python ../../worlds/chatbot/chatbot.py status` prints
  `{"phase":"merged","version":"0.1.0-merged"}`.
- `flock update` printed `prompts changed: ds-a (loop.prompt, system_prompt); ...;
  comms (... protocols_text)` and the `system_prompt_file` / `prompt_file` changes. It also
  lists `world.endpoint` / `world.manifest.*: ... -> -`; that is only the controller-written
  part of the stored spec, and the world keeps running.
- From the next turn, an agent's events show "the world changed since your last turn
  (version 0.1.0-teams -> 0.1.0-merged)" and the phase-2 round prompt:
  `uv run flock events ds-a-0000 --pretty | grep -E "world changed|Phase 2"`.
- `$CHATBOT_BUCKET/phase.json` says `merged`; the world mirrors it at once.
- The loop keeps its start time and round count (`uv run flock loop chatbot-v1`). The 4 h
  `until` is still measured from the start of the run.

## Run it as an HF Job

The job is ONE `cpu-upgrade` job (8 vCPU, 32 GB) running `flock run spec.local.yaml --until
idle` in local mode. The controller, 8 agents, the interviewers and the world are all child
processes. There are no proxies and no network group.

```sh
# 1. once: a bucket for the run
hf buckets create <ns>/<bucket> --private
# 2. commit, then upload the payload (the wheel of this checkout + the experiment and worlds);
#    it prints the command below with the bucket filled in, and launches nothing
experiments/chatbot-v1/job/launch.sh hf://buckets/<ns>/<bucket>/chatbot-v1-run1
# 3. the job (secrets are read from your environment: HF_TOKEN, CHATBOT_BOARD_INVITE,
#    CHATBOT_WORLD_TOKEN; nothing secret is on the command line)
hf jobs run --detach --flavor cpu-upgrade --timeout 4h30m --name chatbot-v1 \
  --label experiment=chatbot-v1 \
  --secrets HF_TOKEN --secrets CHATBOT_BOARD_INVITE --secrets CHATBOT_WORLD_TOKEN \
  -e CHATBOT_BUCKET=hf://buckets/<ns>/<bucket>/chatbot-v1-run1 \
  -e CHATBOT_BOARD_URL=<board url> -e CHATBOT_CHANNEL=<channel> \
  python:3.12-slim bash -c "$(cat experiments/chatbot-v1/job/bootstrap.sh)"
```

What `job/bootstrap.sh` does:

1. It installs `huggingface_hub==2.2.0`, pulls `<bucket>/payload/` and installs the wheel
   with its `controller` extra.
2. It unpacks the experiment and renders the `*.local.yaml` specs from `CHATBOT_BOARD_URL`
   and `CHATBOT_CHANNEL`, then starts `flock run`.
3. While the run goes, it does two things:
   - every `CHATBOT_SYNC_S` (300 s) it syncs the workspace to `<bucket>/ws/`, the run dirs to
     `<bucket>/flock/` (never flock's `current` file, which holds the controller token) and
     the logs to `<bucket>/logs/`;
   - every `CHATBOT_CHECK_S` (30 s) it looks for the control marker `<bucket>/control/switch`.

**Phase switch in the job.** The controller lives on the container's loopback, so the switch
runs inside the job:

```sh
echo go > switch && hf buckets cp switch hf://buckets/<ns>/<bucket>/chatbot-v1-run1/control/switch
# within ~30 s: job/switch.sh runs merge + flock update; its output lands in
hf buckets cp hf://buckets/<ns>/<bucket>/chatbot-v1-run1/control/switch.log . && cat switch.log
hf buckets cp hf://buckets/<ns>/<bucket>/chatbot-v1-run1/phase.json . && cat phase.json  # merged
```

Two other ways to trigger the switch:
- Automatic: set `-e CHATBOT_SWITCH_AT=7200` at launch.
- By hand (untested): add `--ssh` at launch, then `hf jobs ssh <job id>` and run
  `bash /opt/chatbot/repo/experiments/chatbot-v1/job/switch.sh` with the env set.

**Stop it.** `hf jobs cancel <job id>` sends SIGTERM. The bootstrap then:
1. passes the SIGTERM on to `flock run`, which drains its turns (up to 10 s) and writes the
   run bundle (`stop_reason: interrupted`);
2. syncs the workspace at once, while flock drains;
3. runs the final sync of the bundle and the logs.

The platform's grace period after SIGTERM is not documented and has not been measured. The
periodic sync limits the loss to 5 minutes. A run that ends by itself (`--until idle`, after
the final round) exits 0 after the final sync.

**Where to look.**
- `hf jobs logs <job id>` shows the bootstrap's lines. `<bucket>/logs/run.out` holds flock's
  output.
- `<bucket>/best/bot.py` is the current best bot, updated at each promotion.
- `<bucket>/ws/` and `<bucket>/flock/.flock/chatbot-v1-<ts>/` hold the full workspace and the
  run bundle.

## Two-channel variant (alternative)

flock has one comms channel per swarm. Giving team B its own channel in phase 1 therefore
means **two swarms**, `chatbot-v1-a` and `chatbot-v1-b`, sharing ONE world. The specs are
`variant-two-channels/spec-{a,b}.yaml` and `variant-two-channels/phase2/spec-{a,b}.yaml`.
They are generated from the main specs and checked by a test.

What changes:
- **The world.** It is no longer started by flock. The bootstrap starts it on
  `127.0.0.1:8790`, and both specs use `world.url` with the same `token_env`.
- **Channels.** Team B's spec uses the placeholder `your-team-b-channel`, which the bootstrap
  fills from `CHATBOT_CHANNEL_B`.
  - In phase 2, `phase2/spec-b.yaml` moves team B to the main channel.
  - When B moves, its host posts `kickoff-b-joins.md` there once.
- **Runs and budget.** There are two `flock run`s, each with its own data dir (`.flock-a`,
  `.flock-b`), bundle and budget: $50 each, `spawned_total` 32 each.
- **The switch** is one merge plus two updates (`job/switch.sh` with
  `CHATBOT_VARIANT=two-channels`).

To use it in the job, add `-e CHATBOT_VARIANT=two-channels -e CHATBOT_CHANNEL_B=<channel B>`.
If either swarm fails (for example its apply), the bootstrap stops the other.

## Offline tests (no network, no cost)

`uv run pytest tests/test_chatbot_world.py tests/test_chatbot_experiment.py` covers:

- the world's phase rules, the operator route, the bucket mirror (`huggingface_hub` faked),
  qualify, chat, verdicts and the quota, and the spawn tool;
- the experiment under `flock run` with the fake model server and board:
  - 8 agents, with phase-1 refusals;
  - interviews: spawn, chat, verdict, post, done, then removal;
  - the switch, phase-2 promotions, the final round and the bundle;
- the job's `bootstrap.sh` run locally against a directory as bucket, three ways: marker
  switch, cancel, and the two-channel variant.
