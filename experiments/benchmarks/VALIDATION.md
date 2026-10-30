# Validation — 2026-10-30

Implementation and preparation only. **No model inference, Slurm submission, or paid workload
was launched.** The Terminal-Bench adapter is implemented but its real Docker execution gate
is blocked on this host.

| Check | Result |
| --- | --- |
| New bridge/campaign/supervisor tests | 15 passed |
| Final targeted regression set, including World, Pi, Agent Escape and Slurm tests | 92 passed, 13 skipped |
| Repository-wide Ruff | Passed |
| `git diff --check` | Passed |
| Four prepared manifest/spec validations | Passed |
| Slurm submission dry run | Four array elements, `%1` concurrency; no submission |
| Real Agent Escape via OpenEnv and World API | Passed |
| Real Terminal-Bench oracle/invalid-submission gate | Blocked: Docker socket permission denied |
| Full repository pytest (`--maxfail=1`) | 234 passed, then one unrelated chatbot timeout failure |

Agent Escape's two-agent reference gate uses the pinned upstream HTTP server and a real OpenEnv
WebSocket session. It compares observations, investigation and answer responses against a separate
direct-upstream reference session. A wrong answer fails and the evaluator-only reference answer
succeeds. This checks adapter behaviour, not model capability.

The Terminal-Bench source task was downloaded from the official Harbor registry at commit
`69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`. Its instruction/configuration loaded with Harbor
0.25.0. Docker preflight returned:

```text
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

The full-suite failure is
`tests/test_chatbot3_world.py::test_deadline_drops_the_calls_runs_and_frees_the_slots`:
its final `run_bot` call exceeds the test's 0.3-second deadline. The same failure reproduces
when running that test alone. Neither that test nor `worlds/chatbot3/chatbot3.py` was changed.
The full suite stopped after this failure; the remaining tests were not run. The full repository gate is not green. The user subsequently requested a draft PR for review;
the draft records this failure and does not imply merge readiness.

## Review bundle

The final prepared campaign is:

```text
.flock/openenv-pilot-ready-20261030/campaign.json
```

It contains Agent Escape difficulty 5 / scenario index 0 and Terminal-Bench `db-wal-recovery`,
each at 1 and 2 agents, one repetition. Every attempt has its own timestamped board name,
`manifest.json`, `spec.yaml`, and prompt. The two-agent attempts also contain `reference.json`
recording the pass/block above. All manifests validate against the final source hashes.

The checked-in starting points are [pilot.yaml](pilot.yaml) and
[profile-pi.yaml](profile-pi.yaml); execution and review instructions are in [README.md](README.md).
The router URL is retained from the earlier configuration and has not been probed in this turn.
