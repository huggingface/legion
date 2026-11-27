#!/usr/bin/env bash
# The trusted world job of an AntBench run (README.md), on python:3.12-slim, in the run's network
# group as `antbench`: AntBench on port 8000, which only the workers' gateways reach (nothing is
# exposed). legion's launcher starts it with each ant's credential in ANT_TOKEN_<i>, derived from
# ANTBENCH_ENV_SECRET as the gateways derive them; the secret itself stays out of AntBench. The
# transcript stays in /trusted; once the episode ends, its result goes to the log and the
# transcript to /out/$RUN_NAME. The job ends when AntBench exits.
# Mounts: /antbench (the AntBench Space), /spec (the run's experiment folder), /wheel (legion's
# wheel and harbor-deps.tar, scripts/publish-wheel.sh), /out (a bucket folder, read-write).
# Env: ANTBENCH_ENV_SECRET (secret), RUN_NAME.
set -euo pipefail
if [ "${1:-}" != serve ]; then
  : "${ANTBENCH_ENV_SECRET:?}" "${RUN_NAME:?}"
  python3 -m pip install -q --no-cache-dir --root-user-action=ignore uv
  mkdir -p /opt/deps
  tar -xf /wheel/harbor-deps.tar -C /opt/deps
  echo "msgboard @ file://$(ls /opt/deps/msgboard-*.whl)" > /opt/deps/overrides.txt
  uv pip install -q --system --no-cache "$(ls /wheel/legion-*.whl)" \
    -c /opt/deps/constraints.txt --overrides /opt/deps/overrides.txt
  exec python3 -m legion.harbor.env /spec/spec.yaml -- bash "$0" serve
fi

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1  # /antbench is read-only
mkdir -p /trusted && chmod 700 /trusted
# one ant per agent: the launcher set one ANT_TOKEN_<i> per seat of the spec; lifters default to half
ANT_COUNT=$(compgen -v ANT_TOKEN_ | wc -l)
export ANT_COUNT
(cd /antbench && ANT_COMMS=text ANT_AUTO_START=1 \
  ANT_TRANSCRIPT_OUT=/trusted/episode.jsonl PORT=8000 exec python3 ant_service.py) &
ant=$!

# The watcher: the result at the end, and the deadline: 300 s after the first ant is ready, it
# readies the ants still missing (once), so one agent that never comes up blocks no one.
{ python3 - <<'PY' &&
import json, os, time, urllib.request

TOKENS = [os.environ[f"ANT_TOKEN_{i}"] for i in range(int(os.environ["ANT_COUNT"]))]
DEADLINE_S, ROUNDS, BACKOFF_S = int(os.environ.get("ANT_DEADLINE_S", 300)), 3, 2.0

def call(path, token, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:8000{path}", None if body is None else json.dumps(body).encode(),
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

def fill(missing, run_id):
    """Ready each missing seat as itself; seats that fail are tried again, ROUNDS in all."""
    filled, failed = [], {}
    for round_ in range(ROUNDS):
        time.sleep(BACKOFF_S * round_)
        failed = {}
        for i in missing:
            if i in filled:
                continue
            try:
                call("/v1/ready", TOKENS[i], {"run_id": run_id})
                filled.append(i)
            except (OSError, ValueError) as e:  # the type only: a URL error could quote a header
                failed[i] = type(e).__name__
        if not failed:
            break
    print(f"deadline: ready filled in after {DEADLINE_S} s for {len(filled)} of {len(TOKENS)} "
          f"ants: {filled}" + (f"; failed: {failed}" if failed else ""), flush=True)

first, filled_in = None, False
while True:
    try:
        seen = [call("/v1/observe", t) for t in TOKENS]
        if seen[0]["done"]:
            break
        if first is None and any(o["ready"] for o in seen):
            first = time.monotonic()
        if first is not None and not filled_in and time.monotonic() - first >= DEADLINE_S:
            filled_in = True
            if missing := [i for i, o in enumerate(seen) if not o["ready"]]:
                fill(missing, seen[0]["run_id"])
    except (OSError, ValueError):  # not up yet
        pass
    time.sleep(15)
print("antbench result:", json.dumps(call("/v1/result", TOKENS[0])))
PY
  mkdir -p "/out/$RUN_NAME" &&
  cp /trusted/episode.jsonl "/out/$RUN_NAME/" &&
  echo "transcript: /out/$RUN_NAME/episode.jsonl"
} || echo "antbench result: not saved (above)" &  # in the background: AntBench keeps serving
wait "$ant"  # serve until the job is cancelled or times out, or AntBench exits
