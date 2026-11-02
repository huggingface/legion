# The run-3 load smoke as a cpu-basic HF Job (README "Load smoke"). Passed inline:
#   hf jobs run ... python:3.12-slim bash -c "$(cat experiments/run3/load/job.sh)"
# env:     OPENAI_BASE_URL, MODEL_WORKER; LS_B64 (load_smoke.py, gzip + base64: `gzip -9c load_smoke.py | base64 -w0`); LS_OUT (hf://buckets/...
#          or a local dir: where load-smoke.md + results.json go); LS_ARGS (extra flags)
# secrets: HF_TOKEN (the bucket), OPENAI_API_KEY (only if the server has --api-key)
# LS_PREINSTALLED=1: httpx and hf are already there (a local rehearsal).
set -euo pipefail
: "${OPENAI_BASE_URL:?}" "${MODEL_WORKER:?}" "${LS_B64:?}" "${LS_OUT:?}"
if [ -z "${LS_PREINSTALLED:-}" ]; then
  python3 -m pip install -q --no-cache-dir --root-user-action=ignore httpx "huggingface_hub==2.2.0"
fi
W=$(mktemp -d)
echo "$LS_B64" | base64 -d | gunzip > "$W/load_smoke.py"
rc=0
"${PYTHON:-python3}" "$W/load_smoke.py" --out "$W/out" ${LS_ARGS:-} || rc=$?
if [[ $LS_OUT == hf://* ]]; then hf buckets sync "$W/out" "${LS_OUT%/}" -q >/dev/null
else mkdir -p "$LS_OUT" && cp -a "$W/out/." "$LS_OUT/"; fi
echo "load smoke: exit $rc; results in $LS_OUT"
exit "$rc"
