# Job entrypoint, passed inline as `bash -c "$(cat job_bootstrap.sh)"` on image node:22-bookworm.
# Installs Claude Code + hf, pulls experiments/ from the bucket, runs $JOB_SCRIPT $JOB_ARGS.
set -euo pipefail
t0=$(date +%s.%N)
export PATH="$HOME/.local/bin:$PATH" DEBIAN_FRONTEND=noninteractive
command -v python3 >/dev/null || { apt-get update -qq && apt-get install -y -qq python3 >/dev/null; }
npm i -g --silent "@anthropic-ai/claude-code@${CLAUDE_VERSION:-2.1.296}" >/dev/null 2>&1
t1=$(date +%s.%N)
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv tool install -q "huggingface_hub==${HF_HUB_VERSION:-2.1.1}" >/dev/null 2>&1
t2=$(date +%s.%N)
hf buckets sync "hf://buckets/cmpatino/flock/scripts/${SCRIPTS_REV:?}" /scripts -q >/dev/null
echo "bootstrap: claude=$(claude --version) node=$(node --version) python=$(python3 --version) $(hf version 2>/dev/null | head -1)"
python3 -c "import sys; t=[float(x) for x in sys.argv[1:]]; print(f'bootstrap seconds: npm_claude={t[1]-t[0]:.1f} uv_hf={t[2]-t[1]:.1f} total={t[3]-t[0]:.1f}')" "$t0" "$t1" "$t2" "$(date +%s.%N)"
echo "nproc=$(nproc) mem=$(free -m | awk '/Mem/{print $2}')MB user=$(id -un)"
cd /scripts
exec bash "/scripts/${JOB_SCRIPT:?}" ${JOB_ARGS:-}
