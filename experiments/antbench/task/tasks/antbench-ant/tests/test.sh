#!/bin/bash
# Nothing to check in the sandbox: the episode's outcome is AntBench's own result, which the world
# job logs and keeps with its transcript (experiments/antbench/README.md in legion).
mkdir -p "${VERIFIER_DIR:-/logs/verifier}"
echo 0 > "${VERIFIER_DIR:-/logs/verifier}/reward.txt"
