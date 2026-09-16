#!/usr/bin/env bash
# Tiny official-path QA smoke for kimi-for-coding.
# This is an Edges evaluation smoke, not a LoCoMo benchmark claim.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# sets OUT_DIR / DATA_FILE_PATH without clobbering existing API keys
source scripts/env.sh

mkdir -p "${OUT_DIR}"

python3 task_eval/evaluate_qa.py \
    --data-file "${DATA_FILE_PATH}" \
    --out-file "${OUT_DIR}/locomo10_kimi_smoke.json" \
    --model kimi-for-coding \
    --sample-id conv-26 \
    --qa-per-category 1 \
    --batch-size 1
