#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Standard S-NIAH: full NIAH configs, KLA/GDN/Mamba2 only.
# Use a dedicated dataset path to avoid reusing ruler_selflong.jsonl
# that may include CWE/FWE tasks from previous runs.
DATASET_PRESET="${DATASET_PRESET:-ruler_selflong}"
DATASET_OUTPUT="${DATASET_OUTPUT:-./data/long_context_tasks/real/ruler_sniah_standard.jsonl}"
BENCHMARK_NAME="${BENCHMARK_NAME:-RULER_SNIAH_STANDARD}"
RULER_CONFIGS="${RULER_CONFIGS:-niah_single_1_4k,niah_multikey_1_4k,niah_multiquery_4k,niah_multivalue_4k}"
PREP_MAX_SAMPLES="${PREP_MAX_SAMPLES:-0}"
MODELS="${MODELS:-KLA,GDN,Mamba2}"
PROFILE="${PROFILE:-kimi_standard}"

echo "Running standard S-NIAH benchmark"
echo "  DATASET_PRESET=${DATASET_PRESET}"
echo "  DATASET_OUTPUT=${DATASET_OUTPUT}"
echo "  BENCHMARK_NAME=${BENCHMARK_NAME}"
echo "  RULER_CONFIGS=${RULER_CONFIGS}"
echo "  PREP_MAX_SAMPLES=${PREP_MAX_SAMPLES}"
echo "  MODELS=${MODELS}"
echo "  PROFILE=${PROFILE}"

DATASET_PRESET="${DATASET_PRESET}" \
DATASET_OUTPUT="${DATASET_OUTPUT}" \
BENCHMARK_NAME="${BENCHMARK_NAME}" \
RULER_CONFIGS="${RULER_CONFIGS}" \
PREP_MAX_SAMPLES="${PREP_MAX_SAMPLES}" \
MODELS="${MODELS}" \
PROFILE="${PROFILE}" \
./run_long_context_task_benchmark.sh
