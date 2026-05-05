#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
PROJECT_ROOT="${PROJECT_ROOT:-${ROOT}}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/save_dir}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/out/efficiency}"
ANALYSIS_SAVE_DIR="${ANALYSIS_SAVE_DIR:-${PROJECT_ROOT}/analysis_results/efficiency}"

PROFILE="${PROFILE:-kimi_standard}"

case "${PROFILE}" in
  kimi_standard)
    PREFILL_LENGTHS="${PREFILL_LENGTHS:-4096,8192,16384,32768,65536,131072,262144,524288,1048576}"
    DECODE_LENGTHS="${DECODE_LENGTHS:-4096,8192,16384,32768,65536,131072,262144,524288,1048576}"
    PREFILL_BATCH_SIZE="${PREFILL_BATCH_SIZE:-1}"
    DECODE_BATCH_SIZES="${DECODE_BATCH_SIZES:-1}"
    DECODE_NEW_TOKENS="${DECODE_NEW_TOKENS:-128}"
    WARMUP_ITERS="${WARMUP_ITERS:-3}"
    BENCHMARK_ITERS="${BENCHMARK_ITERS:-10}"
    MAX_PREFILL_LATENCY_MS="${MAX_PREFILL_LATENCY_MS:-60000}"
    MAX_DECODE_LATENCY_MS="${MAX_DECODE_LATENCY_MS:-30000}"
    ;;
  kimi_short)
    PREFILL_LENGTHS="${PREFILL_LENGTHS:-4096,8192,16384,32768,65536,131072}"
    DECODE_LENGTHS="${DECODE_LENGTHS:-4096,8192,16384,32768}"
    PREFILL_BATCH_SIZE="${PREFILL_BATCH_SIZE:-1}"
    DECODE_BATCH_SIZES="${DECODE_BATCH_SIZES:-1}"
    DECODE_NEW_TOKENS="${DECODE_NEW_TOKENS:-32}"
    WARMUP_ITERS="${WARMUP_ITERS:-1}"
    BENCHMARK_ITERS="${BENCHMARK_ITERS:-3}"
    MAX_PREFILL_LATENCY_MS="${MAX_PREFILL_LATENCY_MS:-20000}"
    MAX_DECODE_LATENCY_MS="${MAX_DECODE_LATENCY_MS:-12000}"
    ;;
  legacy)
    PREFILL_LENGTHS="${PREFILL_LENGTHS:-1024,2048,4096,8192,16384,32768,65536}"
    DECODE_LENGTHS="${DECODE_LENGTHS:-4096,8192,16384,32768,65536}"
    PREFILL_BATCH_SIZE="${PREFILL_BATCH_SIZE:-1}"
    DECODE_BATCH_SIZES="${DECODE_BATCH_SIZES:-1,2,4}"
    DECODE_NEW_TOKENS="${DECODE_NEW_TOKENS:-128}"
    WARMUP_ITERS="${WARMUP_ITERS:-3}"
    BENCHMARK_ITERS="${BENCHMARK_ITERS:-10}"
    MAX_PREFILL_LATENCY_MS="${MAX_PREFILL_LATENCY_MS:-0}"
    MAX_DECODE_LATENCY_MS="${MAX_DECODE_LATENCY_MS:-0}"
    ;;
  *)
    echo "ERROR: Unsupported PROFILE='${PROFILE}'. Use kimi_standard, kimi_short, or legacy."
    exit 1
    ;;
esac

SEED="${SEED:-1337}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MODELS="${MODELS:-KLA,GDN,Mamba2}"

PLOT_PREFILL_BATCH="${PLOT_PREFILL_BATCH:-${PREFILL_BATCH_SIZE}}"
PLOT_DECODE_BATCH="${PLOT_DECODE_BATCH:-1}"
PLOT_NEW_TOKENS="${PLOT_NEW_TOKENS:-${DECODE_NEW_TOKENS}}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
if [[ -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "${OUT_ROOT}"
mkdir -p "${ANALYSIS_SAVE_DIR}"

declare -A RUN_DIR_MAP
declare -A CONFIG_MAP

RUN_DIR_MAP[KLA]="tsz512x4k_512x4k_1B_RelaxedKaczmarzQNorm_0.4B"
RUN_DIR_MAP[GDN]="tsz512x4k_512x4k_1B_GatedDeltaNet_0.4B"
RUN_DIR_MAP[Mamba2]="tsz512x4k_512x4k_1B_Mamba2_0.4B"
RUN_DIR_MAP[GLA]="tsz512x4k_512x4k_1B_GLA_0.4B"
RUN_DIR_MAP[DeltaNet]="tsz512x4k_512x4k_1B_DeltaNet_0.4B"
RUN_DIR_MAP[Longhorn]="tsz512x4k_512x4k_1B_Longhorn_0.4B"

CONFIG_MAP[KLA]="RelaxedKaczmarzQNorm_0.4B"
CONFIG_MAP[GDN]="GatedDeltaNet_0.4B"
CONFIG_MAP[Mamba2]="Mamba2_0.4B"
CONFIG_MAP[GLA]="GLA_0.4B"
CONFIG_MAP[DeltaNet]="DeltaNet_0.4B"
CONFIG_MAP[Longhorn]="Longhorn_0.4B"

IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"
EVALUATED=0

for model in "${MODEL_LIST[@]}"; do
  model="${model// /}"
  if [[ -z "${model}" ]]; then
    continue
  fi

  if [[ -z "${RUN_DIR_MAP[${model}]:-}" || -z "${CONFIG_MAP[${model}]:-}" ]]; then
    echo "WARNING: Unknown model alias '${model}', skip. Supported: KLA,GDN,Mamba2,GLA,DeltaNet,Longhorn"
    continue
  fi

  run_dir="${RUN_DIR_MAP[${model}]}"
  config_name="${CONFIG_MAP[${model}]}"

  ckpt_path="${SAVE_DIR}/outputs/${run_dir}/final-model-ckpt.pth"
  if [[ ! -f "${ckpt_path}" ]]; then
    fallback_ckpt="${SAVE_DIR}/outputs/${run_dir}/latest-model-ckpt.pth"
    if [[ -f "${fallback_ckpt}" ]]; then
      ckpt_path="${fallback_ckpt}"
    else
      echo "WARNING: No checkpoint found for ${model}, skip. Checked:"
      echo "  - ${SAVE_DIR}/outputs/${run_dir}/final-model-ckpt.pth"
      echo "  - ${SAVE_DIR}/outputs/${run_dir}/latest-model-ckpt.pth"
      continue
    fi
  fi

  model_out_dir="${OUT_ROOT}/${model}"
  mkdir -p "${model_out_dir}"

  echo "=================================================="
  echo "Profile           : ${PROFILE}"
  echo "Model             : ${model}"
  echo "Config            : ${config_name}"
  echo "Checkpoint        : ${ckpt_path}"
  echo "Prefill lengths   : ${PREFILL_LENGTHS}"
  echo "Decode lengths    : ${DECODE_LENGTHS}"
  echo "Decode batch sizes: ${DECODE_BATCH_SIZES}"
  echo "Decode new tokens : ${DECODE_NEW_TOKENS}"
  echo "Warmup / bench    : ${WARMUP_ITERS} / ${BENCHMARK_ITERS}"
  echo "Prefill cap (ms)  : ${MAX_PREFILL_LATENCY_MS}"
  echo "Decode cap (ms)   : ${MAX_DECODE_LATENCY_MS}"
  echo "=================================================="

  "${PYTHON_BIN}" "${PROJECT_ROOT}/efficiency_prefill_decode_benchmark.py" \
    --ckpt_path "${ckpt_path}" \
    --config_name "${config_name}" \
    --model_alias "${model}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --seed "${SEED}" \
    --prefill_lengths "${PREFILL_LENGTHS}" \
    --decode_lengths "${DECODE_LENGTHS}" \
    --prefill_batch_size "${PREFILL_BATCH_SIZE}" \
    --decode_batch_sizes "${DECODE_BATCH_SIZES}" \
    --decode_new_tokens "${DECODE_NEW_TOKENS}" \
    --warmup_iters "${WARMUP_ITERS}" \
    --benchmark_iters "${BENCHMARK_ITERS}" \
    --max_prefill_latency_ms "${MAX_PREFILL_LATENCY_MS}" \
    --max_decode_latency_ms "${MAX_DECODE_LATENCY_MS}" \
    --save_json "${model_out_dir}/results.json" \
    --save_csv "${model_out_dir}/results.csv"

  EVALUATED=$((EVALUATED + 1))
done

if [[ "${EVALUATED}" -eq 0 ]]; then
  echo "ERROR: No model was evaluated."
  exit 1
fi

echo ""
echo "Running efficiency aggregation..."
"${PYTHON_BIN}" "${PROJECT_ROOT}/efficiency_experiment_analysis.py" \
  --out_root "${OUT_ROOT}" \
  --save_dir "${ANALYSIS_SAVE_DIR}" \
  --models "${MODELS}" \
  --plot_prefill_batch "${PLOT_PREFILL_BATCH}" \
  --plot_decode_batch "${PLOT_DECODE_BATCH}" \
  --plot_new_tokens "${PLOT_NEW_TOKENS}"

echo "Done. Artifacts saved in:"
echo "  - Raw evals: ${OUT_ROOT}"
echo "  - Analysis : ${ANALYSIS_SAVE_DIR}"
