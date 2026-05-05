#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
PROJECT_ROOT="${PROJECT_ROOT:-${ROOT}}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/save_dir}"

PYTHON_BIN="${PYTHON_BIN:-python}"
VAL_DATA_DIR="${VAL_DATA_DIR:-${PROJECT_ROOT}/mydata/slimpajama/validation}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/out/long_context}"
ANALYSIS_SAVE_DIR="${ANALYSIS_SAVE_DIR:-${PROJECT_ROOT}/analysis_results/pretrain}"

LENGTHS="${LENGTHS:-1024,2048,4096,8192,16384,32768,65536}"
EVAL_ITERS="${EVAL_ITERS:-15}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-1337}"
DEVICE="${DEVICE:-cuda}"
MODELS="${MODELS:-KLA,GDN,Mamba2}"
BASELINE_LENGTH="${BASELINE_LENGTH:-4096}"

# Set to 0 to keep wandb logging.
DISABLE_WANDB="${DISABLE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-linear_attn_eval}"
WANDB_DIR="${WANDB_DIR:-${OUT_ROOT}/wandb}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
if [[ -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ ! -d "${VAL_DATA_DIR}" ]]; then
  echo "ERROR: VAL_DATA_DIR does not exist: ${VAL_DATA_DIR}"
  exit 1
fi

mkdir -p "${OUT_ROOT}"
mkdir -p "${ANALYSIS_SAVE_DIR}"
mkdir -p "${WANDB_DIR}"

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
  echo "Model: ${model}"
  echo "Config: ${config_name}"
  echo "Checkpoint: ${ckpt_path}"
  echo "Lengths: ${LENGTHS}"
  echo "=================================================="

  cmd=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/eval_context_length.py"
    --ckpt_path "${ckpt_path}"
    --config_name "${config_name}"
    --data_dir "${VAL_DATA_DIR}"
    --lengths "${LENGTHS}"
    --batch_size "${BATCH_SIZE}"
    --eval_iters "${EVAL_ITERS}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_name "lc_${model}"
    --wandb_dir "${WANDB_DIR}"
    --model_alias "${model}"
    --save_json "${model_out_dir}/results.json"
    --save_csv "${model_out_dir}/results.csv"
  )

  if [[ "${DISABLE_WANDB}" == "1" ]]; then
    cmd+=(--disable_wandb)
  fi

  "${cmd[@]}"
  EVALUATED=$((EVALUATED + 1))
done

if [[ "${EVALUATED}" -eq 0 ]]; then
  echo "ERROR: No model was evaluated."
  exit 1
fi

echo "\nRunning long-context extrapolation aggregation..."
"${PYTHON_BIN}" "${PROJECT_ROOT}/long_context_extrapolation_analysis.py" \
  --out_root "${OUT_ROOT}" \
  --save_dir "${ANALYSIS_SAVE_DIR}" \
  --baseline_length "${BASELINE_LENGTH}" \
  --models "${MODELS}"

echo "Done. Artifacts saved in:"
echo "  - Raw evals: ${OUT_ROOT}"
echo "  - Analysis: ${ANALYSIS_SAVE_DIR}"
