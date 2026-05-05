#!/usr/bin/env bash
set -euo pipefail

# Synthetic S-NIAH proxy training using MQAR with a single key-value pair.
# This is a low-cost fallback when zero-shot RULER S-NIAH is too hard for
# pretrain-only checkpoints.

# Fix CXXABI version mismatch for matplotlib/torchmetrics.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CONDA_PREFIX}/lib"

cd "$(dirname "$0")"

PROJECT_ROOT="$(pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/sniah_synth}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/out/sniah_synth}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Model aliases for synthetic-task-sized configs.
MODELS_CSV="${MODELS:-RelaxedKaczmarzQNorm_MQAR,GatedDeltaNet_MQAR,Mamba2_MQAR}"
SEQ_LENS_CSV="${SEQ_LENS:-1024}"
SEEDS_CSV="${SEEDS:-42}"

KEY_LEN="${KEY_LEN:-1}"
NUM_PAIRS="${NUM_PAIRS:-1}"
EXTRAPOL_FACTORS="${EXTRAPOL_FACTORS:-1,2,4,8}"

PROFILE="${PROFILE:-standard}"
case "${PROFILE}" in
  quick)
    NUM_TRAIN="${NUM_TRAIN:-5000}"
    NUM_VAL="${NUM_VAL:-500}"
    NUM_TEST="${NUM_TEST:-500}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    LR="${LR:-1e-3}"
    MAX_STEPS="${MAX_STEPS:-3000}"
    VAL_INTERVAL="${VAL_INTERVAL:-200}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
    ;;
  standard)
    NUM_TRAIN="${NUM_TRAIN:-20000}"
    NUM_VAL="${NUM_VAL:-2000}"
    NUM_TEST="${NUM_TEST:-2000}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    LR="${LR:-1e-3}"
    MAX_STEPS="${MAX_STEPS:-10000}"
    VAL_INTERVAL="${VAL_INTERVAL:-200}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
    ;;
  *)
    echo "ERROR: Unsupported PROFILE='${PROFILE}'. Use quick or standard."
    exit 1
    ;;
esac

IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
IFS=',' read -r -a SEQ_LENS <<< "${SEQ_LENS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

echo "Starting synthetic S-NIAH proxy experiments..."
echo "  Models: ${MODELS_CSV}"
echo "  Seq Lens: ${SEQ_LENS_CSV}"
echo "  Key Len: ${KEY_LEN}"
echo "  Num Pairs: ${NUM_PAIRS}"
echo "  Extrapolation Factors: ${EXTRAPOL_FACTORS}"
echo "  Seeds: ${SEEDS_CSV}"
echo "  Profile: ${PROFILE}"
echo "  Data Root: ${DATA_ROOT}"
echo "  Out Root: ${OUT_ROOT}"
echo "  Python Bin: ${PYTHON_BIN}"

validate_npz_file() {
  local npz_path="$1"
  if [[ ! -f "${npz_path}" ]]; then
    return 1
  fi

  "${PYTHON_BIN}" - "${npz_path}" <<'PY' >/dev/null 2>&1
import sys
import numpy as np

path = sys.argv[1]
try:
    loaded = np.load(path)
    # Force decompression/read to catch truncated/corrupted zip blocks.
    _ = loaded["input_ids"].shape
    _ = loaded["labels"].shape
    _ = loaded["masks"].shape
except Exception:
    raise SystemExit(1)

raise SystemExit(0)
PY
}

dataset_is_ready() {
  local data_dir="$1"
  validate_npz_file "${data_dir}/train.npz" && \
  validate_npz_file "${data_dir}/val.npz" && \
  validate_npz_file "${data_dir}/test.npz"
}

for seq_len in "${SEQ_LENS[@]}"; do
  seq_len="${seq_len// /}"
  [[ -z "${seq_len}" ]] && continue

  for seed in "${SEEDS[@]}"; do
    seed="${seed// /}"
    [[ -z "${seed}" ]] && continue

    DATA_DIR="${DATA_ROOT}/seq${seq_len}_key${KEY_LEN}_pairs${NUM_PAIRS}_seed${seed}"
    if ! dataset_is_ready "${DATA_DIR}"; then
      echo "Dataset missing/corrupted at ${DATA_DIR}; regenerating train/val/test npz..."
      mkdir -p "${DATA_DIR}"
      rm -f "${DATA_DIR}/train.npz" "${DATA_DIR}/val.npz" "${DATA_DIR}/test.npz"
      echo "Generating synthetic S-NIAH proxy data for seq=${seq_len}, seed=${seed}..."
      "${PYTHON_BIN}" mqar_data.py \
        --save_dir "${DATA_DIR}" \
        --num_train "${NUM_TRAIN}" \
        --num_val "${NUM_VAL}" \
        --num_test "${NUM_TEST}" \
        --seq_len "${seq_len}" \
        --key_len "${KEY_LEN}" \
        --num_pairs "${NUM_PAIRS}" \
        --seed "${seed}"
    else
      echo "Data already exists and passed integrity check at ${DATA_DIR}"
    fi

    for model in "${MODELS[@]}"; do
      model="${model// /}"
      [[ -z "${model}" ]] && continue

      EXP_NAME="${model}_sniah_seq${seq_len}_key${KEY_LEN}_pairs${NUM_PAIRS}_seed${seed}_${MAX_STEPS}steps"
      OUT_DIR="${OUT_ROOT}/${EXP_NAME}"

      if [[ -f "${OUT_DIR}/results.json" ]]; then
        echo "Experiment ${EXP_NAME} already completed. Skipping."
        continue
      fi

      echo "Running training for ${EXP_NAME}..."
      "${PYTHON_BIN}" train_mqar.py \
        --data_dir "${DATA_DIR}" \
        --out_dir "${OUT_DIR}" \
        --model_name "${model}" \
        --exp_name "${EXP_NAME}" \
        --seed "${seed}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LR}" \
        --max_steps "${MAX_STEPS}" \
        --val_interval "${VAL_INTERVAL}" \
        --save_interval "${SAVE_INTERVAL}" \
        --early_stop_patience "${EARLY_STOP_PATIENCE}" \
        --num_pairs "${NUM_PAIRS}" \
        --extrapolation_factors "${EXTRAPOL_FACTORS}" \
        --extrapol_base_seq_len "${seq_len}" \
        --extrapol_key_len "${KEY_LEN}" \
        --extrapol_num_val "${NUM_VAL}" \
        --wandb_dir "${OUT_ROOT}/wandb"

      echo "Finished ${EXP_NAME}"
    done
  done
done

echo "All synthetic S-NIAH proxy experiments completed."
