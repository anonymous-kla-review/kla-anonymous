#!/bin/bash
set -e

# Palindrome Experiment Script
# Compare RelaxedKaczmarzQNorm, GatedDeltaNet, and Mamba2 on reversal recall.

# Fix CXXABI version mismatch for matplotlib/torchmetrics
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_PREFIX}/lib

# Ensure we are in the script directory
cd "$(dirname "$0")"

PROJECT_ROOT=$(pwd)
DATA_ROOT="${PROJECT_ROOT}/data/palindrome"
OUT_ROOT="${PROJECT_ROOT}/out/palindrome"

MODELS=(
  "RelaxedKaczmarzQNorm_Palindrome"
  "GatedDeltaNet_Palindrome"
  "Mamba2_Palindrome"
)

# Training source sequence lengths.
# Keep a single train length by default, then evaluate on multiple lengths.
TRAIN_SEQ_LENS=(1024)

# Final evaluation lengths used for "Accuracy vs Sequence length".
EVAL_SEQ_LENS="256,512,1024,2048"
SEEDS=(42)

NUM_TRAIN=20000
NUM_VAL=2000
NUM_TEST=2000
VOCAB_SIZE=128
PREDICT_FIRST_TOKEN=1

# Kimi synthetic setting uses 2 layers, 2 heads (head dim 128 when n_embd=256).
N_HEAD=2

BATCH_SIZE=32
LR=1e-3
MAX_STEPS=20000
VAL_INTERVAL=500
SAVE_INTERVAL=1000
EARLY_STOP_PATIENCE=20

echo "Starting Palindrome experiments..."
echo "Models: ${MODELS[*]}"
echo "Train Seq Lens: ${TRAIN_SEQ_LENS[*]}"
echo "Eval Seq Lens: ${EVAL_SEQ_LENS}"
echo "Predict First Token: ${PREDICT_FIRST_TOKEN}"
echo "Data Root: $DATA_ROOT"
echo "Out Root: $OUT_ROOT"

PREDICT_FIRST_TOKEN_ARGS=()
PREDICT_FIRST_TOKEN_TAG="pf0"
if [ "${PREDICT_FIRST_TOKEN}" -eq 1 ]; then
  PREDICT_FIRST_TOKEN_ARGS+=(--predict_first_token)
  PREDICT_FIRST_TOKEN_TAG="pf1"
fi

for seq_len in "${TRAIN_SEQ_LENS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    DATA_DIR="${DATA_ROOT}/seq${seq_len}_v${VOCAB_SIZE}_${PREDICT_FIRST_TOKEN_TAG}_seed${seed}"
    LEGACY_DATA_DIR="${DATA_ROOT}/seq${seq_len}_seed${seed}"

    if [ -d "${LEGACY_DATA_DIR}" ] && [ "${LEGACY_DATA_DIR}" != "${DATA_DIR}" ]; then
      echo "Found legacy cache at ${LEGACY_DATA_DIR}."
      echo "Using vocab-scoped cache ${DATA_DIR} to avoid vocab mismatch."
    fi

    if [ ! -f "${DATA_DIR}/train.npz" ]; then
      echo "Generating palindrome data for Seq=${seq_len}, Seed=${seed}..."
      python palindrome_data.py \
        --save_dir "${DATA_DIR}" \
        --num_train ${NUM_TRAIN} \
        --num_val ${NUM_VAL} \
        --num_test ${NUM_TEST} \
        --seq_len ${seq_len} \
        --seed ${seed} \
        --vocab_size ${VOCAB_SIZE} \
        "${PREDICT_FIRST_TOKEN_ARGS[@]}"
    else
      echo "Data already exists at ${DATA_DIR}"
    fi

    for model in "${MODELS[@]}"; do
      EXP_NAME="${model}_seq${seq_len}_v${VOCAB_SIZE}_${PREDICT_FIRST_TOKEN_TAG}_h${N_HEAD}_seed${seed}_${MAX_STEPS}steps"
      OUT_DIR="${OUT_ROOT}/${EXP_NAME}"

      if [ -f "${OUT_DIR}/results.json" ]; then
        echo "Experiment ${EXP_NAME} already completed. Skipping."
        continue
      fi

      echo "Running training for ${EXP_NAME}..."
      python train_palindrome.py \
        --data_dir "${DATA_DIR}" \
        --out_dir "${OUT_DIR}" \
        --model_name "${model}" \
        --exp_name "${EXP_NAME}" \
        --seed ${seed} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LR} \
        --max_steps ${MAX_STEPS} \
        --val_interval ${VAL_INTERVAL} \
        --save_interval ${SAVE_INTERVAL} \
        --early_stop_patience ${EARLY_STOP_PATIENCE} \
        --eval_seq_lens "${EVAL_SEQ_LENS}" \
        --eval_num_val ${NUM_VAL} \
        --vocab_size ${VOCAB_SIZE} \
        --n_head ${N_HEAD} \
        "${PREDICT_FIRST_TOKEN_ARGS[@]}" \
        --wandb_dir "${OUT_ROOT}/wandb"

      echo "Finished ${EXP_NAME}"
    done
  done
done

echo "All palindrome experiments completed."
