#!/bin/bash
set -e

# Stack Experiment Script
# Compare RelaxedKaczmarzQNorm, GatedDeltaNet, and Mamba2 on multi-stack tracking.

# Fix CXXABI version mismatch for matplotlib/torchmetrics
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_PREFIX}/lib

cd "$(dirname "$0")"

PROJECT_ROOT=$(pwd)
DATA_ROOT="${PROJECT_ROOT}/data/stack"
OUT_ROOT="${PROJECT_ROOT}/out/stack"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODELS=(
  "RelaxedKaczmarzQNorm_Stack"
  "GatedDeltaNet_Stack"
  "Mamba2_Stack"
)

# Report setup: fixed training length 1024, evaluate 256-2048.
TRAIN_SEQ_LENS=(1024)
EVAL_SEQ_LENS="256,512,1024,2048"
SEEDS=(42)

NUM_TRAIN=20000
NUM_VAL=2000
NUM_TEST=2000
VOCAB_SIZE=128
NUM_STACKS=64
NUM_VALUES=26
PUSH_PROB=0.6

# Kimi synthetic setting uses 2 layers, 2 heads (head dim 128 when n_embd=256).
N_HEAD=2

BATCH_SIZE=32
MAX_STEPS=20000
VAL_INTERVAL=500
SAVE_INTERVAL=1000
EARLY_STOP_PATIENCE=20

# Default: no LR sweep. Set ENABLE_LR_SWEEP=1 to enable report-style grid search.
ENABLE_LR_SWEEP="${ENABLE_LR_SWEEP:-0}"
if [[ "${ENABLE_LR_SWEEP}" == "1" ]]; then
  LRS=(5e-5 1e-4 5e-4 1e-3)
else
  LRS=(1e-3)
fi

echo "Starting Stack experiments..."
echo "Models: ${MODELS[*]}"
echo "Train Seq Lens: ${TRAIN_SEQ_LENS[*]}"
echo "Eval Seq Lens: ${EVAL_SEQ_LENS}"
echo "Data Root: ${DATA_ROOT}"
echo "Out Root: ${OUT_ROOT}"
echo "Python Bin: ${PYTHON_BIN}"
echo "Enable LR Sweep: ${ENABLE_LR_SWEEP}"
echo "Learning Rates: ${LRS[*]}"

for seq_len in "${TRAIN_SEQ_LENS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    DATA_DIR="${DATA_ROOT}/seq${seq_len}_v${VOCAB_SIZE}_s${NUM_STACKS}_val${NUM_VALUES}_seed${seed}"

    if [ ! -f "${DATA_DIR}/train.npz" ]; then
      echo "Generating stack data for Seq=${seq_len}, Seed=${seed}..."
      ${PYTHON_BIN} stack_data.py \
        --save_dir "${DATA_DIR}" \
        --num_train ${NUM_TRAIN} \
        --num_val ${NUM_VAL} \
        --num_test ${NUM_TEST} \
        --seq_len ${seq_len} \
        --seed ${seed} \
        --vocab_size ${VOCAB_SIZE} \
        --num_stacks ${NUM_STACKS} \
        --num_values ${NUM_VALUES} \
        --push_prob ${PUSH_PROB}
    else
      echo "Data already exists at ${DATA_DIR}"
    fi

    for model in "${MODELS[@]}"; do
      for lr in "${LRS[@]}"; do
        LR_TAG="${lr//./p}"
        LR_TAG="${LR_TAG//-/m}"

        EXP_NAME="${model}_seq${seq_len}_v${VOCAB_SIZE}_s${NUM_STACKS}_val${NUM_VALUES}_seed${seed}_lr${LR_TAG}_${MAX_STEPS}steps"
        OUT_DIR="${OUT_ROOT}/${EXP_NAME}"

        if [ -f "${OUT_DIR}/results.json" ]; then
          echo "Experiment ${EXP_NAME} already completed. Skipping."
          continue
        fi

        echo "Running training for ${EXP_NAME}..."
        ${PYTHON_BIN} train_stack.py \
          --data_dir "${DATA_DIR}" \
          --out_dir "${OUT_DIR}" \
          --model_name "${model}" \
          --exp_name "${EXP_NAME}" \
          --seed ${seed} \
          --batch_size ${BATCH_SIZE} \
          --learning_rate ${lr} \
          --max_steps ${MAX_STEPS} \
          --val_interval ${VAL_INTERVAL} \
          --save_interval ${SAVE_INTERVAL} \
          --early_stop_patience ${EARLY_STOP_PATIENCE} \
          --eval_seq_lens "${EVAL_SEQ_LENS}" \
          --eval_num_val ${NUM_VAL} \
          --vocab_size ${VOCAB_SIZE} \
          --num_stacks ${NUM_STACKS} \
          --num_values ${NUM_VALUES} \
          --push_prob ${PUSH_PROB} \
          --n_head ${N_HEAD} \
          --wandb_dir "${OUT_ROOT}/wandb"

        echo "Finished ${EXP_NAME}"
      done
    done
  done
done

echo "All stack experiments completed."
