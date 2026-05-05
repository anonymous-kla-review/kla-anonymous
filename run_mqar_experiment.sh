#!/bin/bash
set -e

# MQAR Experiment Reproduction Script
# Follows arXiv 2312.04927 protocol

# Fix CXXABI version mismatch for matplotlib/torchmetrics
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_PREFIX}/lib

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Configuration
PROJECT_ROOT=$(pwd)
DATA_ROOT="${PROJECT_ROOT}/data/mqar"
OUT_ROOT="${PROJECT_ROOT}/out/mqar"

# Models to test
MODELS=("RelaxedKaczmarzQNorm_MQAR" "RelaxedKaczmarz_MQAR" "Longhorn_MQAR" "Kaczmarz_MQAR" "GatedDeltaNet_MQAR")

# Experiment Conditions
# SEQ_LENS=(128 256 512 1024)
# KEY_LENS=(1 2 3) # 1=2-gram, 2=3-gram, 3=4-gram
# For faster reproduction demonstration, we default to a smaller set.
# Uncomment the above lines for full experiment.
SEQ_LENS=(256)
KEY_LENS=(1)

SEEDS=(42)

# Training Params
BATCH_SIZE=32
LR=1e-3
MAX_STEPS=10000
VAL_INTERVAL=200
SAVE_INTERVAL=1000
EARLY_STOP_PATIENCE=10

echo "Starting MQAR Experiment Reproduction..."
echo "Models: ${MODELS[@]}"
echo "Seq Lens: ${SEQ_LENS[@]}"
echo "Key Lens: ${KEY_LENS[@]}"
echo "Data Root: $DATA_ROOT"
echo "Out Root: $OUT_ROOT"

for seq_len in "${SEQ_LENS[@]}"; do
    for key_len in "${KEY_LENS[@]}"; do
        for seed in "${SEEDS[@]}"; do

            # 1. Data Generation
            DATA_DIR="${DATA_ROOT}/seq${seq_len}_key${key_len}_seed${seed}"
            if [ ! -f "$DATA_DIR/train.npz" ]; then
                echo "Generating data for Seq=${seq_len}, KeyLen=${key_len}, Seed=${seed}..."
                python mqar_data.py \
                    --save_dir "$DATA_DIR" \
                    --num_train 20000 \
                    --num_val 2000 \
                    --num_test 2000 \
                    --seq_len $seq_len \
                    --key_len $key_len \
                    --seed $seed
            else
                echo "Data already exists at $DATA_DIR"
            fi

            # 2. Training
            for model in "${MODELS[@]}"; do
                EXP_NAME="${model}_seq${seq_len}_key${key_len}_seed${seed}"
                OUT_DIR="${OUT_ROOT}/${EXP_NAME}"

                if [ -f "${OUT_DIR}/results.json" ]; then
                    echo "Experiment ${EXP_NAME} already completed. Skipping."
                    continue
                fi

                echo "Running Training for ${EXP_NAME}..."
                python train_mqar.py \
                    --data_dir "$DATA_DIR" \
                    --out_dir "$OUT_DIR" \
                    --model_name "$model" \
                    --exp_name "$EXP_NAME" \
                    --seed $seed \
                    --batch_size $BATCH_SIZE \
                    --learning_rate $LR \
                    --max_steps $MAX_STEPS \
                    --val_interval $VAL_INTERVAL \
                    --save_interval $SAVE_INTERVAL \
                    --early_stop_patience $EARLY_STOP_PATIENCE \
                    --wandb_dir "${OUT_ROOT}/wandb"

                echo "Finished ${EXP_NAME}"
            done
        done
    done
done

echo "All experiments completed."
