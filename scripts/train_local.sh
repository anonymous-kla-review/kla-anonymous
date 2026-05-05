#!/bin/bash

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_ROOT="${PROJECT_ROOT}"
TRAIN_DATA="${OUTPUT_ROOT}/mydata/slimpajama/train"
VALIDATION_DATA="${OUTPUT_ROOT}/mydata/slimpajama/validation"
SAVE_DIR="${OUTPUT_ROOT}/save_dir"

# Experiment settings
FULL_TRAIN_TOKENS=1000000000
NAME="${NAME:-512x4k_1B_GLA_0.4B}"
MODEL="${MODEL:-GLA_0.4B}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-tsz512x4k}"
EVAL_ITERS="${EVAL_ITERS:-15}"
TOTAL_EVALS="${TOTAL_EVALS:-100}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
MAX_TOKENS="${MAX_TOKENS:-$FULL_TRAIN_TOKENS}"
EVAL_STEP_INTERVAL="${EVAL_STEP_INTERVAL:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Directories
LOGS_DIR="${SAVE_DIR}/logs/${NAME}/"
WANDB_DIR="${SAVE_DIR}/wandb/${NAME}/"
TRI_CACHE_DIR="${SAVE_DIR}/triton/${NAME}/"

export PYTHONPATH="${OUTPUT_ROOT}":$PYTHONPATH
export TRITON_CACHE_DIR="${TRI_CACHE_DIR}"
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
unset WANDB_DISABLED
export WANDB_MODE=online
export WANDB_PROJECT="linear-attn-new" # 可以在这里修改为你想要的 wandb project 名称
export WANDB_DIR="${WANDB_DIR}"

PYTHON_BIN="python"

# Fail fast if FLA is missing, since GLA_0.4B depends on fla.ops.simple_gla.
if [[ "${MODEL}" == GLA* ]]; then
  ${PYTHON_BIN} -c "from fla.ops.simple_gla import chunk_simple_gla" >/dev/null 2>&1 || {
    echo "ERROR: MODEL=${MODEL} requires FLA, but 'from fla.ops.simple_gla import chunk_simple_gla' failed."
    echo "Please install in anonymous_project env: conda run -n anonymous_project python -m pip install -U flash-linear-attention"
    exit 1
  }
fi

# Fail fast if Mamba2 dependencies are missing.
if [[ "${MODEL}" == Mamba2* ]]; then
  ${PYTHON_BIN} -c "from mamba_ssm.modules.mamba2 import Mamba2; from mamba_ssm.ops.triton.layer_norm import RMSNorm; from causal_conv1d import causal_conv1d_fn" >/dev/null 2>&1 || {
    echo "ERROR: MODEL=${MODEL} requires mamba-ssm and causal-conv1d, but imports failed."
    echo "Please install in anonymous_project env: conda run -n anonymous_project python -m pip install -U mamba-ssm causal-conv1d"
    exit 1
  }
fi

# Create directories
mkdir -p ${LOGS_DIR}
mkdir -p ${WANDB_DIR}
mkdir -p ${TRI_CACHE_DIR}

echo "Starting training..."
echo "Model: ${MODEL}"
echo "Config: ${CONFIG}"
echo "Output Root: ${OUTPUT_ROOT}"
echo "Train Data: ${TRAIN_DATA}"
echo "Validation Data: ${VALIDATION_DATA}"
echo "Max Tokens: ${MAX_TOKENS}"
echo "Total Evals: ${TOTAL_EVALS}"
echo "W&B Mode: ${WANDB_MODE}"
echo "W&B Dir: ${WANDB_DIR}"
echo "Extra Args: ${EXTRA_ARGS}"

${PYTHON_BIN} -u ${OUTPUT_ROOT}/pretrain.py \
    --train_data_dir ${TRAIN_DATA} \
    --val_data_dir ${VALIDATION_DATA} \
    --output_root ${SAVE_DIR} \
    --exp_name ${NAME} \
    --model_name ${MODEL} \
    --train_config ${CONFIG} \
    --eval_iters ${EVAL_ITERS} \
    --total_evals ${TOTAL_EVALS} \
    --eval_step_interval ${EVAL_STEP_INTERVAL} \
    --learning_rate ${LR} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --max_tokens ${MAX_TOKENS} \
    --seed ${SEED} \
    ${EXTRA_ARGS}
