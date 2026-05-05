#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
PROJECT_ROOT="${PROJECT_ROOT:-${ROOT}}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/save_dir}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/out/long_context_tasks}"
ANALYSIS_SAVE_DIR="${ANALYSIS_SAVE_DIR:-${PROJECT_ROOT}/analysis_results/long_context_tasks}"

DATASET_PRESET="${DATASET_PRESET:-ruler_selflong}"
DATASET_OUTPUT="${DATASET_OUTPUT:-${PROJECT_ROOT}/data/long_context_tasks/real/${DATASET_PRESET}.jsonl}"
AUTO_PREP_DATASET="${AUTO_PREP_DATASET:-1}"
ALLOW_DEMO_DATASET="${ALLOW_DEMO_DATASET:-0}"
AUTO_LINK_REAL_DATA="${AUTO_LINK_REAL_DATA:-1}"
REAL_DATA_LINK_TARGET="${REAL_DATA_LINK_TARGET:-${PROJECT_ROOT}/mydata/long_context_tasks/real}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${PROJECT_ROOT}/mydata/hf_cache/datasets}"

RULER_CONFIGS="${RULER_CONFIGS:-niah_single_1_4k,niah_multikey_1_4k,niah_multiquery_4k,niah_multivalue_4k,cwe_4k,fwe_4k}"
RULER_SPLIT="${RULER_SPLIT:-validation}"
LONGBENCH_LENGTHS="${LONGBENCH_LENGTHS:-long}"
LONGBENCH_DOMAINS="${LONGBENCH_DOMAINS:-}"
LONGBENCH_INCLUDE_CHOICE_TEXT="${LONGBENCH_INCLUDE_CHOICE_TEXT:-0}"
MRCR_NEEDLES="${MRCR_NEEDLES:-2}"
MRCR_MAX_NEW_TOKENS="${MRCR_MAX_NEW_TOKENS:-256}"
PREP_MAX_SAMPLES="${PREP_MAX_SAMPLES:-0}"
PREP_SHUFFLE="${PREP_SHUFFLE:-0}"

PROFILE="${PROFILE:-kimi_standard}"

case "${PROFILE}" in
  kimi_standard)
    LENGTH_BUCKETS="${LENGTH_BUCKETS:-4096,8192,16384,32768,65536,131072}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
    MAX_SAMPLES="${MAX_SAMPLES:-0}"
    LOG_EVERY="${LOG_EVERY:-20}"
    ;;
  kimi_quick)
    LENGTH_BUCKETS="${LENGTH_BUCKETS:-4096,8192,16384,32768}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
    MAX_SAMPLES="${MAX_SAMPLES:-200}"
    LOG_EVERY="${LOG_EVERY:-10}"
    ;;
  *)
    echo "ERROR: Unsupported PROFILE='${PROFILE}'. Use kimi_standard or kimi_quick."
    exit 1
    ;;
esac

case "${DATASET_PRESET}" in
  ruler_selflong)
    DEFAULT_BENCHMARK_NAME="RULER_REAL"
    ;;
  longbench_v2)
    DEFAULT_BENCHMARK_NAME="LONGBENCH_V2"
    ;;
  mrcr_openai)
    DEFAULT_BENCHMARK_NAME="MRCR_REAL"
    ;;
  custom)
    DEFAULT_BENCHMARK_NAME="CUSTOM"
    ;;
  *)
    echo "ERROR: Unsupported DATASET_PRESET='${DATASET_PRESET}'."
    echo "Supported presets: ruler_selflong, longbench_v2, mrcr_openai, custom"
    exit 1
    ;;
esac

BENCHMARK_NAME="${BENCHMARK_NAME:-${DEFAULT_BENCHMARK_NAME}}"
TASK_NAME="${TASK_NAME:-}"
INPUT_JSONL="${INPUT_JSONL:-${DATASET_OUTPUT}}"

MODELS="${MODELS:-KLA,GDN,Mamba2}"
SEED="${SEED:-1337}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

TOKENIZER_NAME="${TOKENIZER_NAME:-TinyLlama/TinyLlama_v1.1}"
TOKENIZER_DIR="${TOKENIZER_DIR:-}"

MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-0}"
PROMPT_TRUNCATION="${PROMPT_TRUNCATION:-none}"
STOP_STRINGS="${STOP_STRINGS:-}"
ALLOW_OOM_SKIP="${ALLOW_OOM_SKIP:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
LONGBENCH_SAFE_MAX_PROMPT_TOKENS="${LONGBENCH_SAFE_MAX_PROMPT_TOKENS:-32768}"
ALLOW_UNBOUNDED_LONGBENCH_PROMPT="${ALLOW_UNBOUNDED_LONGBENCH_PROMPT:-0}"

if [[ "${DATASET_PRESET}" == "longbench_v2" && "${MAX_PROMPT_TOKENS}" == "0" && "${ALLOW_UNBOUNDED_LONGBENCH_PROMPT}" != "1" ]]; then
  MAX_PROMPT_TOKENS="${LONGBENCH_SAFE_MAX_PROMPT_TOKENS}"
  if [[ "${PROMPT_TRUNCATION}" == "none" ]]; then
    PROMPT_TRUNCATION="left"
  fi
  echo "WARNING: LongBench-v2 long prompts can exceed CUDA kernel stability limits."
  echo "         Auto-set MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS}, PROMPT_TRUNCATION=${PROMPT_TRUNCATION}."
  echo "         Set ALLOW_UNBOUNDED_LONGBENCH_PROMPT=1 to keep full prompts (may crash)."
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
if [[ -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

REAL_DATA_DIR="${PROJECT_ROOT}/data/long_context_tasks/real"
if [[ "${AUTO_LINK_REAL_DATA}" == "1" ]]; then
  if [[ ! -e "${REAL_DATA_DIR}" && -e "${PROJECT_ROOT}/mydata" ]]; then
    mkdir -p "${REAL_DATA_LINK_TARGET}"
    ln -s "${REAL_DATA_LINK_TARGET}" "${REAL_DATA_DIR}"
    echo "Linked real dataset dir: ${REAL_DATA_DIR} -> ${REAL_DATA_LINK_TARGET}"
  fi
fi

if [[ -n "${HF_CACHE_DIR}" ]]; then
  mkdir -p "${HF_CACHE_DIR}"
fi

if [[ ! -f "${INPUT_JSONL}" ]]; then
  if [[ "${AUTO_PREP_DATASET}" != "1" ]]; then
    echo "ERROR: INPUT_JSONL does not exist: ${INPUT_JSONL}"
    echo "Set AUTO_PREP_DATASET=1 to auto-prepare from real benchmark preset."
    exit 1
  fi

  if [[ "${DATASET_PRESET}" == "custom" ]]; then
    echo "ERROR: DATASET_PRESET=custom cannot auto-prepare INPUT_JSONL."
    echo "Please provide INPUT_JSONL manually."
    exit 1
  fi

  echo "Preparing real benchmark dataset..."
  prep_cmd=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/prepare_long_context_task_data.py"
    --preset "${DATASET_PRESET}"
    --output_jsonl "${INPUT_JSONL}"
    --max_samples "${PREP_MAX_SAMPLES}"
    --seed "${SEED}"
  )

  if [[ -n "${HF_CACHE_DIR}" ]]; then
    prep_cmd+=(--hf_cache_dir "${HF_CACHE_DIR}")
  fi

  if [[ "${PREP_SHUFFLE}" == "1" ]]; then
    prep_cmd+=(--shuffle)
  fi

  case "${DATASET_PRESET}" in
    ruler_selflong)
      prep_cmd+=(
        --ruler_configs "${RULER_CONFIGS}"
        --hf_split "${RULER_SPLIT}"
      )
      ;;
    longbench_v2)
      prep_cmd+=(
        --longbench_lengths "${LONGBENCH_LENGTHS}"
      )
      if [[ -n "${LONGBENCH_DOMAINS}" ]]; then
        prep_cmd+=(--longbench_domains "${LONGBENCH_DOMAINS}")
      fi
      if [[ "${LONGBENCH_INCLUDE_CHOICE_TEXT}" == "1" ]]; then
        prep_cmd+=(--longbench_include_choice_text)
      fi
      ;;
    mrcr_openai)
      prep_cmd+=(
        --mrcr_needles "${MRCR_NEEDLES}"
        --mrcr_max_new_tokens "${MRCR_MAX_NEW_TOKENS}"
      )
      ;;
  esac

  "${prep_cmd[@]}"
fi

if [[ "${ALLOW_DEMO_DATASET}" != "1" ]]; then
  if grep -q '"id"[[:space:]]*:[[:space:]]*"demo_' "${INPUT_JSONL}"; then
    echo "ERROR: INPUT_JSONL appears to be a demo dataset: ${INPUT_JSONL}"
    echo "For paper results, run with real presets (default) or set ALLOW_DEMO_DATASET=1 to override."
    exit 1
  fi
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
BENCH_OUT_ROOT="${OUT_ROOT}/${BENCHMARK_NAME}"
mkdir -p "${BENCH_OUT_ROOT}"
BENCH_ANALYSIS_DIR="${ANALYSIS_SAVE_DIR}/${BENCHMARK_NAME}"
mkdir -p "${BENCH_ANALYSIS_DIR}"
FIGURE_STEM="${FIGURE_STEM:-$(echo "${BENCHMARK_NAME}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')}"

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

  model_out_dir="${BENCH_OUT_ROOT}/${model}"
  mkdir -p "${model_out_dir}"

  echo "=================================================="
  echo "Profile         : ${PROFILE}"
  echo "Dataset preset  : ${DATASET_PRESET}"
  echo "Benchmark       : ${BENCHMARK_NAME}"
  echo "Task            : ${TASK_NAME:-auto-from-dataset}"
  echo "Model           : ${model}"
  echo "Config          : ${config_name}"
  echo "Checkpoint      : ${ckpt_path}"
  echo "Input JSONL     : ${INPUT_JSONL}"
  echo "Length Buckets  : ${LENGTH_BUCKETS}"
  echo "Max New Tokens  : ${MAX_NEW_TOKENS}"
  echo "Max Samples     : ${MAX_SAMPLES}"
  echo "=================================================="

  cmd=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/long_context_task_benchmark.py"
    --ckpt_path "${ckpt_path}"
    --config_name "${config_name}"
    --input_jsonl "${INPUT_JSONL}"
    --benchmark_name "${BENCHMARK_NAME}"
    --model_alias "${model}"
    --tokenizer_name "${TOKENIZER_NAME}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --seed "${SEED}"
    --max_samples "${MAX_SAMPLES}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --length_buckets "${LENGTH_BUCKETS}"
    --max_prompt_tokens "${MAX_PROMPT_TOKENS}"
    --prompt_truncation "${PROMPT_TRUNCATION}"
    --log_every "${LOG_EVERY}"
    --save_json "${model_out_dir}/results.json"
    --save_csv "${model_out_dir}/results.csv"
  )

  if [[ -n "${TASK_NAME}" ]]; then
    cmd+=(--task_name "${TASK_NAME}")
  fi
  if [[ -n "${TOKENIZER_DIR}" ]]; then
    cmd+=(--tokenizer_dir "${TOKENIZER_DIR}")
  fi
  if [[ -n "${STOP_STRINGS}" ]]; then
    cmd+=(--default_stop_strings "${STOP_STRINGS}")
  fi
  if [[ "${ALLOW_OOM_SKIP}" == "1" ]]; then
    cmd+=(--allow_oom_skip)
  fi
  if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
    cmd+=(--save_predictions)
  fi

  "${cmd[@]}"
  EVALUATED=$((EVALUATED + 1))
done

if [[ "${EVALUATED}" -eq 0 ]]; then
  echo "ERROR: No model was evaluated."
  exit 1
fi

echo ""
echo "Running long-context task aggregation..."
"${PYTHON_BIN}" "${PROJECT_ROOT}/long_context_task_analysis.py" \
  --out_root "${BENCH_OUT_ROOT}" \
  --save_dir "${BENCH_ANALYSIS_DIR}" \
  --figure_stem "${FIGURE_STEM}" \
  --models "${MODELS}"

echo "Done. Artifacts saved in:"
echo "  - Raw evals: ${BENCH_OUT_ROOT}"
echo "  - Analysis : ${BENCH_ANALYSIS_DIR}"
