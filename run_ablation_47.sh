#!/usr/bin/env bash
set -euo pipefail

# 4.7 Ablation runner: MQAR + small pretrain
# Default profile runs the full 100M-token pretrain plan.

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CONDA_PREFIX}/lib"

cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
SAVE_DIR="${SAVE_DIR:-${ROOT}/save_dir}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE="${PROFILE:-standard}"
GROUPS_CSV="${ABLATION_GROUPS:-ab1_norm,ab2_seq_factor,ab3_gate,ab4_state}"
VARIANTS_CSV="${VARIANTS:-}"
DRY_RUN="${DRY_RUN:-0}"
AUTO_REPORT="${AUTO_REPORT:-1}"
PRETRAIN_MIN_MICRO_BATCH="${PRETRAIN_MIN_MICRO_BATCH:-1}"
PRETRAIN_MAX_RETRIES="${PRETRAIN_MAX_RETRIES:-4}"
FORCE_RERUN_MQAR="${FORCE_RERUN_MQAR:-0}"

SEED="${SEED:-42}"
MQAR_SEQ_LEN="${MQAR_SEQ_LEN:-256}"
MQAR_KEY_LEN="${MQAR_KEY_LEN:-1}"
MQAR_NUM_PAIRS="${MQAR_NUM_PAIRS:-32}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/mqar_ablation47}"
MQAR_OUT_ROOT="${MQAR_OUT_ROOT:-${PROJECT_ROOT}/out/ablation_47/mqar}"
ANALYSIS_SAVE_DIR="${ANALYSIS_SAVE_DIR:-${PROJECT_ROOT}/analysis_results/ablation_47}"

case "${PROFILE}" in
  quick)
    MQAR_NUM_TRAIN="${MQAR_NUM_TRAIN:-5000}"
    MQAR_NUM_VAL="${MQAR_NUM_VAL:-500}"
    MQAR_NUM_TEST="${MQAR_NUM_TEST:-500}"
    MQAR_BATCH_SIZE="${MQAR_BATCH_SIZE:-32}"
    MQAR_LR="${MQAR_LR:-1e-3}"
    MQAR_MAX_STEPS="${MQAR_MAX_STEPS:-2000}"
    MQAR_VAL_INTERVAL="${MQAR_VAL_INTERVAL:-200}"
    MQAR_SAVE_INTERVAL="${MQAR_SAVE_INTERVAL:-1000}"
    MQAR_EARLY_STOP="${MQAR_EARLY_STOP:-8}"

    PRETRAIN_MAX_TOKENS="${PRETRAIN_MAX_TOKENS:-5000000}"
    PRETRAIN_TOTAL_EVALS="${PRETRAIN_TOTAL_EVALS:-10}"
    PRETRAIN_EVAL_ITERS="${PRETRAIN_EVAL_ITERS:-8}"
    PRETRAIN_MICRO_BATCH="${PRETRAIN_MICRO_BATCH:-4}"
    PRETRAIN_LR="${PRETRAIN_LR:-1e-4}"
    ;;
  standard)
    MQAR_NUM_TRAIN="${MQAR_NUM_TRAIN:-20000}"
    MQAR_NUM_VAL="${MQAR_NUM_VAL:-2000}"
    MQAR_NUM_TEST="${MQAR_NUM_TEST:-2000}"
    MQAR_BATCH_SIZE="${MQAR_BATCH_SIZE:-32}"
    MQAR_LR="${MQAR_LR:-1e-3}"
    MQAR_MAX_STEPS="${MQAR_MAX_STEPS:-10000}"
    MQAR_VAL_INTERVAL="${MQAR_VAL_INTERVAL:-200}"
    MQAR_SAVE_INTERVAL="${MQAR_SAVE_INTERVAL:-1000}"
    MQAR_EARLY_STOP="${MQAR_EARLY_STOP:-10}"

    PRETRAIN_MAX_TOKENS="${PRETRAIN_MAX_TOKENS:-100000000}"
    PRETRAIN_TOTAL_EVALS="${PRETRAIN_TOTAL_EVALS:-20}"
    PRETRAIN_EVAL_ITERS="${PRETRAIN_EVAL_ITERS:-15}"
    PRETRAIN_MICRO_BATCH="${PRETRAIN_MICRO_BATCH:-4}"
    PRETRAIN_LR="${PRETRAIN_LR:-1e-4}"
    ;;
  *)
    echo "ERROR: Unsupported PROFILE='${PROFILE}'. Use quick or standard."
    exit 1
    ;;
esac

mkdir -p "${DATA_ROOT}" "${MQAR_OUT_ROOT}" "${ANALYSIS_SAVE_DIR}"

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

group_enabled() {
  local target="$1"
  local group
  IFS=',' read -r -a _groups <<< "${GROUPS_CSV}"
  for group in "${_groups[@]}"; do
    group="${group// /}"
    if [[ "${group}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

variant_enabled() {
  local target="$1"
  local variant
  if [[ -z "${VARIANTS_CSV}" ]]; then
    return 0
  fi
  IFS=',' read -r -a _variants <<< "${VARIANTS_CSV}"
  for variant in "${_variants[@]}"; do
    variant="${variant// /}"
    if [[ "${variant}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

format_tokens_tag() {
  local tokens="$1"
  if (( tokens % 1000000000 == 0 )); then
    echo "$((tokens / 1000000000))B"
  elif (( tokens % 1000000 == 0 )); then
    echo "$((tokens / 1000000))M"
  elif (( tokens % 1000 == 0 )); then
    echo "$((tokens / 1000))K"
  else
    echo "${tokens}"
  fi
}

TOKENS_TAG="$(format_tokens_tag "${PRETRAIN_MAX_TOKENS}")"
DATA_REGENERATED=0

select_pretrain_micro_batch() {
  local variant="$1"
  case "${variant}" in
    a4_expand_v16)
      echo "${PRETRAIN_MICRO_BATCH_A4_EXPAND_V16:-1}"
      ;;
    a4_expand_v8)
      echo "${PRETRAIN_MICRO_BATCH_A4_EXPAND_V8:-2}"
      ;;
    *)
      echo "${PRETRAIN_MICRO_BATCH}"
      ;;
  esac
}

verify_mqar_results() {
  local result_path="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" - <<'PY' "${result_path}"
import json
import os
import sys

path = sys.argv[1]
if not os.path.isfile(path):
    raise SystemExit(f"missing results.json: {path}")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

required_top = ["test_acc", "best_val_acc", "final_val_acc", "args"]
for k in required_top:
    if k not in data:
        raise SystemExit(f"results.json missing key: {k}")

final_val_acc = data.get("final_val_acc")
if not isinstance(final_val_acc, dict):
    raise SystemExit("final_val_acc must be a dict")

for factor in ("1x", "2x", "4x", "8x"):
    if factor not in final_val_acc:
        raise SystemExit(f"final_val_acc missing factor: {factor}")
PY
}

verify_pretrain_outputs() {
  local pretrain_exp_name="$1"
  local pretrain_run_dir="tsz512x4k_${pretrain_exp_name}"
  local ckpt_dir="${SAVE_DIR}/outputs/${pretrain_run_dir}"
  local wandb_root="${SAVE_DIR}/wandb/${pretrain_run_dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "${ckpt_dir}/final-model-ckpt.pth" && ! -f "${ckpt_dir}/latest-model-ckpt.pth" ]]; then
    return 1
  fi
  "${PYTHON_BIN}" - <<'PY' "${wandb_root}"
import glob
import json
import os
import sys

root = sys.argv[1]
pattern = os.path.join(root, "wandb", "run-*", "files", "wandb-summary.json")
paths = glob.glob(pattern)
if not paths:
    raise SystemExit(1)

target_key = "metric/val_ppl@1x"
for p in sorted(paths):
    try:
        with open(p, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if isinstance(summary, dict) and target_key in summary:
            raise SystemExit(0)
    except Exception:
        continue
raise SystemExit(1)
PY
}

run_pretrain_with_retry() {
  local variant="$1"
  local pretrain_exp_name="$2"
  local pre_model="$3"
  local pre_args="$4"

  local current_micro_batch
  current_micro_batch="$(select_pretrain_micro_batch "${variant}")"
  local attempt=1

  while (( attempt <= PRETRAIN_MAX_RETRIES )); do
    if (( current_micro_batch < PRETRAIN_MIN_MICRO_BATCH )); then
      break
    fi

    echo "Pretrain attempt ${attempt}/${PRETRAIN_MAX_RETRIES} | variant=${variant} | micro_batch=${current_micro_batch}"
    if run_cmd env \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
      NAME="${pretrain_exp_name}" \
      MODEL="${pre_model}" \
      CONFIG="tsz512x4k" \
      MAX_TOKENS="${PRETRAIN_MAX_TOKENS}" \
      TOTAL_EVALS="${PRETRAIN_TOTAL_EVALS}" \
      EVAL_ITERS="${PRETRAIN_EVAL_ITERS}" \
      MICRO_BATCH_SIZE="${current_micro_batch}" \
      LR="${PRETRAIN_LR}" \
      SEED="${SEED}" \
      SAVE_DIR="${SAVE_DIR}" \
      OUTPUT_ROOT="${PROJECT_ROOT}" \
      EXTRA_ARGS="${pre_args}" \
      bash "${PROJECT_ROOT}/scripts/train_local.sh"; then
      if verify_pretrain_outputs "${pretrain_exp_name}"; then
        echo "Pretrain succeeded and outputs verified for ${pretrain_exp_name}."
        return 0
      fi
      echo "Pretrain command finished but outputs are incomplete; retry with smaller micro batch."
    else
      echo "Pretrain failed (likely OOM or runtime error); retry with smaller micro batch."
    fi

    if (( current_micro_batch == PRETRAIN_MIN_MICRO_BATCH )); then
      break
    fi
    current_micro_batch=$(( current_micro_batch / 2 ))
    if (( current_micro_batch < PRETRAIN_MIN_MICRO_BATCH )); then
      current_micro_batch="${PRETRAIN_MIN_MICRO_BATCH}"
    fi
    attempt=$((attempt + 1))
  done

  return 1
}

DATA_DIR="${DATA_ROOT}/seq${MQAR_SEQ_LEN}_key${MQAR_KEY_LEN}_seed${SEED}"
if [[ ! -f "${DATA_DIR}/train.npz" || ! -f "${DATA_DIR}/val.npz" || ! -f "${DATA_DIR}/test.npz" ]]; then
  needs_regen=1
else
  if [[ "${DRY_RUN}" == "1" ]]; then
    needs_regen=0
  else
    if "${PYTHON_BIN}" - <<'PY' "${DATA_DIR}" "${MQAR_NUM_TRAIN}" "${MQAR_NUM_VAL}" "${MQAR_NUM_TEST}" "${MQAR_SEQ_LEN}"
import numpy as np
import os
import sys

root = sys.argv[1]
expected_train = int(sys.argv[2])
expected_val = int(sys.argv[3])
expected_test = int(sys.argv[4])
expected_seq = int(sys.argv[5])

checks = [("train.npz", expected_train), ("val.npz", expected_val), ("test.npz", expected_test)]
for fname, expected_n in checks:
    path = os.path.join(root, fname)
    data = np.load(path)
    x = data["input_ids"]
    if x.shape[0] != expected_n or x.shape[1] != expected_seq:
        raise SystemExit(1)
raise SystemExit(0)
PY
    then
      needs_regen=0
    else
      needs_regen=1
    fi
  fi
fi

if [[ "${needs_regen}" == "1" ]]; then
  echo "Regenerating MQAR data to match requested sizes/sequence length..."
  if [[ "${DRY_RUN}" != "1" ]]; then
    rm -rf "${DATA_DIR}"
  fi
  run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/mqar_data.py" \
    --save_dir "${DATA_DIR}" \
    --num_train "${MQAR_NUM_TRAIN}" \
    --num_val "${MQAR_NUM_VAL}" \
    --num_test "${MQAR_NUM_TEST}" \
    --seq_len "${MQAR_SEQ_LEN}" \
    --key_len "${MQAR_KEY_LEN}" \
    --num_pairs "${MQAR_NUM_PAIRS}" \
    --seed "${SEED}"
  DATA_REGENERATED=1
else
  echo "MQAR data already matches requested setup: ${DATA_DIR}"
fi

VARIANTS=(
  "ab1_norm|a1_kla|KLA|RelaxedKaczmarzQNorm_MQAR|RelaxedKaczmarzQNorm_0.4B|--qk_norm relaxed_kaczmarz_q_norm --gate_mode dual --seq_factor_mode none|--qk_norm relaxed_kaczmarz_q_norm --gate_mode dual --seq_factor_mode none"
  "ab1_norm|a1_no_norm|NoNorm|GatedDeltaNet_MQAR|GatedDeltaNet_0.4B|--qk_norm no_norm --gate_mode dual --seq_factor_mode none|--qk_norm no_norm --gate_mode dual --seq_factor_mode none"
  "ab1_norm|a1_k_norm_only|KNormOnly|GatedDeltaNet_MQAR|GatedDeltaNet_0.4B|--qk_norm k_norm_only --gate_mode dual --seq_factor_mode none|--qk_norm k_norm_only --gate_mode dual --seq_factor_mode none"
  "ab1_norm|a1_seq_only|SeqOnly1OverT|GatedDeltaNet_MQAR|GatedDeltaNet_0.4B|--qk_norm seq_only --gate_mode dual --seq_factor_mode none|--qk_norm seq_only --gate_mode dual --seq_factor_mode none"
  "ab1_norm|a1_learned_norm|LearnedNorm|GatedDeltaNet_MQAR|GatedDeltaNet_0.4B|--qk_norm learned_scalar --gate_mode dual --seq_factor_mode none --learned_norm_init 1.0|--qk_norm learned_scalar --gate_mode dual --seq_factor_mode none --learned_norm_init 1.0"

  "ab2_seq_factor|a2_no_seq_factor|NoSeqFactor|RelaxedKaczmarzQNorm_MQAR|RelaxedKaczmarzQNorm_0.4B|--qk_norm relaxed_kaczmarz_q_norm --seq_factor_mode none --gate_mode dual|--qk_norm relaxed_kaczmarz_q_norm --seq_factor_mode none --gate_mode dual"
  "ab2_seq_factor|a2_inv_t|OneOverT|RelaxedKaczmarzQNorm_MQAR|RelaxedKaczmarzQNorm_0.4B|--qk_norm relaxed_kaczmarz_q_norm --seq_factor_mode inv_t --gate_mode dual|--qk_norm relaxed_kaczmarz_q_norm --seq_factor_mode inv_t --gate_mode dual"
  "ab2_seq_factor|a2_inv_sqrt_t|OneOverSqrtT|RelaxedKaczmarzQNorm_MQAR|RelaxedKaczmarzQNorm_0.4B|--qk_norm relaxed_kaczmarz_q_norm --seq_factor_mode inv_sqrt_t --gate_mode dual|--qk_norm relaxed_kaczmarz_q_norm --seq_factor_mode inv_sqrt_t --gate_mode dual"
  "ab2_seq_factor|a2_inv_log_t|OneOverLogT|RelaxedKaczmarzQNorm_MQAR|RelaxedKaczmarzQNorm_0.4B|--qk_norm relaxed_kaczmarz_q_norm --seq_factor_mode inv_log_t --gate_mode dual|--qk_norm relaxed_kaczmarz_q_norm --seq_factor_mode inv_log_t --gate_mode dual"

  "ab3_gate|a3_kla_single_gate|KLA_SingleGate|RelaxedKaczmarzQNorm_MQAR|RelaxedKaczmarzQNorm_0.4B|--qk_norm relaxed_kaczmarz_q_norm --gate_mode single --seq_factor_mode none|--qk_norm relaxed_kaczmarz_q_norm --gate_mode single --seq_factor_mode none"
  "ab3_gate|a3_gdn_dual_gate|GDN_DualGate|GatedDeltaNet_MQAR|GatedDeltaNet_0.4B|--gate_mode dual --seq_factor_mode none|--gate_mode dual --seq_factor_mode none"
  "ab3_gate|a3_gla_independent_gate|GLA_IndependentGate|GLA_MQAR|GLA_0.4B||"

  "ab4_state|a4_expand_v2|StateExpandV2|RelaxedKaczmarzQNorm_MQAR|RelaxedKaczmarzQNorm_0.4B|--qk_norm relaxed_kaczmarz_q_norm --expand_v 2 --gate_mode dual --seq_factor_mode none|--qk_norm relaxed_kaczmarz_q_norm --expand_v 2 --gate_mode dual --seq_factor_mode none"
  "ab4_state|a4_expand_v4|StateExpandV4|RelaxedKaczmarzQNorm_MQAR|RelaxedKaczmarzQNorm_0.4B|--qk_norm relaxed_kaczmarz_q_norm --expand_v 4 --gate_mode dual --seq_factor_mode none|--qk_norm relaxed_kaczmarz_q_norm --expand_v 4 --gate_mode dual --seq_factor_mode none"
  "ab4_state|a4_expand_v8|StateExpandV8|RelaxedKaczmarzQNorm_MQAR|RelaxedKaczmarzQNorm_0.4B|--qk_norm relaxed_kaczmarz_q_norm --expand_v 8 --gate_mode dual --seq_factor_mode none|--qk_norm relaxed_kaczmarz_q_norm --expand_v 8 --gate_mode dual --seq_factor_mode none"
  "ab4_state|a4_expand_v16|StateExpandV16|RelaxedKaczmarzQNorm_MQAR|RelaxedKaczmarzQNorm_0.4B|--qk_norm relaxed_kaczmarz_q_norm --expand_v 16 --gate_mode dual --seq_factor_mode none|--qk_norm relaxed_kaczmarz_q_norm --expand_v 16 --gate_mode dual --seq_factor_mode none"
)

MANIFEST_PATH="${ANALYSIS_SAVE_DIR}/ablation_47_manifest.csv"
echo "group,variant,label,mqar_model,pretrain_model,mqar_args,pretrain_args" > "${MANIFEST_PATH}"
for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r group variant label mqar_model pre_model mqar_args pre_args <<< "${entry}"
  echo "${group},${variant},${label},${mqar_model},${pre_model},\"${mqar_args}\",\"${pre_args}\"" >> "${MANIFEST_PATH}"
done

echo "Starting 4.7 ablation runs..."
echo "  Profile: ${PROFILE}"
echo "  Groups: ${GROUPS_CSV}"
echo "  Variants: ${VARIANTS_CSV:-ALL}"
echo "  MQAR out root: ${MQAR_OUT_ROOT}"
echo "  Save dir: ${SAVE_DIR}"
echo "  Manifest: ${MANIFEST_PATH}"

for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r group variant label mqar_model pre_model mqar_args pre_args <<< "${entry}"

  if ! group_enabled "${group}"; then
    continue
  fi
  if ! variant_enabled "${variant}"; then
    continue
  fi

  echo "--------------------------------------------------"
  echo "Group: ${group} | Variant: ${variant} | Label: ${label}"

  mqar_exp_name="ab47_${variant}_mqar_seq${MQAR_SEQ_LEN}_key${MQAR_KEY_LEN}_seed${SEED}"
  mqar_out_dir="${MQAR_OUT_ROOT}/${mqar_exp_name}"

  rerun_mqar="${FORCE_RERUN_MQAR}"
  if [[ "${DATA_REGENERATED}" == "1" ]]; then
    rerun_mqar=1
  fi

  if [[ -f "${mqar_out_dir}/results.json" && "${rerun_mqar}" != "1" ]]; then
    echo "MQAR already done, skip: ${mqar_exp_name}"
    verify_mqar_results "${mqar_out_dir}/results.json"
  else
    if [[ -d "${mqar_out_dir}" && "${rerun_mqar}" == "1" ]]; then
      echo "Re-running MQAR due to refreshed data/force flag: ${mqar_exp_name}"
      if [[ "${DRY_RUN}" != "1" ]]; then
        rm -rf "${mqar_out_dir}"
      fi
    fi
    mkdir -p "${mqar_out_dir}"
    mqar_val_interval="${MQAR_VAL_INTERVAL}"
    if (( MQAR_MAX_STEPS < mqar_val_interval )); then
      mqar_val_interval="${MQAR_MAX_STEPS}"
    fi
    mqar_save_interval="${MQAR_SAVE_INTERVAL}"
    if (( MQAR_MAX_STEPS < mqar_save_interval )); then
      mqar_save_interval="${MQAR_MAX_STEPS}"
    fi
    mqar_cmd=(
      "${PYTHON_BIN}" "${PROJECT_ROOT}/train_mqar.py"
      --data_dir "${DATA_DIR}"
      --out_dir "${mqar_out_dir}"
      --model_name "${mqar_model}"
      --exp_name "${mqar_exp_name}"
      --seed "${SEED}"
      --batch_size "${MQAR_BATCH_SIZE}"
      --learning_rate "${MQAR_LR}"
      --max_steps "${MQAR_MAX_STEPS}"
      --val_interval "${mqar_val_interval}"
      --save_interval "${mqar_save_interval}"
      --early_stop_patience "${MQAR_EARLY_STOP}"
      --num_pairs "${MQAR_NUM_PAIRS}"
      --extrapolation_factors "1,2,4,8"
      --extrapol_base_seq_len "${MQAR_SEQ_LEN}"
      --extrapol_key_len "${MQAR_KEY_LEN}"
      --extrapol_num_val "${MQAR_NUM_VAL}"
      --wandb_dir "${MQAR_OUT_ROOT}/wandb"
    )
    if [[ -n "${mqar_args}" ]]; then
      read -r -a mqar_extra <<< "${mqar_args}"
      mqar_cmd+=("${mqar_extra[@]}")
    fi
    run_cmd "${mqar_cmd[@]}"
    verify_mqar_results "${mqar_out_dir}/results.json"
  fi

  pretrain_exp_name="ab47_${variant}_${TOKENS_TAG}"
  pretrain_run_dir="tsz512x4k_${pretrain_exp_name}"
  pretrain_ckpt_dir="${SAVE_DIR}/outputs/${pretrain_run_dir}"
  if verify_pretrain_outputs "${pretrain_exp_name}"; then
    echo "Pretrain already done, skip: ${pretrain_exp_name}"
  else
    if ! run_pretrain_with_retry "${variant}" "${pretrain_exp_name}" "${pre_model}" "${pre_args}"; then
      echo "ERROR: Pretrain failed after retries for ${pretrain_exp_name}."
      exit 1
    fi
  fi
done

if [[ "${AUTO_REPORT}" == "1" ]]; then
  echo "Generating ablation 4.7 report..."
  run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/ablation_47_analysis.py" \
    --mqar_out_root "${MQAR_OUT_ROOT}" \
    --pretrain_wandb_root "${SAVE_DIR}/wandb" \
    --save_dir "${ANALYSIS_SAVE_DIR}" \
    --manifest "${MANIFEST_PATH}"
fi

echo "All done."
echo "Report: ${ANALYSIS_SAVE_DIR}/ablation_47_report.md"
