#!/usr/bin/env bash
set -euo pipefail

# LLaDA Instruct DAWN 4-benchmark 5-point sweep
# Example:
#   GPU_ID=1 LIMIT=1 bash run_dawn_llada_instruct_4bench_5point_sweep.sh

GPU_ID="${GPU_ID:-0}"
LIMIT="${LIMIT:-9999}"
BASE_PORT="${BASE_PORT:-12650}"
MODEL_ID="${MODEL_ID:-GSAI-ML/LLaDA-8B-Instruct}"
TASKS="${TASKS:-gsm8k_cot humaneval_instruct mbpp_instruct ifeval}"
ACCELERATE_BIN="${ACCELERATE_BIN:-}"

# Conservative -> Aggressive
POINT_LABELS=(
  "p1_conservative"
  "p2_safe"
  "p3_balanced"
  "p4_fast"
  "p5_aggressive"
)
# For LLaDA DAWN path, tau_low is the effective confidence gate (recalibrated).
TAU_LOW_LEVELS=(0.97 0.95 0.93 0.90 0.86)
# High-confidence anchor threshold sweep (conservative -> aggressive).
HIGH_CONF_LEVELS=(0.995 0.99 0.97 0.95 0.92)

OUT_ROOT="${OUT_ROOT:-output_dawn_llada_instruct_4bench_5point_limit${LIMIT}}"

if [[ -z "${ACCELERATE_BIN}" ]]; then
  if [[ -x /workspace/venvs/dawnvenv/bin/accelerate ]]; then
    ACCELERATE_BIN="/workspace/venvs/dawnvenv/bin/accelerate"
  elif command -v accelerate >/dev/null 2>&1; then
    ACCELERATE_BIN="$(command -v accelerate)"
  elif [[ -x /workspace/venvs/lladavenv/bin/accelerate ]]; then
    ACCELERATE_BIN="/workspace/venvs/lladavenv/bin/accelerate"
  elif [[ -x /workspace/venvs/real_lladavenv/bin/accelerate ]]; then
    ACCELERATE_BIN="/workspace/venvs/real_lladavenv/bin/accelerate"
  else
    echo "Could not find accelerate binary. Set ACCELERATE_BIN=/path/to/accelerate" >&2
    exit 127
  fi
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=.
export LM_EVAL_INCLUDE_PATH="${LM_EVAL_INCLUDE_PATH:-/workspace/dawn_experiment/data/tasks}"

task_gen_length () {
  case "$1" in
    gsm8k_cot) echo 256 ;;
    humaneval_instruct) echo 512 ;;
    mbpp_instruct) echo 768 ;;
    ifeval) echo 768 ;;
    *)
      echo "Unknown task: $1" >&2
      return 2
      ;;
  esac
}

run_one () {
  local task="$1"
  local gen_length="$2"
  local point_label="$3"
  local tau_low="$4"
  local high_conf="$5"
  local port="$6"

  local run_dir="${OUT_ROOT}/${point_label}/${task}/step_${gen_length}"
  local speed_jsonl="${run_dir}/speed/nfe_stats.jsonl"
  mkdir -p "${run_dir}" "$(dirname "${speed_jsonl}")"

  echo "=================================================="
  echo "[LLaDA][${point_label}] task=${task} gen=${gen_length} tau_low=${tau_low} high_conf=${high_conf} limit=${LIMIT} gpu=${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${ACCELERATE_BIN}" launch --main_process_port "${port}" eval_llada.py \
    --tasks "${task}" \
    --include_path "${LM_EVAL_INCLUDE_PATH}" \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --model llada_dist \
    --model_args model_path=${MODEL_ID},gen_length=${gen_length},steps=${gen_length},block_length=${gen_length},temperature=0.1,show_speed=True,dawn=True,tau_sink=0.01,tau_edge=0.07,tau_induce=0.70,tau_low=${tau_low},high_conf_threshold=${high_conf},outp_path=${speed_jsonl} \
    --limit "${LIMIT}" \
    --include_path /workspace/dawn_experiment/data/tasks \
    --output_path "${run_dir}" \
    --log_samples
}

idx=0
for i in "${!POINT_LABELS[@]}"; do
  point_label="${POINT_LABELS[$i]}"
  tau_low="${TAU_LOW_LEVELS[$i]}"
  high_conf="${HIGH_CONF_LEVELS[$i]}"
  for task in ${TASKS}; do
    gen_length="$(task_gen_length "${task}")"
    run_one "${task}" "${gen_length}" "${point_label}" "${tau_low}" "${high_conf}" $((BASE_PORT + idx))
    idx=$((idx + 1))
  done
done

echo "Completed LLaDA DAWN 5-point sweep. Outputs: ${OUT_ROOT}"
