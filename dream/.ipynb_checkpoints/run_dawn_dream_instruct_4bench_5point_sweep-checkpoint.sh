#!/usr/bin/env bash
set -euo pipefail

# Dream Instruct DAWN 4-benchmark 5-point sweep
# Example:
#   GPU_ID=1 LIMIT=9999 bash run_dawn_dream_instruct_4bench_5point_sweep.sh

GPU_ID="${GPU_ID:-0}"
LIMIT="${LIMIT:-9999}"
BASE_PORT="${BASE_PORT:-12550}"
MODEL_ID="${MODEL_ID:-Dream-org/Dream-v0-Instruct-7B}"
TASKS="${TASKS:-gsm8k_cot humaneval_instruct mbpp_instruct ifeval}"
ACCELERATE_BIN="${ACCELERATE_BIN:-}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
export LM_EVAL_INCLUDE_PATH="${LM_EVAL_INCLUDE_PATH:-/workspace/dawn_experiment/data/tasks}"

# Conservative -> Aggressive
POINT_LABELS=(
  "p1_conservative"
  "p2_safe"
  "p3_balanced"
  "p4_fast"
  "p5_aggressive"
)
# DAWN conservative -> aggressive settings (recalibrated for temperature=0.0).
# Keep block_length == generation length (no block-diffusion sweep).
CONF_LEVELS=(0.99 0.98 0.93 0.85 0.82)
HIGH_CONF_LEVELS=(0.99 0.96 0.85 0.75 0.70)
TAU_INDUCE_LEVELS=(0.97 0.90 0.70 0.50 0.45)
TAU_EDGE_LEVELS=(0.02 0.05 0.12 0.25 0.30)
TAU_SINK="${TAU_SINK:-0.03}"

OUT_ROOT="${OUT_ROOT:-output_dawn_dream_instruct_4bench_5point_limit${LIMIT}}"

if [[ -z "${ACCELERATE_BIN}" ]]; then
  if [[ -x /workspace/venvs/dawnvenv/bin/accelerate ]]; then
    ACCELERATE_BIN="/workspace/venvs/dawnvenv/bin/accelerate"
  elif command -v accelerate >/dev/null 2>&1; then
    ACCELERATE_BIN="$(command -v accelerate)"
  elif [[ -x /workspace/venvs/real_dreamvenv/bin/accelerate ]]; then
    ACCELERATE_BIN="/workspace/venvs/real_dreamvenv/bin/accelerate"
  elif [[ -x /workspace/venvs/klassvenv/bin/accelerate ]]; then
    ACCELERATE_BIN="/workspace/venvs/klassvenv/bin/accelerate"
  else
    echo "Could not find accelerate binary. Set ACCELERATE_BIN=/path/to/accelerate" >&2
    exit 127
  fi
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=.
#export LM_EVAL_INCLUDE_PATH="${LM_EVAL_INCLUDE_PATH:-/workspace/Dream/eval_instruct/lm_eval/tasks}"

task_max_new_tokens () {
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

is_done () {
  local run_dir="$1"
  local nfe_file="${run_dir}/speed/nfe_stats.jsonl"
  local result_file
  result_file="$(find "${run_dir}" -type f -name 'results_*.json' -print -quit 2>/dev/null || true)"
  [[ -n "${result_file}" && -s "${nfe_file}" ]]
}

run_one () {
  local task="$1"
  local max_new_tokens="$2"
  local point_label="$3"
  local conf_threshold="$4"
  local high_conf_threshold="$5"
  local tau_induce="$6"
  local tau_edge="$7"
  local port="$8"

  local run_dir="${OUT_ROOT}/${point_label}/${task}/step_${max_new_tokens}"
  local speed_jsonl="${run_dir}/speed/nfe_stats.jsonl"
  local fp_stats_json="${run_dir}/step_stats/fp_stats.json"

  if is_done "${run_dir}"; then
    echo "[SKIP][Dream][${point_label}] task=${task} step=${max_new_tokens} already completed: ${run_dir}"
    return 0
  fi

  mkdir -p "${run_dir}" "$(dirname "${speed_jsonl}")" "$(dirname "${fp_stats_json}")"

  echo "=================================================="
  echo "[Dream][${point_label}] task=${task} gen=${max_new_tokens} conf=${conf_threshold} high_conf=${high_conf_threshold} tau_induce=${tau_induce} tau_edge=${tau_edge} block_length=${max_new_tokens} limit=${LIMIT} gpu=${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${ACCELERATE_BIN}" launch --main_process_port "${port}" eval.py \
    --model dream \
    --model_args pretrained=${MODEL_ID},trust_remote_code=True,dtype=bfloat16,max_new_tokens=${max_new_tokens},diffusion_steps=${max_new_tokens},block_length=${max_new_tokens},temperature=${TEMPERATURE},top_p=${TOP_P},alg=dawn,conf_threshold=${conf_threshold},high_conf_threshold=${high_conf_threshold},tau_induce=${tau_induce},tau_sink=${TAU_SINK},tau_edge=${tau_edge},show_speed=True,outp_path=${speed_jsonl},fp_stats_path=${fp_stats_json} \
    --tasks "${task}" \
    --include_path "${LM_EVAL_INCLUDE_PATH}" \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --limit "${LIMIT}" \
    --output_path "${run_dir}" \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template
}

idx=0
for i in "${!POINT_LABELS[@]}"; do
  point_label="${POINT_LABELS[$i]}"
  conf="${CONF_LEVELS[$i]}"
  high_conf="${HIGH_CONF_LEVELS[$i]}"
  tau_induce="${TAU_INDUCE_LEVELS[$i]}"
  tau_edge="${TAU_EDGE_LEVELS[$i]}"
  for task in ${TASKS}; do
    max_new_tokens="$(task_max_new_tokens "${task}")"
    run_one "${task}" "${max_new_tokens}" "${point_label}" "${conf}" "${high_conf}" "${tau_induce}" "${tau_edge}" $((BASE_PORT + idx))
    idx=$((idx + 1))
  done
done

echo "Completed Dream DAWN 5-point sweep. Outputs: ${OUT_ROOT}"
