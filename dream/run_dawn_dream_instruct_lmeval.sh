#!/usr/bin/env bash
set -euo pipefail

# DAWN Dream wrapper + lm-eval harness
# Default: full eval (limit=9999)
# Smoke test example:
#   GPU_ID=1 LIMIT=1 bash run_dawn_dream_instruct_lmeval.sh
# DAWN conf sweep example:
#   ALG=dawn DAWN_CONF_LIST="0.95 0.85 0.80 0.75 0.70" GPU_ID=1 LIMIT=1 bash run_dawn_dream_instruct_lmeval.sh

GPU_ID="${GPU_ID:-0}"
LIMIT="${LIMIT:-9999}"
BASE_PORT="${BASE_PORT:-12440}"
ALG="${ALG:-entropy}"
TASKS="${TASKS:-gsm8k_cot humaneval_instruct mbpp_instruct ifeval}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"

MODEL_ID="${MODEL_ID:-Dream-org/Dream-v0-Instruct-7B}"
OUT_ROOT="${OUT_ROOT:-output_dawn_lmeval_limit${LIMIT}}"
ACCELERATE_BIN="${ACCELERATE_BIN:-}"

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

# Ensure custom instruct tasks (gsm8k_cot / humaneval_instruct / mbpp_instruct) are available.
export LM_EVAL_INCLUDE_PATH="${LM_EVAL_INCLUDE_PATH:-/workspace/Dream/eval_instruct/lm_eval/tasks}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=.

normalize_conf_threshold () {
  local raw="$1"
  if [[ "${raw}" =~ ^[0-9]+$ ]]; then
    awk -v v="${raw}" 'BEGIN { if (v > 1) printf "%.2f", v / 100; else printf "%s", v }'
    return 0
  fi
  if [[ "${raw}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    awk -v v="${raw}" 'BEGIN { if (v > 1) printf "%.2f", v / 100; else printf "%.2f", v }'
    return 0
  fi
  echo "${raw}"
}

run_one () {
  local task="$1"
  local max_new_tokens="$2"
  local port="$3"
  local conf_raw="${4:-}"
  local alg_extra_args=""
  local run_prefix="${OUT_ROOT}"
  local block_length_arg=",block_length=${max_new_tokens}"

  if [[ "${ALG}" == "dawn" ]]; then
    local conf_threshold="${conf_raw:-${DAWN_CONF_THRESHOLD:-0.8}}"
    local conf_threshold_norm
    conf_threshold_norm="$(normalize_conf_threshold "${conf_threshold}")"
    local conf_tag="${conf_threshold_norm//./p}"
    local dawn_tau_induce="${DAWN_TAU_INDUCE:-0.75}"
    local dawn_tau_sink="${DAWN_TAU_SINK:-0.03}"
    local dawn_tau_edge="${DAWN_TAU_EDGE:-0.10}"
    local dawn_high_conf_threshold="${DAWN_HIGH_CONF_THRESHOLD:-0.90}"
    alg_extra_args=",conf_threshold=${conf_threshold_norm},high_conf_threshold=${dawn_high_conf_threshold},tau_induce=${dawn_tau_induce},tau_sink=${dawn_tau_sink},tau_edge=${dawn_tau_edge}"
    run_prefix="${OUT_ROOT}/conf_${conf_tag}"
  fi

  local run_dir="${run_prefix}/${task}/step_${max_new_tokens}"
  local speed_jsonl="${run_dir}/speed/nfe_stats.jsonl"
  local fp_stats_json="${run_dir}/step_stats/fp_stats.json"

  mkdir -p "${run_dir}" "$(dirname "${speed_jsonl}")" "$(dirname "${fp_stats_json}")"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${ACCELERATE_BIN}" launch --main_process_port "${port}" eval.py \
    --model dream \
    --model_args pretrained=${MODEL_ID},trust_remote_code=True,dtype=bfloat16,max_new_tokens=${max_new_tokens},diffusion_steps=${max_new_tokens}${block_length_arg},temperature=${TEMPERATURE},top_p=${TOP_P},alg=${ALG}${alg_extra_args},show_speed=True,outp_path=${speed_jsonl},fp_stats_path=${fp_stats_json} \
    --tasks "${task}" \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --limit "${LIMIT}" \
    --output_path "${run_dir}" \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template
}

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

idx=0
if [[ "${ALG}" == "dawn" && -n "${DAWN_CONF_LIST:-}" ]]; then
  DAWN_CONF_LIST="${DAWN_CONF_LIST//,/ }"
  for conf in ${DAWN_CONF_LIST}; do
    conf_norm="$(normalize_conf_threshold "${conf}")"
    for task in ${TASKS}; do
      max_new_tokens="$(task_max_new_tokens "${task}")"
      run_one "${task}" "${max_new_tokens}" $((BASE_PORT + idx)) "${conf_norm}"
      idx=$((idx + 1))
    done
  done
else
  for task in ${TASKS}; do
    max_new_tokens="$(task_max_new_tokens "${task}")"
    run_one "${task}" "${max_new_tokens}" $((BASE_PORT + idx))
    idx=$((idx + 1))
  done
fi

echo "Completed. Outputs under: ${OUT_ROOT}"
