#!/usr/bin/env bash
set -euo pipefail

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

model="Dream-org/Dream-v0-Instruct-7B"
model_name="Dream-v0-Instruct-7B"
device=0
include_path="/workspace/dawn_experiment/data/tasks"

# DAWN confidence thresholds (task-specific)
gsm8k_conf_threshold=0.80
humaneval_conf_threshold=0.80
mbpp_conf_threshold=0.80
ifeval_conf_threshold="${mbpp_conf_threshold}"  # keep same as mbpp

run_dawn () {
  local task="$1"
  local length="$2"
  local block_length="$3"
  local num_fewshot="$4"
  local conf_threshold="$5"
  local log_flag="${6:-}"

  local run_tag="${task}-ns${num_fewshot}-${length}"
  local out_dir="evals_results_${model_name}/dawn/${run_tag}"
  local out_jsonl="${out_dir}/results.jsonl"

  CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=dawn,show_speed=True,conf_threshold=${conf_threshold},tau_induce=0.75,tau_sink=0.03,tau_edge=0.10,outp_path=${out_jsonl} \
    --tasks ${task} \
    --include_path ${include_path} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path ${out_dir} ${log_flag}
}

############################################### gsm8k evaluations ###############################################
run_dawn gsm8k 256 256 0 "${gsm8k_conf_threshold}"

############################################### humaneval_instruct evaluations ###############################################
run_dawn humaneval_instruct 512 512 0 "${humaneval_conf_threshold}" "--log_samples"

############################################### mbpp_instruct evaluations ###############################################
run_dawn mbpp_instruct 768 768 0 "${mbpp_conf_threshold}" "--log_samples"

############################################### ifeval evaluations ###############################################
run_dawn ifeval 768 768 0 "${ifeval_conf_threshold}" "--log_samples"
