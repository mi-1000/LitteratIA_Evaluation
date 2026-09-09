#!/usr/bin/env bash
set -uo pipefail

# Folders to store logs, runs, and pids
DIR_LOGS="../logs"
DIR_RUNS="../data/judge_runs"
DIR_PIDS="../pids"

mkdir -p "$DIR_LOGS" "$DIR_RUNS" "$DIR_PIDS"

# Datasets
LEARNER_DATASET="reactions"
TEACHER_DATASET="profs_phase2"

# Input files
INPUT_LEARNERS="../data/{$LEARNER_DATASET}.json"
INPUT_TEACHERS="../data/{$TEACHER_DATASET}.json"

echo "Launching all runs in parallel..."

# Function to launch a job in the background

launch_job() {
    local input_path=$1
    local dataset_name=$2
    local provider=$3
    local model_exact=$4
    local model_alias=$5

    local output_path="${DIR_RUNS}/${dataset_name}__${model_alias}.jsonl"
    local log_path="${DIR_LOGS}/${dataset_name}__${model_alias}.log"
    local pid_path="${DIR_PIDS}/${dataset_name}__${model_alias}.pid"

    nohup python generate_llm_as_a_judge_pairwise_preferences.py \
      --input "$input_path" \
      --output "$output_path" \
      --provider "$provider" \
      --model "$model_exact" \
      >> "$log_path" 2>&1 &
    
    echo $! > "$pid_path"
    echo "  -> $(basename "$input_path") / $model_alias (PID $(cat "$pid_path"))"
}

# Launch jobs

# learner data x phi4
launch_job "$INPUT_LEARNERS" "$LEARNER_DATASET" "ollama" "phi4" "phi4"

# learner data x qwen3-235b
launch_job "$INPUT_LEARNERS" "$LEARNER_DATASET" "openrouter" "qwen/qwen3-235b-a22b-2507" "qwen3-235b"

# teacher data x phi4
launch_job "$INPUT_TEACHERS" "$TEACHER_DATASET" "ollama" "phi4" "phi4"

# teacher data x qwen3-235b
launch_job "$INPUT_TEACHERS" "$TEACHER_DATASET" "openrouter" "qwen/qwen3-235b-a22b-2507" "qwen3-235b"

# Track progress
echo ""
echo "Track progress:"
echo "  tail -f ${DIR_LOGS}/*.log"
echo "  watch -n5 'wc -l ${DIR_RUNS}/*.jsonl'"
echo ""
echo "To wait for runs to finish before launching a new script:"
echo "  wait"