#!/usr/bin/env bash
set -uo pipefail

mkdir -p ../logs ../data/judge_runs ../pids

echo "Launching all runs in parallel..."

# 1) reactions.json x phi4 (ollama)
nohup python generate_llm_as_a_judge_pairwise_preferences.py \
  --input ../data/reactions.json \
  --output ../data/judge_runs/reactions__phi4.jsonl \
  --provider ollama \
  --model phi4 \
  >> ../logs/reactions__phi4.log 2>&1 &
echo $! > ../pids/reactions__phi4.pid
echo "  -> reactions.json / phi4 (PID $(cat ../pids/reactions__phi4.pid))"

# 2) reactions.json x qwen3-235b (openrouter)
nohup python generate_llm_as_a_judge_pairwise_preferences.py \
  --input ../data/reactions.json \
  --output ../data/judge_runs/reactions__qwen3-235b.jsonl \
  --provider openrouter \
  --model qwen/qwen3-235b-a22b-2507 \
  >> ../logs/reactions__qwen3-235b.log 2>&1 &
echo $! > ../pids/reactions__qwen3-235b.pid
echo "  -> reactions.json / qwen3-235b (PID $(cat ../pids/reactions__qwen3-235b.pid))"

# 3) profs_phase2.json x phi4 (ollama)
nohup python generate_llm_as_a_judge_pairwise_preferences.py \
  --input ../data/profs_phase2.json \
  --output ../data/judge_runs/profs_phase2__phi4.jsonl \
  --provider ollama \
  --model phi4 \
  >> ../logs/profs_phase2__phi4.log 2>&1 &
echo $! > ../pids/profs_phase2__phi4.pid
echo "  -> profs_phase2.json / phi4 (PID $(cat ../pids/profs_phase2__phi4.pid))"

# 4) profs_phase2.json x qwen3-235b (openrouter)
nohup python generate_llm_as_a_judge_pairwise_preferences.py \
  --input ../data/profs_phase2.json \
  --output ../data/judge_runs/profs_phase2__qwen3-235b.jsonl \
  --provider openrouter \
  --model qwen/qwen3-235b-a22b-2507 \
  >> ../logs/profs_phase2__qwen3-235b.log 2>&1 &
echo $! > ../pids/profs_phase2__qwen3-235b.pid
echo "  -> profs_phase2.json / qwen3-235b (PID $(cat ../pids/profs_phase2__qwen3-235b.pid))"

echo ""
echo "Track progress:"
echo "  tail -f ../logs/*.log"
echo "  watch -n5 'wc -l ../data/judge_runs/*.jsonl'"
echo ""
echo "To wait for runs to finish before launching a new script:"
echo "  wait"