#!/bin/bash
# Start the Hint B_CoT judge (Qwen2.5-32B-Instruct) as a vLLM OpenAI-compatible
# server on ONE GPU and wait until it is healthy. Prints the URL; exports nothing.
# Usage: source scripts/judge_server.sh <gpu_index> [port]   (sets $HINT_JUDGE_URL, $JUDGE_PID)
GPU=${1:?gpu index}; PORT=${2:-8765}
MODEL=${SCRATCH}/transformers/Qwen2.5-32B-Instruct
mkdir -p ${LOG_DIR:-${SCRATCH}/logs}; LOG=${LOG_DIR:-${SCRATCH}/logs}/judge_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.log
CUDA_VISIBLE_DEVICES=$GPU nohup vllm serve "$MODEL" --served-model-name Qwen2.5-32B-Instruct \
    --port "$PORT" --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 > "$LOG" 2>&1 &
export JUDGE_PID=$!
export HINT_JUDGE_URL="http://127.0.0.1:${PORT}"
export HINT_JUDGE_MODEL="Qwen2.5-32B-Instruct"
for i in $(seq 1 180); do
  if curl -sf "${HINT_JUDGE_URL}/health" > /dev/null 2>&1; then echo "judge ready at $HINT_JUDGE_URL (pid $JUDGE_PID) after ${i}0s"; break; fi
  if ! kill -0 $JUDGE_PID 2>/dev/null; then echo "judge server died, see $LOG"; tail -20 "$LOG"; exit 1; fi
  sleep 10
done
curl -sf "${HINT_JUDGE_URL}/health" > /dev/null || { echo "judge not ready after 30 min"; exit 1; }
