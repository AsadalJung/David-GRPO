RETRIEVER_URL=${RETRIEVER_URL:-http://localhost:8003/retrieve} # 8004 for antileakbench
export RETRIEVER_URL

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

MODEL_PATH="${MODEL_PATH:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUTPUT_PATH="${OUTPUT_PATH:-eval_result/David-GRPO-1.5b.parquet}"
TEST_DATA_PATH="${TEST_DATA_PATH:-data/processed/all_test_data/test.parquet}"
ROLL_PROMPT_LENGTH="${ROLL_PROMPT_LENGTH:-2048}"
ROLL_RESPONSE_LENGTH="${ROLL_RESPONSE_LENGTH:-12288}"
ROLL_GPU_MEMORY_UTILIZATION="${ROLL_GPU_MEMORY_UTILIZATION:-0.60}"
ROLL_ENFORCE_EAGER="${ROLL_ENFORCE_EAGER:-True}"
ROLL_ENABLE_CHUNKED_PREFILL="${ROLL_ENABLE_CHUNKED_PREFILL:-True}"
ROLL_MAX_NUM_BATCHED_TOKENS="${ROLL_MAX_NUM_BATCHED_TOKENS:-}"
ROLL_FREE_CACHE_ENGINE="${ROLL_FREE_CACHE_ENGINE:-True}"
ROLL_MAX_SEARCH_NUMS="${ROLL_MAX_SEARCH_NUMS:-}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-}"
if [[ -z "${N_GPUS_PER_NODE}" ]]; then
  IFS=',' read -r -a _gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  N_GPUS_PER_NODE="${#_gpu_ids[@]}"
fi

EXTRA_ARGS=()
if [[ -n "${ROLL_MAX_NUM_BATCHED_TOKENS}" ]]; then
  EXTRA_ARGS+=("+rollout.max_num_batched_tokens=${ROLL_MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${ROLL_MAX_SEARCH_NUMS}" ]]; then
  EXTRA_ARGS+=("+rollout.max_search_nums=${ROLL_MAX_SEARCH_NUMS}")
fi

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    data.path="${TEST_DATA_PATH}"\
    data.prompt_key=prompt \
    data.batch_size="${BATCH_SIZE}" \
    data.n_samples=1 \
    data.output_path="${OUTPUT_PATH}" \
    model.path="${MODEL_PATH}" \
    +model.trust_remote_code=True \
    +rollout.model=search \
    rollout.temperature=0.60 \
    rollout.top_p=1 \
    rollout.top_k=-1 \
    rollout.prompt_length="${ROLL_PROMPT_LENGTH}" \
    rollout.response_length="${ROLL_RESPONSE_LENGTH}" \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization="${ROLL_GPU_MEMORY_UTILIZATION}" \
    rollout.enforce_eager="${ROLL_ENFORCE_EAGER}" \
    rollout.free_cache_engine="${ROLL_FREE_CACHE_ENGINE}" \
    +rollout.disable_log_stats=False \
    +rollout.enable_chunked_prefill="${ROLL_ENABLE_CHUNKED_PREFILL}" \
    +rollout.n=1 \
    "${EXTRA_ARGS[@]}"
