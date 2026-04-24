#!/bin/bash
# Sequential DeepSeek-V4-Pro long-context evaluation for selected RULER tasks.

set -euo pipefail

cd "$(dirname "$0")"

ENV_FILE="${ENV_FILE:-../.env}"
if [ -f "${ENV_FILE}" ]; then
    set -a
    source "${ENV_FILE}"
    set +a
fi

MODEL_NAME="deepseek-v4-pro"
BENCHMARK="synthetic"
ROOT_DIR="${ROOT_DIR:-benchmark_root/deepseek-v4-pro-long-context}"
MODEL_DIR="${MODEL_DIR:-../..}"
ENGINE_DIR="${ENGINE_DIR:-.}"
BATCH_SIZE="${BATCH_SIZE:-1}"

export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
export AZURE_API_ID="${AZURE_API_ID:-}"
export AZURE_API_SECRET="${AZURE_API_SECRET:-}"
export AZURE_API_ENDPOINT="${AZURE_API_ENDPOINT:-}"

source config_models.sh
MODEL_CONFIG=$(MODEL_SELECT "${MODEL_NAME}" "${MODEL_DIR}" "${ENGINE_DIR}")
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"

NUM_SAMPLES="${NUM_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-32}"
RULER_SEQ_LENGTHS="${RULER_SEQ_LENGTHS:-4096,8192,16384,32768,65536,131072,262144}"
RULER_TASKS="${RULER_TASKS:-niah_single_1}"
IFS="," read -r -a SEQ_LENGTHS <<< "${RULER_SEQ_LENGTHS}"
IFS="," read -r -a TASKS <<< "${RULER_TASKS}"
TASKS_ARG="${RULER_TASKS}"

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "DEEPSEEK_API_KEY or OPENAI_API_KEY must be set in the environment."
    exit 1
fi

if [ "${MODEL_FRAMEWORK}" != "openai" ]; then
    echo "Expected ${MODEL_NAME} to resolve to openai framework, got: ${MODEL_FRAMEWORK}"
    exit 1
fi

export OPENAI_API_KEY
export GEMINI_API_KEY
export AZURE_API_ID="${AZURE_ID}"
export AZURE_API_SECRET="${AZURE_SECRET}"
export AZURE_API_ENDPOINT="${AZURE_ENDPOINT}"

if [[ ",${RULER_TASKS}," == *",qa_1,"* ]] && [ ! -f "data/synthetic/json/squad.json" ]; then
    echo "Downloading SQuAD dev set for qa_1..."
    curl -L "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json" -o "data/synthetic/json/squad.json"
fi

echo "DeepSeek RULER evaluation"
echo "  model       : ${MODEL_PATH}"
echo "  base_url    : ${DEEPSEEK_BASE_URL}"
echo "  tasks       : ${RULER_TASKS}"
echo "  lengths     : ${RULER_SEQ_LENGTHS}"
echo "  num_samples : ${NUM_SAMPLES}"
echo "  output      : ${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}"
echo

total_time=0
total_lengths=${#SEQ_LENGTHS[@]}
total_tasks=${#TASKS[@]}
for length_idx in "${!SEQ_LENGTHS[@]}"; do
    MAX_SEQ_LENGTH="${SEQ_LENGTHS[$length_idx]}"
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p "${DATA_DIR}" "${PRED_DIR}"

    echo "============================================================"
    echo "Length $((length_idx + 1))/${total_lengths}: ${MAX_SEQ_LENGTH}"
    echo "============================================================"

    for task_idx in "${!TASKS[@]}"; do
        TASK="${TASKS[$task_idx]}"
        echo
        echo "---- Task $((task_idx + 1))/${total_tasks}: ${TASK} @ ${MAX_SEQ_LENGTH} ----"
        echo "[prepare] ${TASK}"
        python data/prepare.py \
            --save_dir "${DATA_DIR}" \
            --benchmark "${BENCHMARK}" \
            --task "${TASK}" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --tokenizer_type "${TOKENIZER_TYPE}" \
            --max_seq_length "${MAX_SEQ_LENGTH}" \
            --model_template_type "${MODEL_TEMPLATE_TYPE}" \
            --num_samples "${NUM_SAMPLES}"

        echo "[predict] ${TASK}"
        start_time=$(date +%s)
        python pred/call_api.py \
            --data_dir "${DATA_DIR}" \
            --save_dir "${PRED_DIR}" \
            --benchmark "${BENCHMARK}" \
            --task "${TASK}" \
            --server_type "${MODEL_FRAMEWORK}" \
            --model_name_or_path "${MODEL_PATH}" \
            --temperature "${TEMPERATURE}" \
            --top_k "${TOP_K}" \
            --top_p "${TOP_P}" \
            --batch_size "${BATCH_SIZE}" \
            --threads 1
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
        echo "[predict] ${TASK} completed in ${time_diff}s"
    done

    echo
    echo "[evaluate] ${MAX_SEQ_LENGTH}"
    python eval/evaluate.py \
        --data_dir "${PRED_DIR}" \
        --benchmark "${BENCHMARK}" \
        --tasks "${TASKS_ARG}"
done

echo "Total time spent on call_api: ${total_time} seconds"
echo "Results saved under ${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}"
