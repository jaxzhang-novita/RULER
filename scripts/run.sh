#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh MODEL_NAME BENCHMARK_NAME

ENV_FILE="${ENV_FILE:-../.env}"
if [ -f "${ENV_FILE}" ]; then
    set -a
    source "${ENV_FILE}"
    set +a
fi

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> $1 <benchmark_name>"
    exit 1
fi


# Root Directories
GPUS="${GPUS:-1}" # GPU size for tensor_parallel.
ROOT_DIR="${ROOT_DIR:-benchmark_root}" # the path that stores generated task samples and model predictions.
MODEL_DIR="${MODEL_DIR:-../..}" # the path that contains individual model folders from HUggingface.
ENGINE_DIR="${ENGINE_DIR:-.}" # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE="${BATCH_SIZE:-1}"  # increase to improve GPU utilization


# Model and Tokenizer
source config_models.sh
MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi


export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}


# Benchmark and Tasks
source config_tasks.sh
BENCHMARK=${2}
if ! declare -p "${BENCHMARK}" >/dev/null 2>&1; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi
eval "TASKS=(\"\${${BENCHMARK}[@]}\")"
if [ -n "${RULER_TASKS:-}" ]; then
    IFS="," read -r -a TASKS <<< "${RULER_TASKS}"
fi
if [ -n "${RULER_SEQ_LENGTHS:-}" ]; then
    IFS="," read -r -a SEQ_LENGTHS <<< "${RULER_SEQ_LENGTHS}"
fi
INPUT_PRICE_PER_M="${INPUT_PRICE_PER_M:-12}"
OUTPUT_PRICE_PER_M="${OUTPUT_PRICE_PER_M:-24}"
COST_CURRENCY="${COST_CURRENCY:-CNY}"

echo "RULER evaluation"
echo "  model       : ${MODEL_NAME} (${MODEL_PATH})"
echo "  framework   : ${MODEL_FRAMEWORK}"
echo "  benchmark   : ${BENCHMARK}"
echo "  tasks       : ${TASKS[*]}"
echo "  lengths     : ${SEQ_LENGTHS[*]}"
echo "  num_samples : ${NUM_SAMPLES}"
echo "  pricing     : ${INPUT_PRICE_PER_M}/${OUTPUT_PRICE_PER_M} ${COST_CURRENCY} per 1M input/output tokens"
echo "  output      : ${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}"
echo


# Start server (you may want to run in other container.)
if [ "$MODEL_FRAMEWORK" == "vllm" ]; then
    python pred/serve_vllm.py \
        --model=${MODEL_PATH} \
        --tensor-parallel-size=${GPUS} \
        --dtype bfloat16 \
        --disable-custom-all-reduce \
        &

elif [ "$MODEL_FRAMEWORK" == "trtllm" ]; then
    python pred/serve_trt.py \
        --model_path=${MODEL_PATH} \
        &

elif [ "$MODEL_FRAMEWORK" == "sglang" ]; then
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --tp ${GPUS} \
        --port 5000 \
        --enable-flashinfer \
        &
    # use sglang/test/killall_sglang.sh to kill sglang server if it hangs

fi


# Start client (prepare data / call model API / obtain final metrics)
total_time=0
run_start_time=$(date +%s)
total_lengths=${#SEQ_LENGTHS[@]}
total_tasks=${#TASKS[@]}
for length_idx in "${!SEQ_LENGTHS[@]}"; do
    MAX_SEQ_LENGTH="${SEQ_LENGTHS[$length_idx]}"
    length_start_time=$(date +%s)
    
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}

    echo "============================================================"
    echo "Length $((length_idx + 1))/${total_lengths}: ${MAX_SEQ_LENGTH}"
    echo "============================================================"

    for task_idx in "${!TASKS[@]}"; do
        TASK="${TASKS[$task_idx]}"
        echo
        echo "---- Task $((task_idx + 1))/${total_tasks}: ${TASK} @ ${MAX_SEQ_LENGTH} ----"
        echo "[prepare] ${TASK}"
        python data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}
        
        echo "[predict] ${TASK}"
        start_time=$(date +%s)
        python pred/call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --model_name_or_path ${MODEL_PATH} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            --batch_size ${BATCH_SIZE} \
            ${STOP_WORDS}
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
        echo "[predict] ${TASK} completed in ${time_diff}s"
    done

    echo
    echo "[evaluate] ${MAX_SEQ_LENGTH}"
    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK} \
        ${RULER_TASKS:+--tasks ${RULER_TASKS}}

    length_end_time=$(date +%s)
    length_elapsed=$((length_end_time - length_start_time))
    echo
    echo "[cost/throughput] ${MAX_SEQ_LENGTH}"
    python summarize_cost.py \
        --pred_dir ${PRED_DIR} \
        --input_price ${INPUT_PRICE_PER_M} \
        --output_price ${OUTPUT_PRICE_PER_M} \
        --currency ${COST_CURRENCY} \
        --elapsed_seconds ${length_elapsed}
done

run_end_time=$(date +%s)
run_elapsed=$((run_end_time - run_start_time))
echo "Total time spent on call_api: $total_time seconds"
echo "Total wall time: ${run_elapsed} seconds"

echo
echo "[cost/throughput] total"
python summarize_cost.py \
    --pred_dir "${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}" \
    --recursive \
    --input_price ${INPUT_PRICE_PER_M} \
    --output_price ${OUTPUT_PRICE_PER_M} \
    --currency ${COST_CURRENCY} \
    --elapsed_seconds ${run_elapsed}
