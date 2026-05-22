#!/usr/bin/env bash
set -euo pipefail

set +u
source ~/.bashrc
set -u

PYTHON_BIN="/home/yanai-lab/xiong-p/qwen/.venv/bin/python"
PIPELINE_PY="/home/yanai-lab/xiong-p/qwen/subject_dataset_pipeline.py"
DEFAULT_DATASET_PATH="/home/yanai-lab/xiong-p/qwen/subject_driven_datasets/dataset_qwen_pe_fixed_subject_driven_no_remove.json"
DEFAULT_OUTPUT_ROOT="/home/yanai-lab/xiong-p/qwen/subject_dataset_runs"
DEFAULT_VLM_MODEL="Qwen/Qwen3.6-27B-FP8"
DEFAULT_VLM_PORT="30000"
DEFAULT_VLM_GPUS="0,1"
DEFAULT_GEN_GPUS="2,3,4,5,6,7"
DEFAULT_STEPS="50"
DEFAULT_SEED_BASE="20260523"
DEFAULT_IMAGE_BASE="/home/yanai-lab/xiong-p/qwen/picobanana/openimages"
DEFAULT_PROCESSING_LOG="/home/yanai-lab/xiong-p/qwen/picobanana/openimages/pico_sam_output_ALL_20251206_032609/processing_log.json"

DATASET_PATH="${DEFAULT_DATASET_PATH}"
OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}"
RUN_NAME=""
VLM_MODEL="${DEFAULT_VLM_MODEL}"
VLM_PORT="${DEFAULT_VLM_PORT}"
VLM_GPUS="${DEFAULT_VLM_GPUS}"
GEN_GPUS="${DEFAULT_GEN_GPUS}"
STEPS="${DEFAULT_STEPS}"
SEED_BASE="${DEFAULT_SEED_BASE}"
SUBJECT_LIMIT=""
SUBJECT_OFFSET="0"
PROMPT_LIMIT=""
IMAGE_BASE="${DEFAULT_IMAGE_BASE}"
PROCESSING_LOG_PATH="${DEFAULT_PROCESSING_LOG}"
RESUME=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash /home/yanai-lab/xiong-p/qwen/run_subject_dataset_pipeline.sh [options]

Options:
  --dataset-path PATH
  --output-root PATH
  --run-name NAME
  --vlm-gpus IDS             default: 0,1
  --gen-gpus IDS             default: 2,3,4,5,6,7
  --vlm-port PORT            default: 30000
  --steps N                  default: 50
  --subject-limit N
  --subject-offset N         default: 0
  --seed-base N              default: 20260523
  --prompt-limit N
  --resume
  --dry-run
  --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --vlm-gpus)
      VLM_GPUS="$2"
      shift 2
      ;;
    --gen-gpus)
      GEN_GPUS="$2"
      shift 2
      ;;
    --vlm-port)
      VLM_PORT="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --subject-limit)
      SUBJECT_LIMIT="$2"
      shift 2
      ;;
    --subject-offset)
      SUBJECT_OFFSET="$2"
      shift 2
      ;;
    --seed-base)
      SEED_BASE="$2"
      shift 2
      ;;
    --prompt-limit)
      PROMPT_LIMIT="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_NAME}" ]]; then
  if [[ "${RESUME}" -eq 1 ]]; then
    echo "--resume requires --run-name so the script can reuse the same run directory." >&2
    exit 1
  fi
  RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
fi

RUN_DIR="${OUTPUT_ROOT%/}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

VLM_URL="http://127.0.0.1:${VLM_PORT}/v1/models"
STARTED_VLM=0
VLM_PID=""

cleanup() {
  if [[ "${STARTED_VLM}" -eq 1 && -n "${VLM_PID}" ]]; then
    kill "${VLM_PID}" >/dev/null 2>&1 || true
    wait "${VLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

check_vlm() {
  local body
  body=$(curl -fsS "${VLM_URL}" 2>/dev/null || true)
  [[ -n "${body}" ]] && grep -q "${VLM_MODEL}" <<<"${body}"
}

port_has_other_service() {
  local body
  body=$(curl -fsS "${VLM_URL}" 2>/dev/null || true)
  [[ -n "${body}" ]] && ! grep -q "${VLM_MODEL}" <<<"${body}"
}

start_vlm() {
  local log_file="${LOG_DIR}/vllm_server.log"
  echo "Starting vLLM on GPUs ${VLM_GPUS}, port ${VLM_PORT}..."
  CUDA_VISIBLE_DEVICES="${VLM_GPUS}" nohup \
    uv run --python "${PYTHON_BIN}" python -m vllm.entrypoints.openai.api_server \
      --model "${VLM_MODEL}" \
      --host 0.0.0.0 \
      --port "${VLM_PORT}" \
      --trust-remote-code \
      --tensor-parallel-size 2 \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --mm-encoder-tp-mode data \
      >"${log_file}" 2>&1 &
  VLM_PID="$!"
  STARTED_VLM=1
}

wait_for_vlm() {
  local timeout_seconds=900
  local start_ts
  start_ts=$(date +%s)
  until check_vlm; do
    if [[ "${STARTED_VLM}" -eq 1 && -n "${VLM_PID}" ]] && ! kill -0 "${VLM_PID}" >/dev/null 2>&1; then
      echo "vLLM process exited before becoming ready. Check ${LOG_DIR}/vllm_server.log" >&2
      return 1
    fi
    if (( $(date +%s) - start_ts > timeout_seconds )); then
      echo "Timed out waiting for vLLM at ${VLM_URL}" >&2
      return 1
    fi
    sleep 5
  done
}

if check_vlm; then
  echo "Reusing existing vLLM at ${VLM_URL}"
elif port_has_other_service; then
  echo "Port ${VLM_PORT} is serving /v1/models, but not for ${VLM_MODEL}. Refusing to start a second server on the same port." >&2
  exit 1
else
  start_vlm
  wait_for_vlm
fi

COORD_CMD=(
  uv run --python "${PYTHON_BIN}" python "${PIPELINE_PY}" coordinator
  --dataset-path "${DATASET_PATH}"
  --image-base "${IMAGE_BASE}"
  --processing-log-path "${PROCESSING_LOG_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --steps "${STEPS}"
  --seed-base "${SEED_BASE}"
  --subject-offset "${SUBJECT_OFFSET}"
  --gen-gpus "${GEN_GPUS}"
  --vlm-url "http://127.0.0.1:${VLM_PORT}/v1/responses"
  --vlm-model "${VLM_MODEL}"
)

if [[ -n "${SUBJECT_LIMIT}" ]]; then
  COORD_CMD+=(--subject-limit "${SUBJECT_LIMIT}")
fi

if [[ -n "${PROMPT_LIMIT}" ]]; then
  COORD_CMD+=(--prompt-limit "${PROMPT_LIMIT}")
fi

if [[ "${RESUME}" -eq 1 ]]; then
  COORD_CMD+=(--resume)
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  COORD_CMD+=(--dry-run)
fi

echo "Running coordinator for ${RUN_NAME}..."
"${COORD_CMD[@]}"
