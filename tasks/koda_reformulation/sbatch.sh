#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --nodes=1
#SBATCH -p lem-gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=hopper:2
#SBATCH --tasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=sbatch-serve-model.log
#SBATCH --signal=B:SIGINT@120
set -euo pipefail

PID_A=""
PID_B=""
PID_MLFLOW=""
PID_DVC=""

cleanup() {
  local status=$?
  trap - INT TERM EXIT
  set +e

  echo "[INFO]: Cleaning up..."

  for pid in "$PID_DVC" "$PID_MLFLOW" "$PID_A" "$PID_B"; do
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      echo "[INFO]: Stopping process group $pid"
      kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
      echo "[INFO]: Stopped process group $pid"
    fi
  done

  sleep 3

  for pid in "$PID_DVC" "$PID_MLFLOW" "$PID_A" "$PID_B"; do
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      echo "[WARN]: Force stopping process group $pid"
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      echo "[WARN]: Force stopped process group $pid"
    fi
  done
  echo "[INFO]: Waiting for stderr flush"
  wait 2>/dev/null || true
  echo "[INFO]: Exiting"
  exit "$status"
}

trap cleanup INT TERM EXIT

NUM_GPUS_PER_NODE=4
NUM_NODES=1

GENERATOR_MODEL_PATH=/home/stamar2923/lustre_dir/stanislawm/huggingface_cache/models--google--gemma-4-26B-A4B-it/snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52
GENERATOR_MODEL_NAME=gemma-4-26B-A4B-it

REFLECTION_MODEL_PATH=/home/stamar2923/lustre_dir/stanislawm/huggingface_cache/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
REFLECTION_MODEL_NAME=gpt-oss-120b

# Make sure you have enough space in the cache directories, or set them to a directory with enough space. You can also set them to a shared filesystem like Lustre.
echo [INFO]: Using HF_HUB_CACHE: $HF_HUB_CACHE
echo [INFO]: Using UV_CACHE_DIR: $UV_CACHE_DIR
echo [INFO]: Using VLLM_CACHE_ROOT: $VLLM_CACHE_ROOT

# --- modules ---
source /usr/local/sbin/modules.sh
module load Python/3.11.3-GCCcore-12.3.0 # same as your project
module load CUDA/13.0.0 # same as your project
python_version=3.11 # same as module that was loaded

if [ -z "${EBROOTPYTHON+x}" ]; then
    echo [ERROR]: Python module not loaded. Please load the Python module before running this script.
    exit 1
fi
PYTHON_LIB_PATH="$EBROOTPYTHON/lib"
echo [INFO]: Python library path: $PYTHON_LIB_PATH

nvidia-smi || echo "No NVIDIA GPU detected, proceeding without GPU support."

cp -a . $TMPDIR

echo [INFO]: Creating virtual environments on nodes
srun --label --chdir=$TMPDIR uv venv --python $python_version
echo [INFO]: Installing packages from repo on nodes
srun --label --chdir=$TMPDIR uv sync
echo [INFO]: Installing vllm in the virtual environment on nodes
srun --label --chdir=$TMPDIR uv pip install --quiet --python $python_version vllm==0.26.0 "tokenizers>=0.22.0,<0.23" ninja

export VIRTUAL_ENV="$TMPDIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

setsid env CUDA_VISIBLE_DEVICES=0 "$VIRTUAL_ENV/bin/vllm" serve $GENERATOR_MODEL_PATH --data-parallel-size 1  --served-model-name $GENERATOR_MODEL_NAME --port 8001 > ~/test-remote/vllm-model-generator.log 2>&1 &
PID_A=$!

setsid env CUDA_VISIBLE_DEVICES=1 "$VIRTUAL_ENV/bin/vllm" serve $REFLECTION_MODEL_PATH --data-parallel-size 1 --served-model-name $REFLECTION_MODEL_NAME --port 8002 > ~/test-remote/vllm-model-reflection.log 2>&1 &
PID_B=$!

echo "[INFO]: Started model A on port 8001 with PID $PID_A"
echo "[INFO]: Started model B on port 8002 with PID $PID_B"

wait_for_vllm() {
  local port="$1"
  local pid="$2"
  local name="$3"

  echo "[INFO]: Waiting for $name on port $port..."
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR]: $name process died before becoming ready"
      exit 1
    fi
    sleep 5
  done
  echo "[INFO]: $name is ready on port $port"
}

wait_for_vllm 8001 "$PID_A" "$GENERATOR_MODEL_NAME" &
WAIT_A=$!

wait_for_vllm 8002 "$PID_B" "$REFLECTION_MODEL_NAME" &
WAIT_B=$!

wait "$WAIT_A" "$WAIT_B"
echo "[INFO]: Both vLLM servers are ready"

cd $TMPDIR/tasks/koda_reformulation

setsid "$VIRTUAL_ENV/bin/mlflow" server --host $(yq '.mlflow_tracking_host' ./params.yaml ) --port $(yq '.mlflow_tracking_port' ./params.yaml ) --backend-store-uri sqlite:///mlflow.db --no-serve-artifacts &
PID_MLFLOW=$!

setsid "$VIRTUAL_ENV/bin/dvc" exp run -f &
PID_DVC=$!

wait "$PID_DVC"
