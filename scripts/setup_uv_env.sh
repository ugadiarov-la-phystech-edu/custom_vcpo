#!/bin/bash
# Create the VCPO uv environment on any machine.
#
# Usage: bash scripts/setup_uv_env.sh [ENV_DIR]
#
#   ENV_DIR         where to create the venv (1st argument or env var; default: $HOME/uv-envs/vcpo-env)
#   UV_ACTIVATE     script to source that puts `uv` on PATH (e.g. one exporting UV_CACHE_DIR/UV_PYTHON_INSTALL_DIR)
#   INSTALL_UV=1    install uv into $HOME/.local/bin when it cannot be found (set 0 to fail instead)
#   PYTHON_VERSION  default 3.12 (the prebuilt flash-attn wheel is cp312; override FLASH_ATTN_WHEEL for others)
#   USE_MEGATRON=1  install Megatron-Core + TransformerEngine
#   FORCE=1         remove an existing ENV_DIR first
#
# uv's own variables (UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR, UV_LINK_MODE) are respected; put the cache on the same
# filesystem as ENV_DIR when the home directory is small.
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

ENV_DIR=${1:-${ENV_DIR:-$HOME/uv-envs/vcpo-env}}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
USE_MEGATRON=${USE_MEGATRON:-1}
FORCE=${FORCE:-0}
INSTALL_UV=${INSTALL_UV:-1}
FLASH_ATTN_WHEEL=${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$(dirname "$ENV_DIR")"
ENV_DIR=$(cd "$(dirname "$ENV_DIR")" && pwd)/$(basename "$ENV_DIR")

if [ "$PYTHON_VERSION" != "3.12" ] && [[ "$FLASH_ATTN_WHEEL" == *cp312* ]]; then
    echo "ERROR: PYTHON_VERSION=$PYTHON_VERSION but FLASH_ATTN_WHEEL is a cp312 wheel; set FLASH_ATTN_WHEEL to a matching one."
    exit 1
fi

UV_SOURCED=""
if [ -n "${UV_ACTIVATE:-}" ]; then
    source "$UV_ACTIVATE"
    UV_SOURCED=$UV_ACTIVATE
fi
if ! command -v uv > /dev/null; then
    if [ "$INSTALL_UV" -eq 1 ]; then
        echo "0. uv not found: installing it into $HOME/.local/bin"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
        export PATH="$HOME/.local/bin:$PATH"
        UV_SOURCED=""
    else
        echo "ERROR: uv not found. Put it on PATH, set UV_ACTIVATE=/path/to/script, or rerun with INSTALL_UV=1."
        exit 1
    fi
fi
echo "uv: $(command -v uv) ($(uv --version))"

if [ -e "$ENV_DIR" ]; then
    if [ "$FORCE" -eq 1 ]; then
        echo "FORCE=1: removing existing $ENV_DIR"
        rm -rf "$ENV_DIR"
    else
        echo "ERROR: $ENV_DIR already exists (and may be the production env)."
        echo "Pick another path (bash $0 /new/path or ENV_DIR=...) or pass FORCE=1 to recreate it."
        exit 1
    fi
fi

echo "1. Create venv at $ENV_DIR (Python $PYTHON_VERSION)"
uv venv "$ENV_DIR" --python "$PYTHON_VERSION"
source "$ENV_DIR/bin/activate"

echo "2. Install vLLM 0.11.0 (pins torch 2.8.0), verl 0.7.0 and base packages"
uv pip install \
    "vllm==0.11.0" \
    "verl[vllm,mcore,test]==0.7.0" \
    "transformers[hf_xet]>=4.51.0,<5" \
    "datasets>=3.0" "dill<0.3.9" "numpy<2.0.0" "pyarrow>=19.0.0" "huggingface-hub<1" \
    hf-transfer qwen-vl-utils mathruler liger-kernel \
    "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" \
    uvicorn latex2sympy2_extended math_verify ruff opencv-python-headless matplotlib \
    "torchao==0.17.0"

echo "3. Install FlashAttention (prebuilt wheel for cp312/torch2.8/cu12) and FlashInfer"
uv pip install --no-cache-dir "$FLASH_ATTN_WHEEL"
uv pip install "flashinfer-python==0.3.1"

if [ "$USE_MEGATRON" -eq 1 ]; then
    echo "4. Install Megatron-Core 0.13.1 and TransformerEngine 2.6"
    uv pip install "onnxscript==0.3.1"
    uv pip install --no-deps "megatron-core==0.13.1" mbridge
    SP=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    if ! command -v nvcc > /dev/null; then
        echo "No system nvcc: using pip CUDA headers + shim nvcc for the TE build"
        uv pip install "nvidia-cuda-runtime-cu12==12.8.*" "nvidia-cuda-nvcc-cu12==12.8.*" "nvidia-cuda-cccl-cu12==12.8.*"
        FAKE_CUDA="$ENV_DIR/fake_cuda"
        mkdir -p "$FAKE_CUDA/bin" "$FAKE_CUDA/include"
        printf '#!/bin/bash\necho "Cuda compilation tools, release 12.8, V12.8.93"\n' > "$FAKE_CUDA/bin/nvcc"
        chmod +x "$FAKE_CUDA/bin/nvcc"
        cp -rs "$SP/nvidia/cuda_runtime/include/." "$FAKE_CUDA/include/" 2>/dev/null || true
        export CUDA_HOME="$FAKE_CUDA"
        export PATH="$FAKE_CUDA/bin:$PATH"
        for inc_dir in "$SP"/nvidia/*/include; do
            [ -d "$inc_dir" ] && export CPATH="$inc_dir:${CPATH:-}"
        done
    fi
    CUDNN_PATH="$SP/nvidia/cudnn"
    export NVTE_FRAMEWORK=pytorch MAX_JOBS=${MAX_JOBS:-8}
    export CPATH="$CUDNN_PATH/include:${CPATH:-}" LIBRARY_PATH="$CUDNN_PATH/lib:${LIBRARY_PATH:-}"
    uv pip install --no-build-isolation "transformer_engine[pytorch]==2.6.0.post1"
    sed -i 's/^Version: 2\.6\.0$/Version: 2.6.0.post1/' \
        "$SP"/transformer_engine_torch-2.6.0.dist-info/METADATA 2>/dev/null || true
    echo "5. Pin cudnn python package (avoid being overridden)"
    uv pip install "nvidia-cudnn-cu12==9.10.2.21"
fi

echo "5b. Re-pin numpy / ml-dtypes"
uv pip install "numpy<2.0.0" "ml-dtypes==0.5.4"

echo "6. Sanity check"
python - <<'EOF'
import torch, vllm, verl
print("torch", torch.__version__, "cuda", torch.version.cuda, "available:", torch.cuda.is_available())
print("vllm", vllm.__version__)
print("verl", verl.__version__)
import flash_attn; print("flash_attn", flash_attn.__version__)
import numpy; print("numpy", numpy.__version__)
try:
    import torchao; print("torchao", torchao.__version__)
except Exception as e:
    print("torchao: NOT OK:", e)
try:
    import megatron.core; print("megatron-core", megatron.core.__version__)
except Exception as e:
    print("megatron-core: NOT OK:", e)
try:
    import transformer_engine; print("transformer_engine", transformer_engine.__version__)
except Exception as e:
    print("transformer_engine: NOT OK:", e)
EOF

ACTIVATE="$ENV_DIR/activate_vcpo.sh"
{
    echo "# Generated by scripts/setup_uv_env.sh: puts uv on PATH, activates the venv and enters the repo."
    if [ -n "$UV_SOURCED" ]; then echo "source \"$UV_SOURCED\""; else echo "export PATH=\"$(dirname "$(command -v uv)"):\$PATH\""; fi
    for v in UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_LINK_MODE; do
        [ -n "${!v:-}" ] && echo "export $v=\"${!v}\""
    done
    echo "source \"$ENV_DIR/bin/activate\""
    echo "cd \"$REPO_ROOT\""
} > "$ACTIVATE"

echo "Done. Activate with: source $ACTIVATE"
echo "      (or just the venv: source $ENV_DIR/bin/activate)"
echo "NOTE: run training from the repo root so the local (fork) 'verl' package shadows the installed verl."
