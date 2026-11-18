#!/usr/bin/env bash
# Builds the test-time Megatron environment the oracle and the pipeline replica need
# (plan 5.3, F5, F6). It is test infrastructure: nothing here ships with the tool, and
# no Megatron code is copied into the repository (plan 3.5).
#
# Usage: tools/setup_megatron_env.sh <work-dir> [<hf-home-seed-dir>]
#   <work-dir>           a folder outside the tracked files (for example .work/megatron)
#   <hf-home-seed-dir>   optional: an existing HF_HOME that already holds the gpt2
#                        tokenizer files; it is copied instead of fetched from the hub
#
# What it pins, and why (each condition was shown in a CPU-only run before the build):
#   - a uv-managed CPython 3.12 with headers (the C++ helpers compile against it)
#   - torch from the CPU wheel index (no GPU stack)
#   - Megatron-LM at tag core_v0.20.0, installed from source (tools/preprocess_data.py
#     is not in the megatron-core package)
#   - triton (import megatron.core fails without it)
#   - transformers (the GPT-2 BPE tokenizer class needs it)
#   - Megatron-Bridge v0.7.0 as source only (its sampler file is loaded by path)
set -euo pipefail

WORK=${1:?usage: setup_megatron_env.sh <work-dir> [<hf-home-seed-dir>]}
HF_SEED=${2:-}
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)

LM_URL=https://github.com/NVIDIA/Megatron-LM.git
LM_TAG=core_v0.20.0
LM_COMMIT=685fb8bd970185e9ac3283fe32a5d6667bf4b268
BRIDGE_URL=https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
BRIDGE_TAG=v0.7.0
BRIDGE_COMMIT=04ce2b3efd6f2e387fc244e9f37afd0a1f7952fb

export UV_PYTHON_INSTALL_DIR="$WORK/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_PYTHON_INSTALL_BIN=0

uv python install 3.12
[ -x "$WORK/venv/bin/python" ] || uv venv --python 3.12 "$WORK/venv"
PY="$WORK/venv/bin/python"

clone_at() { # url tag commit dest
  if [ ! -d "$4/.git" ]; then
    git clone --quiet --depth 1 --branch "$2" "$1" "$4"
  fi
  local head
  head=$(git -C "$4" rev-parse HEAD)
  if [ "$head" != "$3" ]; then
    echo "error: $4 is at $head, expected $3 ($2)" >&2
    exit 1
  fi
}
mkdir -p "$WORK/src"
clone_at "$LM_URL" "$LM_TAG" "$LM_COMMIT" "$WORK/src/Megatron-LM"
clone_at "$BRIDGE_URL" "$BRIDGE_TAG" "$BRIDGE_COMMIT" "$WORK/src/Megatron-Bridge"

uv pip install --quiet --python "$PY" --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
uv pip install --quiet --python "$PY" "$WORK/src/Megatron-LM"
uv pip install --quiet --python "$PY" triton==3.9.0 transformers==5.19.0 numpy==2.5.4 pytest

# The helper build is optional in Megatron's setup.py, so a failed compile passes
# silently at install. This import is the check.
"$PY" -I -c "import megatron.core.datasets.helpers_cpp" 

mkdir -p "$WORK/hf_home"
if [ -n "$HF_SEED" ]; then
  cp -rn "$HF_SEED/." "$WORK/hf_home/"
fi
# Build the tokenizer once in this process: an error inside preprocessing's worker
# pool hangs the tool instead of exiting. Offline when a seed was copied.
if [ -n "$HF_SEED" ]; then export HF_HUB_OFFLINE=1; fi
HF_HOME="$WORK/hf_home" "$PY" -I -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('gpt2')
assert t.eos_token_id == 50256, t.eos_token_id
print('tokenizer ok, eos', t.eos_token_id)
"

"$PY" -I - <<'PYEOF'
import torch, numpy, megatron.core, triton, transformers
from importlib.metadata import version
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("megatron-core", version("megatron-core"))
print("numpy", numpy.__version__, "triton", triton.__version__, "transformers", transformers.__version__)
PYEOF
echo "environment ready: $WORK"
