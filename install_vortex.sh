#!/usr/bin/env bash
#
# Reproducible build of the `vortex_v1` conda environment — the default env for
# this project (all the slash commands, RULER, and AIME24 runners expect it).
#
# Unlike `vortex_glm` (see install_vortex_glm.sh), this env KEEPS the pinned
# transformers==4.57.1 that both sglang and vortex_torch require — there is NO
# transformers override step. It therefore does NOT load GLM-4.7-Flash
# (`glm4_moe_lite` needs transformers >= 5.0); use install_vortex_glm.sh for that.
#
# Captured from the working env: python 3.12, torch 2.9.1+cu128, torchvision
# 0.24.1, torchaudio 2.9.1, flashinfer 0.6.3, transformers 4.57.1, sglang
# (editable, vendored v0.5.9), vortex_torch (editable).
#
# Usage:
#   bash install_vortex.sh            # create the env
#   FORCE=1 bash install_vortex.sh    # remove an existing env first
#   ENV_NAME=vortex2 bash install_vortex.sh   # build under a different name
#
# Install only needs CPU (all kernels are prebuilt wheels or JIT-compiled at
# runtime), so it works even while the GPUs are busy.

set -euo pipefail

ENV_NAME="${ENV_NAME:-vortex_v1}"
PY_VER="${PY_VER:-3.12}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$REPO_ROOT/third_party/sglang/v0.5.9/sglang/python"
[ -d "$SGLANG_DIR" ] || { echo "ERROR: vendored sglang not found at $SGLANG_DIR" >&2; exit 1; }

# ---- conda bootstrap -------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [ "${FORCE:-0}" = "1" ]; then
        echo ">>> removing existing env '$ENV_NAME' (FORCE=1)"
        conda env remove -y -n "$ENV_NAME"
    else
        echo "ERROR: conda env '$ENV_NAME' already exists. Re-run with FORCE=1 to recreate." >&2
        exit 1
    fi
fi

echo ">>> [1/4] creating conda env '$ENV_NAME' (python $PY_VER)"
conda create -y -n "$ENV_NAME" python="$PY_VER"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# ---- 2. torch (CUDA 12.8 build) pinned first ------------------------------
# Pinned before sglang sees `torch==2.9.1` so the exact CUDA build is locked in
# and torchvision/torchaudio match. (Default PyPI torch 2.9.1 is the cu128 wheel.)
echo ">>> [2/4] installing torch 2.9.1 + torchvision 0.24.1 + torchaudio 2.9.1 (cu128)"
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

# ---- 3. sglang (editable, vendored) ---------------------------------------
# Pulls the runtime tree: flashinfer_python/cubin 0.6.3, sgl-kernel, xgrammar,
# outlines, cuda-python, etc. (and the pinned transformers 4.57.1 — KEPT here).
echo ">>> [3/4] installing vendored sglang (editable) from $SGLANG_DIR"
pip install -e "$SGLANG_DIR"

# ---- 4. vortex_torch (editable) -------------------------------------------
echo ">>> [4/4] installing vortex_torch (editable) from $REPO_ROOT"
pip install -e "$REPO_ROOT"

# ---- 4b. gt_score_kernels (editable, optional) ----------------------------
# Only the ground-truth submission (submissions/ground_truth_kernel_topk) needs
# it, and its ops import it lazily. Editable install from the sibling checkout
# (not on PyPI; GitHub is tunnel-only here). Skipped with a warning if absent.
GT_KERNELS_DIR="${GT_KERNELS_DIR:-$(dirname "$REPO_ROOT")/gt_score_kernels}"
if [ -f "$GT_KERNELS_DIR/pyproject.toml" ]; then
    echo ">>> [4b] installing gt_score_kernels (editable) from $GT_KERNELS_DIR"
    pip install -e "$GT_KERNELS_DIR"
else
    echo ">>> [4b] gt_score_kernels not found at $GT_KERNELS_DIR — skipping."
    echo "        (needed only for the ground_truth_kernel_topk submission;"
    echo "         set GT_KERNELS_DIR or 'pip install -e <path>' manually.)"
fi

# ---- verify ----------------------------------------------------------------
echo ">>> verifying the environment"
python - <<'PY'
import torch, transformers, sglang, vortex_torch
import flashinfer
print(f"  python        : {__import__('sys').version.split()[0]}")
print(f"  torch         : {torch.__version__}  (cuda {torch.version.cuda})")
print(f"  transformers  : {transformers.__version__}")
print(f"  flashinfer    : {flashinfer.__version__}")
print(f"  sglang        : {sglang.__version__}")
print(f"  vortex_torch  : {getattr(vortex_torch, '__version__', '?')}")
ok = transformers.__version__.startswith("4.57")
print(f"  transformers 4.57.x (sglang/vortex_torch pin): {ok}")
assert ok, f"expected transformers 4.57.x, got {transformers.__version__}"
PY

echo ""
echo ">>> done. Activate with:  conda activate $ENV_NAME"
echo ">>> sanity run (needs a free GPU):"
echo "    conda activate $ENV_NAME"
echo "    CUDA_VISIBLE_DEVICES=<gpu> python algorithm_scientist/run_ruler.py --config <submission>.json"
