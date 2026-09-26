"""kev.serve with MLX memory kept in check.

Kev's MLX backend never releases Metal buffers, and every request has a different sequence length, so
MLX's buffer cache accumulates new shapes without bound: the 0.8B server reached 25 GB after a dozen
requests and pushed this 32 GB machine into swap. This wrapper caps the cache and clears it after
every batch. Run from the kev checkout's env:

    cd kev && uv run --extra serve python ../scripts/serve_kev.py --run jaredpalmer/kev-4b --port 8009
"""

import os
import sys

import kev.mlx_model as mlx_model
import mlx.core as mx
from kev.serve import main

mx.set_cache_limit(int(float(os.environ.get("KEVMAP_MLX_CACHE_GB", "0.5")) * (1 << 30)))
if os.environ.get("KEVMAP_MLX_MEMORY_GB"):
    mx.set_memory_limit(int(float(os.environ["KEVMAP_MLX_MEMORY_GB"]) * (1 << 30)))

_orig_probs_batch = mlx_model.MLXDecisionModel.probs_batch


def _probs_batch(self, *args, **kwargs):
    try:
        return _orig_probs_batch(self, *args, **kwargs)
    finally:
        mx.clear_cache()


mlx_model.MLXDecisionModel.probs_batch = _probs_batch

sys.argv[0] = "kev.serve"
main()
