"""modelopt-fast-moe: VRAM-adaptive batched calibration for nvidia-modelopt MoE quantization.

Drop-in replacement for the naive ``for ids in calib_data: model(ids)`` forward_loop
that most modelopt examples use. On MoE models with per-expert Python dispatch
(e.g. Gemma4, Mixtral, Qwen-MoE), that naive loop leaves the GPU at ~25-30%
utilization — batching amortizes the per-sample Python overhead and cuts
calibration wall-clock by 3-10x in our tests.

Core entry point:
    make_adaptive_calibrate_loop(calib_data, forward_fn=...) -> callable

Example:
    from modelopt_fast_moe import make_adaptive_calibrate_loop
    import modelopt.torch.quantization as mtq

    calib_data = [...]  # list of input tensors
    forward_loop = make_adaptive_calibrate_loop(calib_data)
    mtq.quantize(model, mtq.NVFP4_AWQ_FULL_CFG, forward_loop=forward_loop)
"""

from .adaptive_bs import (
    make_adaptive_calibrate_loop,
    pick_safe_batch_size,
)

__all__ = ["make_adaptive_calibrate_loop", "pick_safe_batch_size"]
__version__ = "0.1.0"
