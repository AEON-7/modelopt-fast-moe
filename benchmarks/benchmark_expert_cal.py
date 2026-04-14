"""Empirical bottleneck test for Gemma4 MoE quantization.

Question: in the awq_lite search phase, which is the bottleneck — Python dispatch
overhead (per-linear hook call) or GPU compute (per-token GEMM work)?

Test: run a mini MoE layer through awq_lite calibration at batch sizes 1, 4, 16.
If Python overhead dominates, wall-clock per sample DROPS with larger batch
(overhead amortized).  If GPU dominates, wall-clock per sample scales linearly
with batch.
"""
import copy
import os
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry


# ── Mini Gemma4TextExperts for testing (reduced config) ──────────────────────
class MiniExperts(nn.Module):
    """Same structure as Gemma4TextExperts but tiny."""

    def __init__(self, num_experts=8, hidden_dim=256, intermediate_dim=128):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16) * 0.02
        )
        self.act_fn = nn.functional.silu


# ── v1 plugin (copy of what's running as PID 2318) ───────────────────────────
class _Gemma4ExpertModule(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)


@QuantModuleRegistry.register({MiniExperts: "mini_experts"})
class _QuantMiniExperts(QuantModule):
    def _setup(self):
        from accelerate import init_empty_weights
        dtype, device = self.gate_up_proj.dtype, self.gate_up_proj.device

        def _copy_weight(module, weight):
            module.to_empty(device=device)
            with torch.no_grad():
                module.weight.data = weight.detach().data.to(dtype=dtype, device=device)

        expert_dim = self.intermediate_dim
        with init_empty_weights():
            expert_modules = nn.ModuleList([
                _Gemma4ExpertModule(self.hidden_dim, expert_dim)
                for _ in range(self.num_experts)
            ])

        for idx in range(self.num_experts):
            _copy_weight(expert_modules[idx].gate_proj, self.gate_up_proj[idx, :expert_dim, :])
            _copy_weight(expert_modules[idx].up_proj, self.gate_up_proj[idx, expert_dim:, :])
            _copy_weight(expert_modules[idx].down_proj, self.down_proj[idx])

        delattr(self, "gate_up_proj")
        delattr(self, "down_proj")
        for idx in range(self.num_experts):
            self.add_module(str(idx), expert_modules[idx])

    def __len__(self):
        return self.num_experts

    def __iter__(self):
        for idx in range(self.num_experts):
            yield getattr(self, str(idx))

    def __getitem__(self, idx):
        return getattr(self, str(int(idx)))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert = self[expert_idx]
            gate = expert.gate_proj(current_state)
            up = expert.up_proj(current_state)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = expert.down_proj(current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


# ── Router mock (produces random top-k routing) ──────────────────────────────
class MockRouter(nn.Module):
    def __init__(self, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        scores = torch.randn(B * S, self.num_experts, device=hidden_states.device)
        probs = torch.softmax(scores, dim=-1)
        top_k_weights, top_k_index = torch.topk(probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_weights, top_k_index


class MoEBlock(nn.Module):
    def __init__(self, num_experts=8, hidden_dim=256, intermediate_dim=128, top_k=2):
        super().__init__()
        self.router = MockRouter(num_experts, top_k)
        self.experts = MiniExperts(num_experts, hidden_dim, intermediate_dim)

    def forward(self, x):
        B, S, H = x.shape
        top_k_weights, top_k_index = self.router(x)
        x_flat = x.view(B * S, H)
        out = self.experts(x_flat, top_k_index, top_k_weights)
        return out.view(B, S, H)


def build_model(num_experts=8, hidden_dim=256, intermediate_dim=128, top_k=2):
    model = MoEBlock(num_experts, hidden_dim, intermediate_dim, top_k).cuda().to(torch.bfloat16)
    return model


def apply_quant(model):
    """Apply modelopt NVFP4_AWQ_FULL calibration setup (mirrors our SuperGemma4 run)."""
    cfg = copy.deepcopy(mtq.NVFP4_AWQ_FULL_CFG)
    # No vision/multi-modal modules in this mini test — skip the standard excludes

    def calibrate_loop(model):
        B = getattr(calibrate_loop, "batch_size", 1)
        S = 512  # much smaller seq len for this test
        for i in range(getattr(calibrate_loop, "n_samples", 4)):
            ids = torch.randn(B, S, model.router.num_experts * 32, dtype=torch.bfloat16, device="cuda")
            model(ids)

    return mtq.quantize(model, cfg, forward_loop=calibrate_loop), calibrate_loop


def time_forwards(model, batch_size, seq_len, n_forwards=3, device="cuda"):
    """Time N forward passes through calibrated model at given batch size."""
    hidden_dim = model.router.num_experts * 32  # matches build
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize()
    # warmup
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_forwards):
            _ = model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_forwards


# ── Main: minimal awq_lite calibration bottleneck sweep ──────────────────────
def main():
    print("=" * 70)
    print("  Gemma4 MoE quantization bottleneck test")
    print("  Config: 8 experts, hidden=256, intermediate=128, top_k=2")
    print("=" * 70)

    # Build + calibrate a fresh model (this is the SLOW part — the whole point of the test)
    print("\n[setup] Building mini MoE...")
    model = build_model()
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    print("\n[setup] Applying NVFP4_AWQ_FULL calibration (mini scale)...")
    t0 = time.perf_counter()
    model, cal_loop = apply_quant(model)
    cal_time = time.perf_counter() - t0
    print(f"  Calibration wall-clock: {cal_time:.2f}s")

    # After calibration, time forward passes at varying batch sizes.
    # These DO NOT trigger the 11-alpha awq_lite loop (that only runs during calibration)
    # — but they help us understand the raw expert-forward cost structure.
    print("\n[bench] Forward-pass timing after calibration:")
    print("   bs |  per-forward  |  per-sample  |  GPU-time-ratio")
    print("  ----+---------------+--------------+------------------")
    baseline = None
    for bs in [1, 4, 16]:
        try:
            t = time_forwards(model, bs, seq_len=512)
            per_sample = t / bs
            if baseline is None:
                baseline = per_sample
            ratio = per_sample / baseline
            print(f"  {bs:3d} |   {t*1000:7.1f}ms  |  {per_sample*1000:7.1f}ms  |  {ratio:.2f}x")
        except Exception as e:
            print(f"  {bs:3d} |  ERROR: {e}")

    print("\n[interpretation]")
    print("  - per-sample time DROPPING with bs: Python-dispatch-bound (our v1 problem)")
    print("  - per-sample time FLAT with bs:      GPU-bound (no win from batching)")
    print("  - per-sample time RISING with bs:    memory-bound (rare)")


if __name__ == "__main__":
    main()
