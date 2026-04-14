"""Measure wall-clock of awq_lite CALIBRATION (not just post-cal forwards)
at different calibration batch sizes.

If batching calibration samples gives a similar speedup to post-cal forwards,
the fix is trivially a ~3-line patch to the calibrate_loop.
"""
import copy
import os
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry


# Reuse the plumbing from the previous benchmark (128 experts this time — closer to real SuperGemma4)
class MiniExperts(nn.Module):
    def __init__(self, num_experts=32, hidden_dim=512, intermediate_dim=256):
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

    def __len__(self): return self.num_experts
    def __iter__(self):
        for idx in range(self.num_experts): yield getattr(self, str(idx))
    def __getitem__(self, idx): return getattr(self, str(int(idx)))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts: continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert = self[expert_idx]
            gate = expert.gate_proj(current_state)
            up = expert.up_proj(current_state)
            h = self.act_fn(gate) * up
            h = expert.down_proj(h)
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
        return final_hidden_states


class MockRouter(nn.Module):
    def __init__(self, num_experts, top_k): super().__init__(); self.num_experts=num_experts; self.top_k=top_k
    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        scores = torch.randn(B*S, self.num_experts, device=hidden_states.device)
        probs = torch.softmax(scores, dim=-1)
        w, i = torch.topk(probs, self.top_k, dim=-1)
        return w / w.sum(dim=-1, keepdim=True), i


class MoEBlock(nn.Module):
    def __init__(self, ne=32, hd=512, id_=256, tk=4):
        super().__init__()
        self.router = MockRouter(ne, tk)
        self.experts = MiniExperts(ne, hd, id_)
    def forward(self, x):
        B, S, H = x.shape
        w, i = self.router(x)
        out = self.experts(x.view(B*S, H), i, w)
        return out.view(B, S, H)


# ── Sweep calibration batch size ─────────────────────────────────────────────
def run_calibration(batch_size, num_samples, seq_len, num_experts=32, hidden_dim=512, intermediate_dim=256):
    """Build a fresh model + run awq_full calibration at given batch size, return wall-clock."""
    torch.manual_seed(0)
    model = MoEBlock(num_experts, hidden_dim, intermediate_dim, 4).cuda().to(torch.bfloat16)

    cfg = copy.deepcopy(mtq.NVFP4_AWQ_FULL_CFG)

    def cal_loop(m):
        # Generate num_samples worth of data, processed in batches of batch_size
        for batch_start in range(0, num_samples, batch_size):
            actual_bs = min(batch_size, num_samples - batch_start)
            x = torch.randn(actual_bs, seq_len, hidden_dim, dtype=torch.bfloat16, device="cuda")
            m(x)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    mtq.quantize(model, cfg, forward_loop=cal_loop)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    N_SAMPLES = 16      # fixed total work
    SEQ_LEN = 512       # fixed seq length
    NE, HD, ID = 32, 512, 256  # mid-size MoE (scaled from real 128/2816/704)

    print("=" * 75)
    print(f"  AWQ_FULL calibration wall-clock vs batch size")
    print(f"  Total samples: {N_SAMPLES} | seq_len: {SEQ_LEN}")
    print(f"  MoE: {NE} experts, hidden={HD}, intermediate={ID}, top_k=4")
    print("=" * 75)
    print()
    print("  bs |  wall-clock  |  speedup  |  per-sample")
    print("  ---+--------------+-----------+-------------")

    baseline = None
    for bs in [1, 2, 4, 8]:
        try:
            t = run_calibration(bs, N_SAMPLES, SEQ_LEN, NE, HD, ID)
            if baseline is None:
                baseline = t
            speedup = baseline / t
            per_sample = t / N_SAMPLES
            print(f"  {bs:2d} |  {t:7.2f}s   |  {speedup:5.2f}x  |  {per_sample*1000:7.1f}ms")
        except Exception as e:
            print(f"  {bs:2d} |  ERROR: {str(e)[:50]}")


if __name__ == "__main__":
    main()
