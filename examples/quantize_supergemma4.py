"""Real-world example: NVFP4 AWQ Full quantization of SuperGemma4 26B MoE.

This is the exact script used to go from a 50h-projected naive calibration job
to ~1.7h on a GB10 (97 GB VRAM). Shows:

  1. How to register a custom per-expert plugin for a 3D-fused MoE (Gemma4 stores
     experts as [E, 2I, H] + [E, H, I], modelopt needs per-expert Linears).
  2. How to wire up modelopt-fast-moe's adaptive calibration loop.
  3. Tier 1 tuning knobs (CALIB_SAMPLES, alpha_step) that matter more than the
     batching itself for end-to-end wall-clock.
  4. Post-export vLLM key fix (modelopt strips `.moe.` from expert paths but vLLM
     expects them).

Run: python3 examples/quantize_supergemma4.py
"""

import copy
import glob
import os
import re
import sys
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry
from safetensors.torch import load_file, save_file

from modelopt_fast_moe import make_adaptive_calibrate_loop


# ── Per-expert plugin for Gemma4's 3D-fused MoE storage ──────────────────────

class _Gemma4ExpertModule(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)


class _QuantGemma4TextExperts(QuantModule):
    """Decomposes Gemma4's [E, 2I, H] + [E, H, I] fused tensors into per-expert
    nn.Linear modules so modelopt's per-linear AWQ hooks can attach."""

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
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            with torch.no_grad():
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


def fix_keys_for_vllm(model_dir):
    """modelopt strips `.moe.` from expert tensor keys during export; vLLM expects it."""
    shard_files = sorted(glob.glob(f"{model_dir}/model-*.safetensors"))
    single_file = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single_file) and not shard_files:
        all_tensors = load_file(single_file)
    else:
        all_tensors = {}
        for f in shard_files:
            all_tensors.update(load_file(f))

    new_tensors = {}
    renamed = 0
    for key, tensor in all_tensors.items():
        new_key = re.sub(r"\.experts\.(\d+)\.", r".moe.experts.\1.", key)
        if new_key != key:
            renamed += 1
        new_tensors[new_key] = tensor

    print(f"  Renamed {renamed} keys (added moe. prefix)")
    if os.path.exists(single_file):
        os.remove(single_file)
    for f in shard_files:
        os.remove(f)
    idx_file = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_file):
        os.remove(idx_file)

    save_file(new_tensors, single_file)
    print(f"  Saved {len(new_tensors)} tensors to model.safetensors")


# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = "/path/to/your/bf16/model"
SAVE_PATH = "/path/to/output/nvfp4-awq"
CALIB_SAMPLES = 512        # modelopt default; AWQ paper uses 128
CALIB_SEQ_LEN = 4096       # long-context hedge; drop to 2048 if quality allows
MAX_BS = 16                # adaptive probe will cap at real VRAM budget
SAFETY_FRACTION = 0.85
PHASE2_OVERHEAD_FACTOR = 2.0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"{'='*60}")
    print(f"  SuperGemma4 26B MoE -> NVFP4 AWQ Full")
    print(f"  Source: {MODEL_PATH}")
    print(f"  Calibration: {CALIB_SAMPLES} samples x {CALIB_SEQ_LEN} tokens")
    print(f"  Adaptive bs: min=1, max={MAX_BS}, safety={SAFETY_FRACTION*100:.0f}%")
    print(f"{'='*60}\n")

    # Register plugin
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
    if Gemma4TextExperts not in QuantModuleRegistry._registry:
        QuantModuleRegistry.register({Gemma4TextExperts: "hf.Gemma4TextExperts"})(
            _QuantGemma4TextExperts
        )

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    print(f"Model loaded in {time.time()-t0:.0f}s, GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Prepare calibration data
    from datasets import load_dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True)

    calib_data = []
    for sample in dataset:
        if len(calib_data) >= CALIB_SAMPLES:
            break
        text = sample.get("article", "") + " " + sample.get("highlights", "")
        if len(text) > 200:
            tokens = tokenizer(text, return_tensors="pt", max_length=CALIB_SEQ_LEN, truncation=True)
            calib_data.append(tokens.input_ids)
    print(f"Collected {len(calib_data)} samples")

    # Pad to uniform seq_len for batched calibration
    # TODO(0.2.0): length bucketing will remove the need for this
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded = []
    for ids in calib_data:
        if ids.shape[1] < CALIB_SEQ_LEN:
            padding = torch.full((1, CALIB_SEQ_LEN - ids.shape[1]), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, padding], dim=1)
        padded.append(ids)
    calib_data = padded

    # ★ This is where modelopt-fast-moe replaces the naive bs=1 loop ★
    def forward_fn(model, batch):
        return model(input_ids=batch)

    forward_loop = make_adaptive_calibrate_loop(
        calib_data, forward_fn=forward_fn,
        min_bs=1, max_bs=MAX_BS,
        safety_fraction=SAFETY_FRACTION,
        phase2_overhead_factor=PHASE2_OVERHEAD_FACTOR,
        progress_every=128,
    )

    # Quantize with Tier 1 tuning (alpha_step: 0.1 -> 0.2, saves ~1.8x on search phase)
    quant_cfg = copy.deepcopy(mtq.NVFP4_AWQ_FULL_CFG)
    quant_cfg["algorithm"] = {"method": "awq_full", "alpha_step": 0.2}
    quant_cfg["quant_cfg"]["*vision*"] = {"enable": False}
    quant_cfg["quant_cfg"]["*embed_vision*"] = {"enable": False}
    quant_cfg["quant_cfg"]["*multi_modal_projector*"] = {"enable": False}

    t1 = time.time()
    model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    print(f"Quantized in {time.time()-t1:.0f}s")

    # Export + vLLM key fix
    os.makedirs(SAVE_PATH, exist_ok=True)
    from modelopt.torch.export import export_hf_checkpoint
    export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    fix_keys_for_vllm(SAVE_PATH)

    print(f"\nDone. Output: {SAVE_PATH}")


if __name__ == "__main__":
    main()
