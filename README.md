# modelopt-fast-moe


[![☕ Tips](https://img.shields.io/badge/%E2%98%95_Tips-Support_the_work-ff5e5b?style=flat)](https://github.com/AEON-7/AEON-7#-support-the-work)
**VRAM-adaptive batched calibration for [nvidia-modelopt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) MoE quantization.**

A drop-in replacement for the naive `for ids in calib_data: model(ids)` forward loop that most modelopt examples use. On Mixture-of-Experts models (Gemma4, Mixtral, Qwen-MoE, DeepSeek-MoE, etc.) the naive loop leaves the GPU at 25-30% utilization because Python dispatch overhead dominates — batching amortizes that overhead and cuts calibration wall-clock by **3–10×** with zero change to the AWQ algorithm itself.

---

## The problem

You're running `mtq.quantize(model, NVFP4_AWQ_FULL_CFG, forward_loop=...)` on a 26B-param MoE and it's projected to take **50+ hours** on a GB10 / H100. That's wrong. The GPU is 25-30% utilized the whole time.

Root cause: modelopt's AWQ calibration calls `forward_loop(model)` twice (cache phase, then alpha-search phase). Most examples implement `forward_loop` as:

```python
def forward_loop(model):
    for ids in calib_data:          # bs=1, one Python dispatch per sample
        model(input_ids=ids)
```

On MoE models, each forward dispatches hundreds of tiny per-expert GEMM kernels through Python. Each CUDA launch is ~10μs of CPU overhead regardless of GEMM size. For small per-expert GEMMs (tokens-per-expert ≈ seq_len × top_k / num_experts is often < 128), overhead dominates the actual compute.

The fix is trivial: **batch the calibration samples**. One Python dispatch per batch instead of per sample. The AWQ statistics modelopt collects (per-channel `act_scale = mean(|activations|)`) are weighted by token count, not batch count — algorithm is batch-size invariant.

The tricky part is picking a batch size that fits in VRAM on arbitrary hardware, and handling the OOM case gracefully. That's what this package does.

---

## Install

```bash
pip install modelopt-fast-moe
```

Requires `torch>=2.3`. Does **not** hard-depend on `nvidia-modelopt` — you bring your own modelopt install.

---

## Usage

Before (naive, 3–10× slower on MoE):

```python
import modelopt.torch.quantization as mtq

def forward_loop(model):
    for ids in calib_data:
        model(input_ids=ids)

mtq.quantize(model, mtq.NVFP4_AWQ_FULL_CFG, forward_loop=forward_loop)
```

After (adaptive batched):

```python
import modelopt.torch.quantization as mtq
from modelopt_fast_moe import make_adaptive_calibrate_loop

forward_loop = make_adaptive_calibrate_loop(calib_data)
mtq.quantize(model, mtq.NVFP4_AWQ_FULL_CFG, forward_loop=forward_loop)
```

That's it. The loop probes VRAM on the first invocation, picks the largest safe batch size, and runs calibration. On any mid-run `OutOfMemoryError` it halves the batch size and retries.

### Custom forward signature

If your model takes more than just `input_ids`:

```python
def my_forward(model, batch):
    return model(input_ids=batch, attention_mask=batch.ne(pad_id))

forward_loop = make_adaptive_calibrate_loop(calib_data, forward_fn=my_forward)
```

### Tuning

```python
forward_loop = make_adaptive_calibrate_loop(
    calib_data,
    min_bs=1,                      # floor (always safe)
    max_bs=16,                     # ceiling (diminishing returns past 8-16 on most models)
    safety_fraction=0.85,          # keep 15% VRAM headroom
    phase2_overhead_factor=2.0,    # pessimistic factor for awq_lite search phase
    progress_every=128,            # print every N samples
)
```

---

## Benchmarks

Measured on a mini MoE (32 experts, hidden=512, intermediate=256, top_k=4) running full `NVFP4_AWQ_FULL_CFG` calibration with 16 samples × 512 seq len on a GB10. Source: `benchmarks/bench_calibration.py`.

| Batch size | Wall-clock | Speedup |
|---|---|---|
| 1 (naive) | 12.68s | 1.00× |
| 2 | 6.26s | 2.03× |
| 4 | 3.52s | 3.60× |
| 8 | 2.25s | **5.65×** |

### Real-world: SuperGemma4 26B on GB10 (97 GB VRAM)

| Configuration | CALIB_SAMPLES | SEQ_LEN | BS | Est. wall-clock |
|---|---|---|---|---|
| v1 (naive, default modelopt) | 4096 | 4096 | 1 | **~50 hours** (projected, killed at 18h) |
| v2.5 (this package + Tier 1 tuning) | 512 | 4096 | 3 (adaptive) | **~1.7 hours** |

**~30× end-to-end speedup** — the bulk comes from reducing `CALIB_SAMPLES 4096 → 512` (8×, modelopt's default) and `alpha_step 0.1 → 0.2` (1.8× on the search phase). The adaptive batching contributed a 2.5× on top (bounded by real VRAM constraints picking bs=3 rather than 8).

On smaller MoE models with seq_len in the 2-4K range, typical adaptive-batching contribution alone is 4-7×.

---

## How it works

### Probe

Runs two forwards at `bs=1` and `bs=2` through the unquantized model (modelopt hasn't set up its alpha-search buffers yet — this is the first `forward_loop(model)` invocation during the cache phase). Measures `torch.cuda.max_memory_allocated()` delta to fit a linear model:

```
activation_overhead(bs) = slope * bs + fixed
```

### Pick

Given total VRAM, baseline allocation (weights + modelopt buffers), and a user-specified `safety_fraction` (default 85%) and `phase2_overhead_factor` (default 2.0× — accounts for awq_lite's search phase holding 11 alpha-loss tensors + scaled inputs simultaneously):

```
budget = safety_fraction * total_vram
max_safe_bs = (budget - baseline - fixed) / (slope * phase2_overhead_factor)
```

Capped at `[min_bs, max_bs]`.

### Run with fallback

Runs the calibration at the picked `bs`. On any `torch.cuda.OutOfMemoryError`:

1. Free the current batch
2. Halve `bs`  (persisted across phases — won't retry the larger value next phase)
3. Retry the current batch
4. If `bs == min_bs` already, re-raise

Result: robust across model sizes and hardware without hand-tuning.

---

## Limitations

**What this package DOES NOT do:**

- **Does not fuse per-expert Python loops into grouped GEMMs.** This is the next frontier — at `bs=3` on real SuperGemma4 we still only see 47% GPU util because the MoE plugin iterates over 128 experts in a Python `for` loop. Tier 3 future work (see Roadmap).
- **Does not touch the awq_lite alpha search loop.** 11 alphas are still evaluated serially per linear. Patching modelopt to batch alphas would add another 5-11× on the search phase specifically.
- **Does not eliminate padding waste.** When `calib_data` has variable-length samples, they must be padded to a uniform length before `torch.cat`. Length-bucketing (coming in 0.2.0) reduces this.

**What you still need to do yourself:**

- Pad variable-length `calib_data` to a uniform `seq_len` (requirement of `torch.cat` at `bs>1`).
- Provide a sensible `forward_fn` if your model takes more than `input_ids`.
- Write your own per-expert plugin if modelopt doesn't have one for your MoE architecture (this package assumes the plugin is already registered).

---

## Roadmap

### 0.2.0 — length bucketing

Sort `calib_data` by length, pad within adjacent-length buckets only. Cuts padding waste from typical 30% → 5%.

### 0.3.0 — `torch.compile` on MoE plugin.forward

Test whether `torch.compile` successfully fuses per-expert Python loops through modelopt's `bind_forward_method` patching. Potential 5-20× additional speedup if compatible.

### 1.0.0 — fused grouped-GEMM MoE plugin

Ship an opt-in `FastMoEPlugin` that replaces the typical per-expert Python for-loop with a single `torch.bmm` over expert-grouped tokens (Megablocks / vLLM-style). Target 85%+ GPU utilization. This is the "true saturation" path and requires more careful numerical validation against modelopt's AWQ calibration.

### 1.1.0 — batched alpha search

Monkey-patch modelopt's `awq_lite.forward` to stack the 11 alpha scaled inputs into a single batched GEMM, split outputs, compute losses in parallel. ~11× on the search phase.

---

## Citation

If this package saves you a weekend of GPU time and you want to credit it in a paper or blog post:

```bibtex
@software{modelopt_fast_moe,
  title  = {modelopt-fast-moe: VRAM-adaptive batched calibration for nvidia-modelopt MoE quantization},
  author = {AEON-7},
  year   = {2026},
  url    = {https://github.com/AEON-7/modelopt-fast-moe},
}
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).

---

## Credits

Born from a 50-hour SuperGemma4 26B NVFP4 quantization job that had no business taking that long. Full story + benchmarks in the original [Gemma4 MoE quantization writeup](https://github.com/AEON-7/modelopt-fast-moe/blob/main/docs/gemma4-case-study.md) (coming soon).

Thanks to NVIDIA for [nvidia-modelopt](https://github.com/NVIDIA/TensorRT-Model-Optimizer), which does all the hard AWQ algorithm work — this package just feeds it data faster.

---

## ☕ Support the work

If this release has been useful, tips are deeply appreciated — they go directly toward more compute, more models, and more open releases.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <strong>₿ Bitcoin (BTC)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/btc.png" alt="BTC QR" width="200"/><br/>
      <sub><code>bc1q09xmzn00q4z3c5raene0f3pzn9d9pvawfm0py4</code></sub>
    </td>
    <td align="center" width="50%">
      <strong>Ξ Ethereum (ETH)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/eth.png" alt="ETH QR" width="200"/><br/>
      <sub><code>0x1512667F6D61454ad531d2E45C0a5d1fd82D0500</code></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>◎ Solana (SOL)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/sol.png" alt="SOL QR" width="200"/><br/>
      <sub><code>DgQsjHdAnT5PNLQTNpJdpLS3tYGpVcsHQCkpoiAKsw8t</code></sub>
    </td>
    <td align="center" width="50%">
      <strong>ⓜ Monero (XMR)</strong><br/>
      <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/xmr.png" alt="XMR QR" width="200"/><br/>
      <sub><code>836XrSKw4R76vNi3QPJ5Fa9ugcyvE2cWmKSPv3AhpTNNKvqP8v5ba9JRL4Vh7UnFNjDz3E2GXZDVVenu3rkZaNdUFhjAvgd</code></sub>
    </td>
  </tr>
</table>

> **Ethereum L2s (Base, Arbitrum, Optimism, Polygon, etc.) and EVM-compatible tokens** can be sent to the same Ethereum address.
