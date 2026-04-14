"""VRAM-adaptive batch sizing for nvidia-modelopt MoE calibration.

Strategy:
  1. On the first phase of awq_lite (cache_mode=True), probe VRAM at bs=1 and bs=2
     to fit a linear model of per-batch memory overhead.
  2. Pick the largest batch size whose projected peak (scaled by a phase-2 safety
     factor) fits within safety_fraction * total_VRAM.
  3. Run the entire calibration at that bs.  On any OOM, halve bs and retry the
     current batch (belt-and-suspenders safety net).

This turns the naive `for ids in calib_data: model(ids)` loop into a 3-5x faster
batched one while staying within VRAM bounds on any hardware.
"""
from __future__ import annotations

import torch


def _default_forward(model, batch):
    return model(input_ids=batch)


def _probe_overhead(
    model,
    sample_template: torch.Tensor,
    forward_fn,
    verbose: bool = True,
) -> tuple[float, float, float, float]:
    """Run two probe forwards (bs=1, bs=2) and fit overhead(bs) = slope*bs + fixed.

    Returns (baseline_gb, slope_gb_per_bs, fixed_gb, total_gb).
    Raises torch.cuda.OutOfMemoryError if even bs=1 can't fit.
    """
    total_gb = torch.cuda.mem_get_info()[1] / 1e9
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_gb = torch.cuda.memory_allocated() / 1e9

    if verbose:
        print(f"  [probe] Total VRAM: {total_gb:.1f} GB | baseline alloc: {baseline_gb:.1f} GB")

    device = next(model.parameters()).device
    points: list[tuple[int, float]] = []
    for probe_bs in (1, 2):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        batch = torch.cat([sample_template] * probe_bs, dim=0).to(device)
        with torch.no_grad():
            forward_fn(model, batch)
        torch.cuda.synchronize()

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        overhead_gb = max(0.0, peak_gb - baseline_gb)
        points.append((probe_bs, overhead_gb))
        if verbose:
            print(f"  [probe] bs={probe_bs}: peak {peak_gb:.1f} GB (+{overhead_gb:.2f} GB activations)")

        del batch
        torch.cuda.empty_cache()

    (bs1, o1), (bs2, o2) = points
    slope = max(0.01, (o2 - o1) / (bs2 - bs1))
    fixed = o1 - slope * bs1
    return baseline_gb, slope, fixed, total_gb


def pick_safe_batch_size(
    baseline_gb: float,
    slope_gb_per_bs: float,
    fixed_gb: float,
    total_gb: float,
    min_bs: int = 1,
    max_bs: int = 16,
    safety_fraction: float = 0.85,
    phase2_overhead_factor: float = 2.0,
    verbose: bool = True,
) -> int:
    """From a fitted overhead model, pick the largest bs that fits in budget."""
    effective_slope = slope_gb_per_bs * phase2_overhead_factor
    budget_gb = safety_fraction * total_gb
    avail = budget_gb - baseline_gb - fixed_gb
    max_safe = int(avail / effective_slope) if effective_slope > 0 else max_bs
    max_safe = max(min_bs, min(max_bs, max_safe))

    if verbose:
        print(
            f"  [probe] overhead model: {slope_gb_per_bs:.2f} GB/bs × "
            f"{phase2_overhead_factor:.1f}× phase-2 safety = "
            f"{effective_slope:.2f} GB/bs effective"
        )
        print(
            f"  [probe] budget {budget_gb:.1f} GB "
            f"({safety_fraction*100:.0f}% of {total_gb:.0f} GB)  →  "
            f"max safe bs = {max_safe}"
        )

    return max_safe


def make_adaptive_calibrate_loop(
    calib_data: list,
    forward_fn=None,
    min_bs: int = 1,
    max_bs: int = 16,
    safety_fraction: float = 0.85,
    phase2_overhead_factor: float = 2.0,
    progress_every: int = 128,
    verbose: bool = True,
):
    """Build a forward_loop for `mtq.quantize()` that auto-sizes batches to VRAM.

    Signature matches modelopt's expectation: `fn(model) -> None`.  This function
    is called by modelopt twice during awq_full (once per awq_lite phase, plus
    awq_clip if enabled) — we probe on the first call and reuse the chosen bs.
    """
    if forward_fn is None:
        forward_fn = _default_forward

    state = {"bs": None, "phase": 0}

    def cal_loop(model):
        state["phase"] += 1
        n = len(calib_data)

        # Probe once on the first call (phase 1 = cache).  Reuse for later phases.
        if state["bs"] is None:
            if verbose:
                print(f"  [phase {state['phase']}] Probing VRAM on first sample...")
            baseline, slope, fixed, total = _probe_overhead(
                model, calib_data[0], forward_fn, verbose=verbose
            )
            state["bs"] = pick_safe_batch_size(
                baseline, slope, fixed, total,
                min_bs=min_bs, max_bs=max_bs,
                safety_fraction=safety_fraction,
                phase2_overhead_factor=phase2_overhead_factor,
                verbose=verbose,
            )

        current_bs = state["bs"]
        if verbose:
            print(f"  [phase {state['phase']}] Running calibration: {n} samples, bs={current_bs}")

        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            i = 0
            last_progress = -1
            while i < n:
                end = min(i + current_bs, n)
                batch = torch.cat(calib_data[i:end], dim=0).to(device)
                try:
                    forward_fn(model, batch)
                    i = end
                except torch.cuda.OutOfMemoryError:
                    del batch
                    torch.cuda.empty_cache()
                    new_bs = max(min_bs, current_bs // 2)
                    if new_bs == current_bs:
                        raise  # can't shrink further
                    if verbose:
                        print(
                            f"    [OOM] bs={current_bs} failed at sample {i}, "
                            f"downgrading to bs={new_bs} and retrying"
                        )
                    current_bs = new_bs
                    state["bs"] = new_bs  # persist across phases
                    continue

                # Progress print every `progress_every` samples
                if verbose and (i - last_progress >= progress_every or i == n):
                    print(f"    {i}/{n}", flush=True)
                    last_progress = i

        if verbose:
            print(f"  [phase {state['phase']}] Calibration complete (final bs={current_bs})")

    return cal_loop
