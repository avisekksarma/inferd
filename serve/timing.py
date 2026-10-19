"""Timing helpers that give correct wall-clock numbers on MPS.

MPS (like CUDA) is asynchronous: a call such as `model(x)` only queues work on the GPU and
returns immediately. To time real GPU work we must wait for the queue to drain
(`torch.mps.synchronize()`) both before starting the clock and before stopping it.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch


def sync(device: torch.device) -> None:
    """Block until all queued work on `device` has finished. No-op on CPU (already synchronous)."""
    if device.type == "mps":
        torch.mps.synchronize()


@dataclass
class Timing:
    """Filled in by `timed()`; `seconds` is valid after the `with` block ends."""

    seconds: float = 0.0


@contextmanager
def timed(device: torch.device) -> Iterator[Timing]:
    """Measure the wall-clock time of the `with` block, including all GPU work it queued.

    Usage:
        with timed(device) as t:
            logits = model(input_ids).logits
        print(t.seconds)
    """
    result = Timing()
    sync(device)  # don't count work queued *before* the block
    start = time.perf_counter()
    try:
        # The caller's `with` block runs here. We hand it `result` now; `seconds` is filled later.
        yield result
    finally:
        # `finally` runs even if the block raised, so a crashing block still gets a time.
        sync(device)  # wait for the block's own GPU work to actually finish
        result.seconds = time.perf_counter() - start


@torch.no_grad()
def warmup(model: torch.nn.Module, tokenizer, n: int = 3) -> None:
    """Run `n` throwaway forward passes so MPS compiles/caches its kernels before we measure.

    The first calls on MPS are much slower than later ones. Warmup covers the shapes used here;
    a benchmark that sweeps many shapes should also do one untimed run per shape.
    """
    # Read the device from the model itself, so warmup always runs where the model lives.
    device = next(model.parameters()).device

    # Tokenizer output is on CPU; move it to the model's device. Content doesn't matter, only shape.
    inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # Output is discarded: we only want the side effect (compiled kernels cached by MPS).
    # use_cache=False keeps warmup away from KV-cache code (that lives in model_runner/batch_state).
    for _ in range(n):
        model(input_ids=input_ids, use_cache=False)

    # Drain the GPU queue so leftover warmup work can't leak into the caller's first measurement.
    sync(device)
