"""Tests for serve/timing.py: timed() measures real elapsed time; warmup() runs the model right."""

import time

import pytest
import torch

from serve.timing import timed, warmup


def get_device() -> torch.device:
    """Use the Apple GPU (MPS) if available, otherwise fall back to CPU."""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def test_timed_measures_elapsed_time() -> None:
    """A 50 ms sleep is measured as at least 50 ms (upper bound is loose, to avoid flaky tests)."""
    with timed(torch.device("cpu")) as t:
        time.sleep(0.05)
    assert 0.05 <= t.seconds < 0.5


def test_timed_includes_gpu_work() -> None:
    """Timing a block of GPU matmuls works and gives a positive duration."""
    device = get_device()
    x = torch.randn(1024, 1024, device=device)
    with timed(device) as t:
        for _ in range(10):
            x = x @ x / 1024  # divide to keep values from blowing up to inf
    assert t.seconds > 0


def test_timed_records_time_even_if_block_raises() -> None:
    """If the timed block crashes, the error still propagates AND a time is still recorded."""
    # pytest.raises: this test *expects* ValueError; it fails if the error does not happen.
    with pytest.raises(ValueError), timed(torch.device("cpu")) as t:
        time.sleep(0.01)
        raise ValueError("boom")
    assert t.seconds >= 0.01


class FakeTokenizer:
    """Stands in for a real tokenizer: always returns the same 4 token ids."""

    def __call__(self, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        """Mimic `tokenizer(text, return_tensors="pt")`, ignoring the actual text."""
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}


class FakeModel(torch.nn.Module):
    """A tiny model that counts forward calls and records whether gradients were enabled.

    Lets us test warmup() without downloading GPT-2.
    """

    def __init__(self) -> None:
        """One small embedding layer: just enough to have parameters on a device."""
        super().__init__()
        self.embed = torch.nn.Embedding(10, 4)
        self.calls = 0
        self.grad_enabled: list[bool] = []

    def forward(self, input_ids: torch.Tensor, use_cache: bool) -> torch.Tensor:
        """Record this call, then do a trivial computation."""
        self.calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.embed(input_ids)


def test_warmup_runs_n_forward_passes_without_grad() -> None:
    """warmup(n=3) calls the model exactly 3 times, each with gradient tracking off."""
    model = FakeModel().to(get_device())
    warmup(model, FakeTokenizer(), n=3)
    assert model.calls == 3
    assert model.grad_enabled == [False, False, False]
