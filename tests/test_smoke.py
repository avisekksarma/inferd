"""Smoke test: the package imports and torch can run a tensor op on the chosen device."""

import torch

import serve


def get_device() -> torch.device:
    """Use the Apple GPU (MPS) if available, otherwise fall back to CPU."""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def test_serve_package_imports() -> None:
    """`import serve` works, i.e. the repo is on the import path (pytest's pythonpath setting)."""
    assert serve.__name__ == "serve"


def test_tensor_op_on_device() -> None:
    """torch really computes on the device: (8x8 ones) @ (8x8 ones) sums to 8*8*8 = 512."""
    device = get_device()
    x = torch.ones(8, 8, device=device)
    assert (x @ x).sum().item() == 8 * 8 * 8
