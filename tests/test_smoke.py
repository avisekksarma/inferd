"""Smoke test: the package imports and torch can run a tensor op on the chosen device."""

import torch

import serve


def get_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def test_serve_package_imports() -> None:
    assert serve.__name__ == "serve"


def test_tensor_op_on_device() -> None:
    device = get_device()
    x = torch.ones(8, 8, device=device)
    assert (x @ x).sum().item() == 8 * 8 * 8
