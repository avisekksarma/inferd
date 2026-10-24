"""Shared pytest fixtures. pytest loads this file automatically for every test in tests/.

A test receives a fixture just by naming it as a parameter, e.g. `def test_x(gpt2_reference): ...`.
"""

import json
from pathlib import Path

import pytest
import torch

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def gpt2_reference() -> dict:
    """The answer key made by scripts/make_fixture.py: prompts + their reference greedy outputs.

    scope="session" -> the JSON is read once per test run and shared by every test that asks.
    """
    return json.loads((FIXTURES_DIR / "gpt2_greedy.json").read_text())


@pytest.fixture(scope="session")
def gpt2() -> tuple:
    """GPT-2 small (fp32) + tokenizer on MPS/CPU, loaded once and shared by all tests.

    Loading takes a few seconds, so doing it once per test run (not per test) keeps tests fast.
    """
    # Imported here, not at the top, so tests that don't need the model never load transformers.
    from serve.model_runner import load_model

    return load_model("gpt2", dtype=torch.float32)
