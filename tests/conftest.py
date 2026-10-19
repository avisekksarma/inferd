"""Shared pytest fixtures. pytest loads this file automatically for every test in tests/.

A test receives a fixture just by naming it as a parameter, e.g. `def test_x(gpt2_reference): ...`.
"""

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def gpt2_reference() -> dict:
    """The answer key made by scripts/make_fixture.py: prompts + their reference greedy outputs.

    scope="session" -> the JSON is read once per test run and shared by every test that asks.
    """
    return json.loads((FIXTURES_DIR / "gpt2_greedy.json").read_text())
