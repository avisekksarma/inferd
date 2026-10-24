"""Tests for serve/model_runner.py: our own generation loops must match the answer key exactly."""

import pytest

from serve.model_runner import naive_generate


# parametrize: pytest runs this test 5 times, with case_index = 0, 1, 2, 3, 4 (one per prompt),
# so a failure tells you exactly which prompt broke.
@pytest.mark.parametrize("case_index", range(5))
def test_naive_generate_matches_reference(
    gpt2: tuple, gpt2_reference: dict, case_index: int
) -> None:
    """For each prompt, the no-cache loop produces exactly the reference greedy tokens."""
    model, _ = gpt2
    case = gpt2_reference["cases"][case_index]

    generated, _ = naive_generate(model, case["prompt_ids"], gpt2_reference["max_new_tokens"])

    assert generated == case["output_ids"]


def test_naive_generate_records_one_time_per_step(gpt2: tuple) -> None:
    """One positive timing per generated token, and the caller's prompt list is left unchanged."""
    model, _ = gpt2
    prompt = [464, 3139, 286, 4881, 318]  # "The capital of France is"

    generated, step_times = naive_generate(model, prompt, max_new_tokens=4)

    assert len(generated) == len(step_times) == 4
    assert all(t > 0 for t in step_times)
    assert prompt == [464, 3139, 286, 4881, 318]
