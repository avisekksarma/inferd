"""Tests for batched prefill + decode with left padding (serve/model_runner.py).

The correctness bar: a request's greedy tokens must be identical whether it runs alone or in a
padded batch with other prompts of different lengths.
"""

import torch

from serve import model_runner


def test_left_pad_shapes_and_mask() -> None:
    """Shorter prompts get zeros on the LEFT; the mask marks real tokens with 1."""
    ids, mask = model_runner.left_pad([[5, 6, 7], [8]], torch.device("cpu"))

    assert ids.tolist() == [[5, 6, 7], [0, 0, 8]]
    assert mask.tolist() == [[1, 1, 1], [0, 0, 1]]


def test_positions_start_at_zero_for_every_row() -> None:
    """Each row's first real token gets position 0, whatever its padding."""
    mask = torch.tensor([[1, 1, 1], [0, 0, 1]])

    positions = model_runner.positions_from_mask(mask)

    assert positions.tolist() == [[0, 1, 2], [0, 0, 0]]


def test_batched_matches_reference_for_all_prompts(gpt2: tuple, gpt2_reference: dict) -> None:
    """All 5 fixture prompts (lengths 1..18) in ONE batch: each gets its exact solo answer."""
    model, _ = gpt2
    cases = gpt2_reference["cases"]

    generated, step_times = model_runner.batched_generate(
        model, [c["prompt_ids"] for c in cases], gpt2_reference["max_new_tokens"]
    )

    assert len(step_times) == gpt2_reference["max_new_tokens"]
    for case, tokens in zip(cases, generated, strict=True):
        assert tokens == case["output_ids"], f"mismatch for prompt {case['prompt']!r}"


def test_batch_of_one_matches_cached_generate(gpt2: tuple, gpt2_reference: dict) -> None:
    """A batch of size 1 (no padding) behaves exactly like the single-request cached loop."""
    model, _ = gpt2
    prompt = gpt2_reference["cases"][2]["prompt_ids"]

    batched, _ = model_runner.batched_generate(model, [prompt], 10)
    solo, _ = model_runner.cached_generate(model, prompt, 10)

    assert batched[0] == solo
