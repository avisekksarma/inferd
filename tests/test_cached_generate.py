"""Tests for the KV-cached loop (prefill + decode) and KV-cache accounting in
serve/model_runner.py.
"""

import pytest
import torch

from serve import model_runner

FRANCE_PROMPT = [464, 3139, 286, 4881, 318]  # "The capital of France is"


# Same check as the naive loop: run once per prompt, compare against the reference outputs exactly.
@pytest.mark.parametrize("case_index", range(5))
def test_cached_generate_matches_reference(
    gpt2: tuple, gpt2_reference: dict, case_index: int
) -> None:
    """For each prompt, the cached loop produces exactly the reference greedy tokens."""
    model, _ = gpt2
    case = gpt2_reference["cases"][case_index]

    generated, _ = model_runner.cached_generate(
        model, case["prompt_ids"], gpt2_reference["max_new_tokens"]
    )

    assert generated == case["output_ids"]


def test_cached_generate_records_one_time_per_step(gpt2: tuple) -> None:
    """One positive timing per generated token (step 0 = prefill), and the prompt is unchanged."""
    model, _ = gpt2
    prompt = list(FRANCE_PROMPT)

    generated, step_times = model_runner.cached_generate(model, prompt, max_new_tokens=4)

    assert len(generated) == len(step_times) == 4
    assert all(t > 0 for t in step_times)
    assert prompt == FRANCE_PROMPT


def test_prefill_returns_last_position_logits_and_fills_cache(gpt2: tuple) -> None:
    """Prefill reads the whole prompt into the cache (5 tokens) but returns only the last logits."""
    model, _ = gpt2
    device = next(model.parameters()).device
    input_ids = torch.tensor([FRANCE_PROMPT], device=device)

    logits, cache = model_runner.prefill(model, input_ids)

    assert logits.shape == (1, 1, model.config.vocab_size)
    assert cache.get_seq_length() == 5


def test_decode_feeds_one_token_and_grows_cache_by_one(gpt2: tuple) -> None:
    """Decode sees only the new token: logits [1, 1, vocab], and the cache grows 5 -> 6."""
    model, _ = gpt2
    device = next(model.parameters()).device
    logits, cache = model_runner.prefill(model, torch.tensor([FRANCE_PROMPT], device=device))
    next_token = logits[0, -1].argmax().view(1, 1)  # shape [1, 1]: batch 1, one new token

    logits, cache = model_runner.decode(model, next_token, cache)

    assert logits.shape == (1, 1, model.config.vocab_size)
    assert cache.get_seq_length() == 6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_measured_cache_bytes_match_formula(gpt2: tuple, dtype: torch.dtype) -> None:
    """Bytes measured from the cache tensors == formula bytes/token x seq_len."""
    model, _ = gpt2
    device = next(model.parameters()).device
    model_cast = model.to(dtype)
    try:
        _, cache = model_runner.prefill(model_cast, torch.tensor([FRANCE_PROMPT], device=device))
        info = model_runner.inspect_cache(cache)
        # GPT-2 small: 12 layers; keys [batch 1, 12 heads, 5 tokens, 64 head_dim].
        assert info.num_layers == 12
        assert info.key_shape == (1, 12, 5, 64)
        per_token = model_runner.kv_bytes_per_token(model.config, dtype)
        assert info.total_bytes == per_token * len(FRANCE_PROMPT)
    finally:
        model.to(torch.float32)  # the gpt2 fixture is shared: restore fp32 for other tests
