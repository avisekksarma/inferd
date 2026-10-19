"""Checks that the answer key (tests/fixtures/gpt2_greedy.json) is well formed."""

GPT2_VOCAB_SIZE = 50257


def test_fixture_metadata(gpt2_reference: dict) -> None:
    """The fixture was made with the setup our correctness tests assume (GPT-2, fp32, 30 tokens)."""
    assert gpt2_reference["model"] == "gpt2"
    assert gpt2_reference["dtype"] == "float32"
    assert gpt2_reference["max_new_tokens"] == 30


def test_fixture_cases_are_well_formed(gpt2_reference: dict) -> None:
    """5 cases, each with a non-empty prompt, exactly 30 outputs, and only valid GPT-2 token ids."""
    cases = gpt2_reference["cases"]
    assert len(cases) == 5
    for case in cases:
        assert len(case["prompt_ids"]) > 0
        assert len(case["output_ids"]) == gpt2_reference["max_new_tokens"]
        assert all(0 <= t < GPT2_VOCAB_SIZE for t in case["prompt_ids"] + case["output_ids"])


def test_prompts_have_different_lengths(gpt2_reference: dict) -> None:
    """Every prompt has a distinct length, so batched tests (step 3.1+) are forced to pad."""
    lengths = [len(c["prompt_ids"]) for c in gpt2_reference["cases"]]
    assert len(set(lengths)) == len(lengths)  # a set drops duplicates; same size => all distinct
