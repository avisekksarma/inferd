"""Runs the real model. Together with batch_state.py, the only place that calls Hugging Face models.

Step 1.1: `naive_generate()` -- a generation loop with NO KV cache. Every step feeds the entire
sequence so far through the model, so step k recomputes all k previous tokens from scratch.
It's slow on purpose: it's the simplest correct loop, and the baseline the cached loop is
compared against.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from serve.timing import timed


def get_device() -> torch.device:
    """Use the Apple GPU (MPS) if available, otherwise fall back to CPU."""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_model(
    name: str = "gpt2",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load a causal LM and its tokenizer, move the model to `device`, and set eval mode.

    fp32 is the default because correctness tests need it (fp16 can flip near-tied argmaxes).
    """
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    model.eval()  # disable dropout -> deterministic outputs
    return model, tokenizer


@torch.no_grad()
def naive_generate(
    model: torch.nn.Module, prompt_ids: list[int], max_new_tokens: int
) -> tuple[list[int], list[float]]:
    """Greedy-generate `max_new_tokens` tokens without a KV cache.

    Returns (generated token ids, seconds taken by each step). EOS is ignored: we always
    generate exactly `max_new_tokens` (fixed-length-output assumption).
    """
    device = next(model.parameters()).device
    tokens = list(prompt_ids)  # copy, so the caller's list is never modified
    generated: list[int] = []
    step_times: list[float] = []

    for _ in range(max_new_tokens):
        with timed(device) as t:
            # Feed the WHOLE sequence every step (shape [1, len(tokens)]). use_cache=False means
            # nothing is remembered between steps -- this is exactly what makes the loop "naive".
            input_ids = torch.tensor([tokens], device=device)
            logits = model(input_ids=input_ids, use_cache=False).logits  # [1, seq_len, vocab]

            # Only the LAST position's logits predict the next token; greedy = take the argmax.
            next_token = int(logits[0, -1].argmax())

        tokens.append(next_token)
        generated.append(next_token)
        step_times.append(t.seconds)

    return generated, step_times
