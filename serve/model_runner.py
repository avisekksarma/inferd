"""Runs the real model. Together with batch_state.py, the only place that calls Hugging Face models.

- `naive_generate()`: generation with NO KV cache. Every step re-feeds the whole sequence, so
  step k recomputes all k previous tokens. The simplest correct loop; the slow reference.
- `prefill()` / `decode()` / `cached_generate()`: generation WITH a KV cache. Prefill runs the
  prompt once and stores each layer's keys/values; each decode step then feeds one token.
- `kv_bytes_per_token()` / `inspect_cache()`: KV-cache memory accounting (formula vs. measured).
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

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
            # logits_to_keep=1: compute vocab scores for the last position only ([1, 1, vocab]).
            # Full logits ([1, seq_len, vocab], ~64 MB at seq 320 in fp32) were never used, and
            # their ever-changing size bloated the MPS memory cache to ~10 GiB, causing multi-second
            # stalls. Same tokens either way; the layers still recompute the whole sequence.
            logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=1).logits

            # Only the LAST position's logits predict the next token; greedy = take the argmax.
            next_token = int(logits[0, -1].argmax())

        tokens.append(next_token)
        generated.append(next_token)
        step_times.append(t.seconds)

    return generated, step_times


@torch.no_grad()
def prefill(model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[torch.Tensor, DynamicCache]:
    """Run the whole prompt once with the KV cache on. Returns (logits [1, 1, vocab], cache).

    Only the last position's logits are computed (logits_to_keep=1): it's the only one that
    predicts a new token, and full-prompt logits ([1, seq, 50257]) would waste memory and time.
    """
    out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    return out.logits, out.past_key_values


@torch.no_grad()
def decode(
    model: torch.nn.Module, token_ids: torch.Tensor, cache: DynamicCache
) -> tuple[torch.Tensor, DynamicCache]:
    """Feed only the newest token(s) [1, 1] plus the cache. Returns (logits [1, 1, vocab], cache).

    The cache already holds K/V for every earlier token, so nothing before is recomputed.
    """
    out = model(input_ids=token_ids, past_key_values=cache, use_cache=True)
    return out.logits, out.past_key_values


@torch.no_grad()
def cached_generate(
    model: torch.nn.Module, prompt_ids: list[int], max_new_tokens: int
) -> tuple[list[int], list[float]]:
    """Greedy generation with a KV cache: same contract as naive_generate (tokens, step seconds).

    Step 0 is the prefill (whole prompt -> first token); every later step is one decode.
    """
    device = next(model.parameters()).device
    generated: list[int] = []
    step_times: list[float] = []

    # Step 0 -- prefill: the whole prompt in one forward pass, which also fills the cache.
    with timed(device) as t:
        logits, cache = prefill(model, torch.tensor([prompt_ids], device=device))
        next_token = int(logits[0, -1].argmax())
    generated.append(next_token)
    step_times.append(t.seconds)

    # Steps 1..N-1 -- decode: feed only the previous token; the cache supplies everything else.
    for _ in range(max_new_tokens - 1):
        with timed(device) as t:
            token_ids = torch.tensor([[next_token]], device=device)  # shape [1, 1]
            logits, cache = decode(model, token_ids, cache)
            next_token = int(logits[0, -1].argmax())
        generated.append(next_token)
        step_times.append(t.seconds)

    return generated, step_times


def kv_bytes_per_token(config, dtype: torch.dtype) -> int:
    """KV-cache bytes one token costs: 2 (K and V) x layers x kv_heads x head_dim x bytes/element.

    Works for GPT-2 and for GQA models (Llama/Qwen), where n_kv_heads < n_heads.
    """
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", None) or n_heads  # GPT-2 has no GQA
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // n_heads
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_elem


@dataclass
class CacheInfo:
    """What a KV cache holds right now (step 1.3)."""

    num_layers: int
    key_shape: tuple[int, ...]  # layer 0 keys: [batch, kv_heads, seq_len, head_dim]
    num_tokens: int  # seq_len stored per sequence
    total_bytes: int  # all layers, keys + values, as actually allocated


def inspect_cache(cache: DynamicCache) -> CacheInfo:
    """Measure a cache directly from its tensors (shapes and bytes), with no formula involved."""
    total = 0
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):
            total += tensor.numel() * tensor.element_size()
    return CacheInfo(
        num_layers=len(cache.layers),
        key_shape=tuple(cache.layers[0].keys.shape),
        num_tokens=cache.get_seq_length(),
        total_bytes=total,
    )
