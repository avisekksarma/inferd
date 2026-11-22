"""Runs the real model. Together with batch_state.py, the only place that calls Hugging Face models.

- `naive_generate()`: generation with NO KV cache. Every step re-feeds the whole sequence, so
  step k recomputes all k previous tokens. The simplest correct loop; the slow reference.
- `prefill()` / `decode()` / `cached_generate()`: generation WITH a KV cache. Prefill runs the
  prompt once and stores each layer's keys/values; each decode step then feeds one token.
- `left_pad()` / `positions_from_mask()` / `batched_generate()`: several prompts in one forward
  pass (static batch), left-padded, with attention_mask + position_ids so padding is invisible.
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


def left_pad(
    prompts: list[list[int]], device: torch.device, pad_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack prompts of different lengths into one [B, T] batch, padding on the LEFT.

    Returns (input_ids [B, T], attention_mask [B, T] with 1 = real token, 0 = padding).
    Left padding makes every row END in the same column, so "the last position" (logits[:, -1])
    is each row's real last token, and new tokens are appended in the same column for all rows.
    pad_id's value doesn't matter: the mask hides those positions from attention.
    """
    max_len = max(len(p) for p in prompts)
    ids = [[pad_id] * (max_len - len(p)) + p for p in prompts]
    mask = [[0] * (max_len - len(p)) + [1] * len(p) for p in prompts]
    return torch.tensor(ids, device=device), torch.tensor(mask, device=device)


def positions_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position id of every column: count real tokens so far, minus 1 -> [B, T].

    Example row mask [0, 0, 1, 1, 1] -> cumsum [0, 0, 1, 2, 3] -> minus 1 -> [-1, -1, 0, 1, 2].
    So each request's first REAL token is position 0, exactly as if it ran alone. Without this,
    a padded prompt would start at position 2 and get different (wrong) outputs.
    Pad columns get a dummy 0 (they are masked anyway; negatives would break the embedding lookup).
    """
    positions = attention_mask.cumsum(-1) - 1
    return positions.clamp(min=0)


@torch.no_grad()
def batched_generate(
    model: torch.nn.Module, prompts: list[list[int]], max_new_tokens: int
) -> tuple[list[list[int]], list[float]]:
    """Greedy-generate for several prompts at once (static batch): one forward pass per step.

    Every request generates exactly `max_new_tokens` (fixed-length outputs). Returns
    (generated ids per request, seconds per step). Correctness bar: each request's tokens equal
    what cached_generate gives it alone.
    """
    device = next(model.parameters()).device
    batch_size = len(prompts)
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    step_times: list[float] = []

    def record(logits: torch.Tensor) -> torch.Tensor:
        """Greedy-pick one token per row, append it to each request's output, return it [B]."""
        # logits [B, 1, V] -> last position [B, V] -> best token per row [B]
        next_tokens = logits[:, -1].argmax(-1)
        for row, token in enumerate(next_tokens.tolist()):  # one GPU->CPU copy for the batch
            generated[row].append(token)
        return next_tokens

    # Step 0 -- prefill: all padded prompts in one pass. Mask + positions make padding invisible.
    input_ids, attention_mask = left_pad(prompts, device)  # both [B, T]
    with timed(device) as t:
        out = model(
            input_ids=input_ids,  # [B, T]
            attention_mask=attention_mask,  # [B, T]
            position_ids=positions_from_mask(attention_mask),  # [B, T]
            use_cache=True,
            logits_to_keep=1,
        )
        cache = out.past_key_values
        next_tokens = record(out.logits)
    step_times.append(t.seconds)

    # Steps 1..N-1 -- decode: one new token per row, added as a new (real) column of the mask.
    ones = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
    for _ in range(max_new_tokens - 1):
        with timed(device) as t:
            attention_mask = torch.cat([attention_mask, ones], dim=1)  # [B, S] -> [B, S+1]
            out = model(
                input_ids=next_tokens[:, None],  # [B] -> [B, 1]
                attention_mask=attention_mask,  # covers every cached token + the new one
                position_ids=positions_from_mask(attention_mask)[:, -1:],  # last column [B, 1]
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            next_tokens = record(out.logits)
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
    """What a KV cache holds right now."""

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
