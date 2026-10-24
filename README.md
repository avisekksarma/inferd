# inferd

An LLM inference serving engine built from scratch in PyTorch, with continuous batching,
block-based KV-cache management and pluggable scheduling policies.

inferd runs its own generation loop instead of calling a library's `generate()`, so the engine
controls every step: which requests run together, when a new request joins, and what happens
when KV-cache memory runs out.

> **Status:** early development. The single-request generation loop works and is verified
> token-for-token against Hugging Face's reference output. Batching, memory management and
> scheduling are in progress.

---

## How LLM text generation works

Everything in a serving engine is built around one loop: **autoregressive decoding**. The model
predicts one token at a time, and each prediction is appended to the input for the next one.
This section walks through that pipeline as it appears in PyTorch + Hugging Face code.

### The layers

```
┌──────────────────────────────────────────────────────────────────┐
│ inferd (serve/)      generation loop, batching, scheduling,      │
│                      KV-cache budget, timing                     │
├──────────────────────────────────────────────────────────────────┤
│ HF transformers      model architectures (GPT-2, Llama, ...),    │
│                      weight loading, tokenizers                  │
├──────────────────────────────────────────────────────────────────┤
│ PyTorch              tensors, math on CPU / GPU (MPS, CUDA),     │
│                      nn.Module                                   │
└──────────────────────────────────────────────────────────────────┘
```

Four PyTorch concepts cover most of the code:

| Concept | Meaning |
|---|---|
| **Tensor** | n-dimensional array of numbers that can live on a GPU |
| **Shape** | size of each dimension, e.g. `[1, 5]` = 1 sequence of 5 tokens |
| **Device** | where a tensor lives and is computed (`cpu`, `mps`, `cuda`) |
| **`nn.Module`** | a model: trained weights + a `forward()` that maps tensors to tensors |

### The pipeline, step by step

Shapes are traced for the prompt `"The capital of France is"` on GPT-2 small
(vocabulary of 50,257 tokens).

```
 text ──► tokenizer ──► token ids ──► model.forward ──► logits ──► pick next token ──┐
  ▲                     [1, seq]                       [1, seq, vocab]               │
  │                         ▲                                                        │
  │                         └──────────── append token, repeat ──────────────────────┘
  └──────────────────────── tokenizer.decode(ids) ◄── when done
```

**1. Pick a device.** Every tensor in a computation must be on the same device.
```python
device = torch.device("mps")            # Apple GPU; "cuda" on NVIDIA, "cpu" anywhere
```

**2. Load the tokenizer and model.** `from_pretrained` downloads the config and weights, builds the
architecture as a tree of `nn.Module`s, and fills in the trained weights.
```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)
```

**3. Move to the device and switch to inference mode.** `eval()` disables training-only
behaviour such as dropout, so outputs are deterministic.
```python
model = model.to(device)
model.eval()
```

**4. Tokenize.** Text becomes integer token ids with shape `[batch, seq_len]`. The batch
dimension is present even for a single sequence.
```python
ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"].to(device)
# [[464, 3139, 286, 4881, 318]]        shape [1, 5]
```

**5. Forward pass.** The model returns **logits**: a score for every vocabulary token at every
position. Position *i*'s scores predict token *i + 1*. `no_grad` skips the bookkeeping that
training needs.
```python
with torch.no_grad():
    logits = model(input_ids=ids).logits  # shape [1, 5, 50257]
```

**6. Pick the next token.** Only the last position predicts an unseen token. Greedy decoding
takes the highest score.
```python
next_id = logits[0, -1].argmax()        # 262 -> " the"
```

**7. Loop.** Append the token and repeat steps 5–6 until `max_new_tokens` (or an end-of-sequence
token). One forward pass yields exactly one new token.

**8. Decode.** Token ids back to text.
```python
tokenizer.decode(generated)             # " the capital of the French Republic, ..."
```

### Shapes at a glance

```
token ids        [batch, seq]            [1, 5]
hidden states    [batch, seq, hidden]    [1, 5, 768]      (inside the model)
logits           [batch, seq, vocab]     [1, 5, 50257]
logits[0, -1]    [vocab]                 [50257]          (sequence 0, last position)
argmax           scalar                  262
```

### Decoding strategies

The model is the same in every case; only step 6 changes.

| Strategy | Step 6 | Notes |
|---|---|---|
| **Greedy** | `argmax(logits)` | Deterministic. Used by inferd so outputs can be verified exactly |
| **Sampling** | `softmax(logits / T)`, then draw | Temperature `T` controls randomness |
| **Top-k / top-p** | sample only among the most likely tokens | Common default in chat applications |
| **Beam search** | keep several candidate sequences | More compute per output token |

### The KV cache

The loop above re-feeds the **entire** sequence on every step, so step *k* recomputes all
previous tokens. Each layer's attention computes a key (K) and value (V) vector per token, and
those never change once computed. A **KV cache** stores them, which splits generation into two
phases:

- **Prefill:** run the whole prompt once and fill the cache. This produces the first output token.
- **Decode:** feed only the newest token and reuse the cached K/V for everything before it.

The cache costs memory that grows with every token generated:

```
KV bytes per token = 2 (K and V) × n_layers × n_kv_heads × head_dim × bytes_per_element
```

Managing that memory across many concurrent requests is a core job of a serving engine.

### Why inferd writes its own loop

| Level | Example | Control |
|---|---|---|
| `pipeline` | `pipeline("text-generation", "gpt2")("Hello")` | almost none |
| `generate()` | `model.generate(ids, max_new_tokens=30)` | decoding settings; the loop is hidden |
| **Own loop (inferd)** | steps 5–7 written explicitly | every step: batching, admission, eviction, timing |

A serving engine has to act *between* steps: admit a new request into a running batch, preempt
one when memory is full, record time-to-first-token. That requires owning the loop.
`generate()` appears only once in this repo, in `scripts/make_fixture.py`, to produce reference
outputs that inferd's loops are tested against.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/). Developed on an Apple Silicon Mac (MPS backend);
falls back to CPU elsewhere.

```bash
uv sync                                   # create .venv with pinned dependencies
uv run pytest -q                          # run the tests
uv run ruff check .                       # lint
uv run python scripts/make_fixture.py     # regenerate the GPT-2 reference outputs
```

## Repository layout

```
serve/          engine code (model runner, timing, and soon batching and scheduling)
tests/          tests + reference fixtures
scripts/        one-off utilities
experiments/    benchmark and experiment scripts
analysis/       plotting from saved results
docs/           plan, progress log, design notes
```
