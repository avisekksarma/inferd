# PyTorch / Hugging Face cheat sheet for inferd

Inference code is mostly **shape plumbing**. Every line answers one question: *I have shape A, I
need shape B, which op gets me there?* This page lists the ops inferd uses, the model's call
contract, and the shape of every value in the pipeline (GPT-2 numbers, measured).

Run `PYTHONPATH=. uv run python inferd-play/shapes_walkthrough.py` to see every shape printed live.

## Letters used for dimensions

| Letter | Meaning | GPT-2 value |
|---|---|---|
| `B` | batch: how many sequences at once | 1 for now; 2+ from Phase 3 |
| `T` | tokens fed in this call | prompt length on prefill, 1 on decode |
| `S` | tokens stored in the cache so far | grows by 1 every decode |
| `V` | vocabulary size | 50257 |
| `H` | attention heads (KV heads for GQA models) | 12 |
| `D` | head dimension | 64 |
| `L` | layers (transformer blocks) | 12 |

## The pipeline, one shape per step

```
"The capital of France is"            str
  tokenizer.encode(text)           -> [464, 3139, 286, 4881, 318]        list[int], 5 ids
  torch.tensor([ids], device=dev)  -> input_ids      [B=1, T=5]          int64
  model(input_ids, use_cache=True, logits_to_keep=1)
                                   -> out.logits     [B, 1, V]           float32
                                   -> out.past_key_values: DynamicCache
                                        L=12 layers, each .keys/.values [B, H, S=5, D]
  logits[0, -1]                    -> [V]            scores of batch row 0, last position
  .argmax()                        -> []             0-dim tensor (the token id, on the GPU)
  .item()                          -> 262            plain Python int  (" the")
  torch.tensor([[262]], device=dev)-> [1, 1]         the input for the next decode
  model(token, past_key_values=cache, use_cache=True)
                                   -> logits [1, 1, V]; cache keys now [1, 12, 6, 64]
```

The loop just repeats the last three lines. On decode only `S` changes; every other shape stays
the same.

## The model's call contract (treat it as an API endpoint)

`model(...)` is a function call with keyword arguments. It returns an object whose fields you read
like a JSON response.

| Argument | Shape | Meaning |
|---|---|---|
| `input_ids` | `[B, T]` int64 | the tokens to process in this call |
| `past_key_values` | `DynamicCache` | K/V of earlier tokens (decode only) |
| `use_cache=True` | | build or extend the cache and return it |
| `logits_to_keep=1` | | compute scores for the last position only |
| `attention_mask` | `[B, S+T]`, 1 = real, 0 = pad | Phase 3: tells attention to ignore padding |
| `position_ids` | `[B, T]` | Phase 3: each token's position, skipping pads |

| Response field | Shape | Meaning |
|---|---|---|
| `out.logits` | `[B, T or 1, V]` | one score per vocabulary word, per position |
| `out.past_key_values` | `DynamicCache` | the cache to pass into the next call |

Where these names come from: the model's `forward()` signature in the HF docs or source, and
`list(out.keys())` in a REPL. Nobody memorises them; you look them up once, the same way you would
read an API reference.

## The ~15 ops that cover inferd

| Op | Example | Shape effect |
|---|---|---|
| `torch.tensor(x, device=d)` | `torch.tensor([[tok]], device=d)` | Python list to tensor; each `[` adds a dimension |
| `.to(device)` | `model.to("mps")` | moves data to the GPU; shape unchanged |
| `.shape` | `logits.shape` | `torch.Size([1, 1, 50257])`; **print this whenever you're lost** |
| index `x[i, j]` | `logits[0, -1]` | each integer index removes a dimension: `[1,1,V]` to `[V]` |
| slice `x[:, j]` | `logits[:, -1]` | `:` keeps that dimension: `[B,1,V]` to `[B,V]` |
| `.argmax(dim)` | `logits[:, -1].argmax(-1)` | removes `dim`: `[B,V]` to `[B]` (one token per row) |
| `.item()` / `.tolist()` | `best.item()` | GPU tensor to Python int / list (forces a sync) |
| `.softmax(-1)` | `(logits / T).softmax(-1)` | same shape; scores become probabilities |
| `torch.multinomial(p, 1)` | sampling | `[V]` to `[1]`: draws one id at random |
| `torch.cat([a, b], dim)` | join along `dim` | other dimensions must match |
| `torch.stack(list)` | list of `[T]` | to `[N, T]`: adds a new dimension |
| `x.numel()`, `x.element_size()` | memory math | element count, bytes per element |
| `torch.no_grad()` | decorator / `with` | inference only: no gradient bookkeeping |
| `torch.mps.synchronize()` | in `timed()` | waits for the GPU (MPS is async) |
| `x.cumsum(-1)` | `mask.cumsum(-1) - 1` | Phase 3: position ids from an attention mask |

## How to figure out the next line yourself

1. **Goal.** What do I want? For example, "the next token for each running request."
2. **Have.** What do I hold, and what shape? Print `.shape`.
3. **Need.** What shape does the next call want? Check the call contract above.
4. **Bridge.** Which op turns *have* into *need*? Check the op table.

Debugging is the same loop: when something breaks, print the shape just before the failing line.
Nine times out of ten, a dimension is missing (`[5]` where `[1, 5]` was needed) or an extra one is
present.
