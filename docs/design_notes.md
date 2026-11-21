# Design notes

Key design decisions, the reasoning behind them, and the evidence for them. One section per phase.

---

## Phase 1: Own generation loop

**Owning the loop.** The engine never calls `model.generate()`. Serving decisions (admit, preempt,
batch, timestamp) happen *between* forward passes, so the loop has to be ours. `generate()` is used
exactly once, in `scripts/make_fixture.py`, to produce the reference outputs.

**Correctness = token-exact match.** Every loop is tested against greedy reference outputs
(fp32, each prompt run alone). Greedy is deterministic; fp32 avoids near-tied logits flipping
(observed: top-2 candidates 0.048 vs 0.046 for one prompt). Comparing ids, not text, catches
silent bugs such as wrong position ids, which still produce fluent text.

**KV cache: prefill + decode.** Prefill runs the prompt once and stores each layer's keys/values;
decode feeds only the newest token. Result (Plot #1, GPT-2 fp32, 64-token prompt, 256 new tokens):
naive per-step latency grows linearly (≈10 → 37 ms) because it re-processes the whole sequence,
while cached decode stays roughly flat (≈12–14 ms). The price is memory:
`2 × layers × kv_heads × head_dim × bytes` per token = **72 KB/token** for GPT-2 fp32
(verified: measured cache bytes equal the formula exactly, fp32 and fp16).

**Only the last position's logits.** Both loops request `logits_to_keep=1`. Full logits are
`[1, seq, 50257]` (≈64 MB at seq 320, fp32) but only the last row predicts a token. Computing all of
them in the naive loop bloated the MPS allocator cache to ~10 GiB on a 16 GB machine and caused
4–6 s stalls; with `logits_to_keep=1` driver memory stays ≈2.2 GiB and step times are smooth.
Tokens are identical either way.

**Measurement hygiene on MPS.** MPS is asynchronous, so every timing syncs before reading the
clock (`timed()`). Unsynced timing under-reported a GPU workload by ~100×. MPS also compiles
kernels per tensor shape; the first run of a loop was ~2× slower than later runs. Experiments
therefore run the full workload once untimed before the measured run.

*Observed, not yet explained:* cached decode shows a periodic sawtooth (≈12 → 22 ms over ~40
steps). Likely the cache growing by concatenation and reallocating; not investigated further.

## Phase 2: Requests and workloads

**Request lifecycle.** WAITING → RUNNING → FINISHED, plus PREEMPTED (evicted for memory, re-prefilled
later) and REJECTED. Timestamps (`first_scheduled`, `first_token_time`, `finish_time`) give queue
time, TTFT, TPOT and end-to-end latency. Request/scheduler code is pure Python, with no torch.

**Synthetic, seeded, saved traces.** Prompt *content* doesn't affect cost, only length, so prompts
are random token ids. Arrivals are Poisson (exponential gaps) or bursty (alternating high/low-rate
periods). Length mixes: short-only, long-only, bimodal (80 % short / 20 % long). 20 % of requests are
latency-sensitive. Every policy replays the same saved trace (same seed → identical trace).

**Fixed-length outputs** (EOS ignored, as in vLLM's benchmarks), which makes the SJF oracle exact.

## Phase 1.4: Sequential baseline (the 1× reference)

One request at a time, in arrival order, with the cached loop. Capacity ≈ 1 / (time per decode step)
≈ 73 tok/s for GPT-2 fp32 on M4. **Saturation:** arrival rate λ_sat ≈ capacity / mean output
length. For bimodal (mean 70 tokens) that's ≈1 req/s, so at 1.5 req/s the queue grows without
bound: TTFT p50 11 s vs 0.6 s for the short trace below saturation.
