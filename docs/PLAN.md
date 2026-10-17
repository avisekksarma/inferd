# Implementation Plan — inferd

Labels: **[You]** I write it first · **[Pair]** I sketch, AI implements · **[AI]** boilerplate.
Every step has a **Done when** (acceptance test). Every phase ends with **Checkpoint** questions
I must be able to answer out loud before moving on.

Approximate schedule: Day 3 → Phases 0+1 · Day 4 → 2 · Day 5 → 3 · Day 6 → 4 · Day 7 → 5 (+ start 6)
· Day 8 → finish 6, then 7 · Day 9 → 8 · Day 10 → 9.

If behind schedule, cut in this order: (1) priority policy, (2) one workload mix, (3) one memory
budget level, (4) the adaptive policy. **Never cut:** the baseline, the block manager, the
join/leave correctness test (4.2), and the write-up.

---

## Phase 0 — Setup

**0.1 [AI] Environment.** Python 3.13, uv-managed env (`pyproject.toml` + `uv.lock`); torch, transformers (exact pins), pytest, ruff,
matplotlib, pandas, pyyaml.
Done when: `torch.backends.mps.is_available()` is True.

**0.2 [AI] Skeleton.** The repo layout from CLAUDE.md, `.gitignore` (results/, venv, caches), ruff
config, `pytest` passing on a smoke test, git init.

**0.3 [AI] Timing utility.** `serve/timing.py`: `timed()` context manager that calls
`torch.mps.synchronize()` before start and stop; `warmup(model, tokenizer, n=3)`.

**0.4 [AI] Reference fixture.** `scripts/make_fixture.py`: load GPT-2 small, run greedy
`model.generate` (fp32, 30 new tokens) on 5 fixed prompts, and save the token ids to
`tests/fixtures/gpt2_greedy.json`. This is the only place `generate()` is ever used.
Done when: the fixture exists and a test loads it.

---

## Phase 1 — Own generation loop

**1.1 [AI] Naive loop, no cache.** Each step feeds the whole sequence so far, takes argmax of the
last logits, and appends it. Log the time of each step.
Done when: the tokens match the fixture exactly for all 5 prompts.

**1.2 [You] Cached loop.** In `serve/model_runner.py`:
`prefill(input_ids) -> (logits, cache)` with `use_cache=True`, and
`decode(token_ids, cache) -> (logits, cache)`, which feeds only the new token. About 15 lines.
Done when: the tokens are identical to 1.1 / the fixture.
**Plot #1:** per-step latency vs. step index, cached vs. uncached.

**1.3 [Pair] Cache inspector.** Print K/V shape per layer (`[batch, kv_heads, seq, head_dim]`) and
total bytes.
Done when: measured bytes == `2 × n_layers × n_kv_heads × head_dim × bytes_per_elem × seq_len`.

**1.4 [AI] Sequential baseline runner.** Serve a list of requests one at a time (fixed output
lengths, ignore EOS). Record TTFT, total time, and tokens per request, and overall output tokens/sec.
This is the 1× reference for all later comparisons.

**Checkpoint:** Why does the uncached loop slow down every step while the cached one doesn't? What
does caching cost instead?

---

## Phase 2 — Requests & workload

**2.1 [Pair] `Request` dataclass + `State` enum** (WAITING, RUNNING, PREEMPTED, FINISHED, REJECTED).
- Fields: `id`, `prompt_ids`, `max_new_tokens`, `arrival_time`, `priority`.
- Runtime fields: `state`, `output_ids`, `first_scheduled`, `first_token_time`, `finish_time`,
  `num_preemptions`.

**2.2 [AI] `generate_trace(config, seed) -> list[Request]`.**
- Arrivals: Poisson (exponential gaps at rate λ) and bursty (alternating high/low rate periods).
- Length mixes: short-only, long-only, bimodal (e.g. 80% short / 20% long).
- A fraction of requests marked latency-sensitive (priority=1).
- Random prompt token ids. Save and load traces as JSON.
Done when: same seed → identical trace; save/load round-trips.

**2.3 [AI] Trace sanity plots.** Length histograms and an arrival timeline.

**Checkpoint:** If the system serves X tokens/s and the average output is Y tokens, which arrival
rate λ saturates it? (Experiments later use λ near this knee.)

---

## Phase 3 — Static batching

**3.1 [Pair] Batched prefill + decode with padding.** Left padding, `attention_mask`, explicit
`position_ids`.
Done when: each request's batched greedy output matches its solo output (fp32).

**3.2 [AI] Microbenchmark** (`experiments/microbench.py`). Measure:
- decode step time vs. batch size (1, 2, 4 … 64) at fixed context;
- decode step time vs. context length at fixed batch size;
- prefill time vs. prompt length.

Save the raw numbers; Phase 6 calibrates from them.
**Plot #2:** step time vs. batch size (flat, then bends).

**3.3 [AI] Static batching server.** Wait for N requests or a timeout, run the batch until the
longest finishes, repeat. Replay a trace in wall-clock time. Count wasted slots (finished sequences
still occupying the batch).

**3.4 Compare to baseline** on the same trace: throughput, p50/p99 latency. Log the numbers in
PROGRESS.md.

**Checkpoint:** Why is static batching's waste worst on the bimodal workload? Why not just use
smaller static batches?

---

## Phase 4 — Continuous batching (hardest; budget a full day)

**4.1 [Pair] `serve/batch_state.py` — `BatchState`.** Holds the batched cache, attention mask, the
row→request mapping, and per-row next position ids.
- `add(request)`: prefill the newcomer alone, left-pad the shorter of (its cache, the batch cache)
  along the sequence dimension (mask zeros), then concatenate along the batch dimension.
- `remove(rows)`: `index_select` the kept rows in every layer's K/V, the mask, and positions.
- `trim()`: drop left columns that are padding in every row.

I write the pseudocode for these three first.

**4.2 [AI] Randomized join/leave test.** 200 steps of random add/remove. Assert every request's
tokens == its solo greedy output. Nothing after this phase is trustworthy until this passes.

**4.3 [Pair] `serve/engine.py` + scheduler/runner split.** Implement the loop shape from CLAUDE.md.
Use a trivial FCFS scheduler up to `max_batch_size`, with no memory limit yet. Define a
`SchedulerDecision(admit, preempt)` dataclass.

**4.4 Compare static vs. continuous** on short-only and bimodal traces.
**Plot #3:** throughput and p50/p99 for baseline vs. static vs. continuous.

Report note: short requests joining a long batch get padded in the physical tensor. That is the
fragmentation PagedAttention removes; our logical block manager accounts memory as if paged.

**Checkpoint:** Why does admitting a request briefly slow everyone's next token? What would chunked
prefill change? (Awareness only.)

---

## Phase 5 — KV-cache block manager

**5.1 [You] `serve/block_manager.py`** (~80 lines, pure Python):
```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16): ...
    def blocks_needed(self, num_tokens: int) -> int
    def can_allocate(self, num_tokens: int) -> bool
    def allocate(self, req_id: int, num_tokens: int) -> list[int]
    def append_token(self, req_id: int) -> bool   # False = needs a new block, none free
    def free(self, req_id: int) -> None
    def utilization(self) -> float
```

**5.2 [AI] Tests + `check_invariants()`** (run every iteration in debug mode):
- no block has two owners;
- used + free == total;
- everything is free at the end.

**5.3 [You] Admission control.** Admit only if prompt + 1 token fits, with an optional small
watermark. Express the budget in MB using the KV bytes/token formula.

**5.4 [You] Preemption.** When `append_token` fails:
- pick a victim (start with the most recently admitted);
- free its blocks, evict its row from BatchState;
- push it to the front of the waiting queue;
- on readmission, re-prefill prompt + generated tokens.

Track `num_preemptions` and recomputed tokens.

**5.5 [Pair] Allocation experiment.** Reserve-max (blocks for `max_model_len` upfront) vs. paged
on-demand, at the same budget.
**Plot #4:** max concurrency, throughput, memory utilization over time.

**5.6 [You] Rejection rule.** Reject if the prompt can never fit; optionally after a max queue
wait. Count rejections.

**Checkpoint:** What does thrashing look like under too-aggressive admission, and how would I
detect it from the metrics?

---

## Phase 6 — Calibrated simulator

**6.1 [AI] Fit the cost model** from the Phase 3 microbenchmark data:
- `decode_time ≈ a + b·batch + c·total_context_tokens`
- `prefill_time ≈ d + e·prompt_tokens`

Report R². If the fit is poor, add a term; don't hand-wave it.

**6.2 [AI] `serve/sim_runner.py`.** Same interface as `ModelRunner`. It emits dummy tokens and
advances the `VirtualClock` by the predicted time.
Done when: the engine, schedulers, and block manager need zero changes to use it.

**6.3 [Pair] Validation.** Run 2–3 configs on the real and simulated engines with the same trace.
Compare throughput and p50/p99; target agreement within ~10–20%.
**Plot #9:** simulated vs. real.

---

## Phase 7 — Scheduling policies

Before starting: read theory Step 5 (FCFS/SJF/priority, tail latency, starvation).

**7.1 [You] `schedulers/base.py`.** The base class implements `schedule()` once via two hooks:
`order_waiting(waiting, now)` and `pick_victim(running, now)`.

**7.2 [You] FCFS.** Order by arrival; the victim is the latest arrival.

**7.3 [You] SJF (oracle).** Order by remaining tokens; the victim has the most remaining tokens.

**7.4 [You] Priority.** Latency-sensitive first, FCFS within class; the victim is low priority,
latest arrival. Optional aging.

**7.5 [AI] Hand-computed tests.** Tiny scenarios (3–4 requests, ~6-block budget) where I compute
the expected order on paper first.

**7.6 [You] Adaptive policy.** Only after running 7.2–7.4 on real workloads:
1. Find where each one fails (starvation, p99 blow-up, preemption thrash).
2. Write a half-page rationale in `docs/adaptive_rationale.md` before coding.
3. Choose signals from: queue length, memory utilization, oldest wait age, recent preemption rate.
4. Pick one direction, e.g. SJF + aging, switching FCFS↔SJF on queue pressure, or memory-aware
   admission that backs off when preemptions spike.

Optional: add noise to SJF's length predictions and show how it degrades.

---

## Phase 8 — Metrics & experiment matrix

**8.1 [AI] `serve/metrics.py`.**
- Per request: TTFT, TPOT, end-to-end latency, queue time, preemptions.
- Aggregates: p50/p90/p99, throughput (output tokens / makespan), rejections, memory utilization
  time series, starvation (max/mean wait by length bucket and priority).

**8.2 [AI] Matrix runner.** YAML configs covering policy × workload × memory budget × arrival rate
× 3–5 seeds. Writes JSONL rows with config + git hash. Resumable (skips finished runs).

**8.3 [AI] Plots.**
- Headline: p99 vs. arrival rate, one line per policy.
- Latency CDFs.
- Starvation by length bucket.
- Memory utilization with preemption markers.
- Plus plots #1–4 and #9 from earlier phases.

**8.4 [You] Findings.** Three one-sentence findings, at least one where the adaptive policy loses.

---

## Phase 9 — Report & diagram

Sections: motivation, design, experimental setup, results, discussion, limitations, future work.

Limitations must include:
- logical vs. physical paging;
- oracle lengths;
- fixed-length outputs;
- a single consumer device;
- simulator error bounds.

Plus an architecture diagram and a README with how to reproduce every plot.

---

## Task card template (for prompting an AI on any step)

```
CONTEXT: [what exists so far, which files]
GOAL: [one sentence]
FILES TO CREATE/EDIT: [exact paths]
INTERFACE: [signatures already decided]
CONSTRAINTS: [e.g. no model.generate(); no new deps; scheduler must not import torch]
DONE WHEN: [tests / outputs that must pass]
OUT OF SCOPE: [what not to build yet]
```

After each step: run tests → write 2 lines per function in my notes → break it on purpose and
predict the failure → commit with a message about what I learned.