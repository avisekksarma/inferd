# Progress

**Current step:** 3.1

Claude Code: work only on the current step. When it's done, tick it, set "Current step" to the next
one, and log any decisions or measured numbers below.

## Steps
- [x] 0.1 Environment
- [x] 0.2 Skeleton
- [x] 0.3 Timing utility
- [x] 0.4 Reference fixture
- [x] 1.1 Naive loop (no cache)
- [x] 1.2 Cached loop [You]
- [x] 1.3 Cache inspector
- [x] 1.4 Sequential baseline
- [x] 2.1 Request dataclass
- [x] 2.2 Trace generator
- [x] 2.3 Trace plots
- [ ] 3.1 Batched prefill/decode with padding
- [ ] 3.2 Microbenchmark
- [ ] 3.3 Static batching server
- [ ] 3.4 Static vs baseline
- [ ] 4.1 BatchState
- [ ] 4.2 Randomized join/leave test
- [ ] 4.3 Engine loop + scheduler/runner split
- [ ] 4.4 Static vs continuous
- [ ] 5.1 BlockManager [You]
- [ ] 5.2 Invariants + tests
- [ ] 5.3 Admission control [You]
- [ ] 5.4 Preemption [You]
- [ ] 5.5 Reserve-max vs paged
- [ ] 5.6 Rejection rule [You]
- [ ] 6.1 Cost model fit
- [ ] 6.2 SimRunner
- [ ] 6.3 Sim vs real validation
- [ ] 7.1 Scheduler base [You]
- [ ] 7.2 FCFS [You]
- [ ] 7.3 SJF [You]
- [ ] 7.4 Priority [You]
- [ ] 7.5 Hand-computed scheduler tests
- [ ] 7.6 Adaptive policy [You]
- [ ] 8.1 Metrics
- [ ] 8.2 Matrix runner
- [ ] 8.3 Plots
- [ ] 8.4 Findings [You]
- [ ] 9 Report + diagram + README

## Decisions log
<!-- date — decision — why -->
- 2026-10-16 — Python 3.13 (not 3.11 as in CLAUDE.md), env managed by uv (`pyproject.toml` + `uv.lock`) — author's choice; CLAUDE.md/PLAN.md may be out of date.
- 2026-10-16 — Pinned torch==2.11.0, transformers==5.20.1 (v5: no legacy tuple past_key_values; cache is `DynamicCache` with `cache.layers[i].keys/values`) — current release, matches current HF docs.
- 2026-10-16 — `uv sync` removed the old FastAPI packages from `.venv`; `main.py` (FastAPI hello-world) is not part of the plan and no longer runs.
- 2026-10-17 — Ruff + pytest config live in `pyproject.toml`; pytest uses `pythonpath = ["."]` so `serve` is imported straight from the repo (no build/install step) — simplest setup.
- 2026-10-17 — `git init -b main`.
- 2026-10-17 — Project renamed mini-serve → **inferd** ("inference daemon"); package folder stays `serve/`. Framed as a general LLM serving engine (first milestone: batching / KV memory / scheduling), not a mini clone of an existing engine.
- 2026-10-17 — Deleted `main.py` (unrelated FastAPI hello-world).
- 2026-10-17 — `timed(device)` takes the device explicitly (sync only on MPS, so the same code works on CPU) and yields a `Timing` object whose `.seconds` is set on exit, even if the block raises. `warmup()` runs forward passes with `use_cache=False`, so it never touches KV-cache code (that stays in model_runner/batch_state).
- 2026-10-17 — Fixture: 5 prompts of different lengths (1/5/11/7/18 tokens) so batched tests later must pad; generated one at a time (batch 1, no padding), fp32, on MPS. EOS ignored with `eos_token_id=None` (not `min_new_tokens`, which would *suppress* EOS and change the argmax; our loop just keeps going). Metadata (versions, device) stored in the JSON. Loaded in tests via the `gpt2_reference` fixture in `tests/conftest.py`.
- 2026-10-17 — Added `scripts/` to the repo layout (wasn't in CLAUDE.md).
- 2026-10-19 — Workflow: after Claude writes code (and may verify it), the author runs the terminal commands themselves (scripts, tests, experiments, plots, commits) to follow the flow. Added to CLAUDE.md.
- 2026-10-19 — Commenting convention: docstring on every function (tests too) + short what/why comments on non-obvious logic only. Added to CLAUDE.md; applied to all existing code.
- 2026-10-19 — 1.1: `naive_generate()` + `load_model()` live in `serve/model_runner.py` (rule 2: all HF model calls there). Returns (tokens, per-step seconds). Tests get the model from a session-scoped `gpt2` fixture in `tests/conftest.py` (loaded once, ~5 s). All 5 prompts match the answer key exactly.
- 2026-10-19 — `inferd-play/` = personal scratch folder inside the repo; ignored by git (`.gitignore`) and ruff (`extend-exclude`).
- 2026-10-24 — Added `README.md` (public): project summary, the text-generation pipeline (8 steps with shapes, decoding strategies, KV cache, why own loop), setup, layout. More sections to be discussed later.
- 2026-10-24 — 1.2 interface: `prefill(model, input_ids)`, `decode(model, token_ids, cache)`, `cached_generate(model, prompt_ids, max_new_tokens)` (model passed explicitly, matching `naive_generate`). Tests in `tests/test_cached_generate.py`; author wrote prefill/decode and most of cached_generate.

- 2026-11-20 — **Fast mode**: Claude implements all remaining steps (incl. [You]/[Pair]); author reviews, runs key commands, commits. Design rationale goes to `docs/design_notes.md`. ~1 week target. Recorded in CLAUDE.md.
- 2026-11-20 — `logits_to_keep=1` in naive loop and prefill: only the last position's logits are computed (full logits bloated the MPS cache to ~10 GiB → 4–6 s stalls). Prefill test now expects `[1, 1, vocab]`.
- 2026-11-20 — Warmup policy for experiments: run the whole workload once untimed, then measure (handles per-shape kernel compilation).
- 2026-11-20 — 1.4 built after Phase 2 so the baseline replays the same saved traces as every later system. Added `serve/clock.py` (needed for replay), minimal `serve/metrics.py` (extended in 8.1), `serve/baseline.py`, `experiments/common.py` (JSONL + git stamp), `experiments/make_traces.py`, `analysis/plots.py`. Traces live in `results/traces/` (regenerated from seed).
- 2026-11-21 — Added `docs/pytorch_cheatsheet.md` (ops, model call contract, measured pipeline shapes) and a shape walkthrough in each phase hand-off — the author reads code shape-first.

## Results log
<!-- date — experiment — key numbers (e.g. "baseline: X tok/s on M4, GPT-2, fp16") -->
- 2026-10-16 — env check — `torch.backends.mps.is_available()` == True; 1024×1024 matmul on MPS correct.
- 2026-11-20 — Plot #1 (GPT-2 fp32, prompt 64, 256 new tokens, warmed): naive 10→37 ms/step (total 6.0 s), cached ≈12–14 ms/step (total 3.6 s), identical tokens.
- 2026-11-20 — KV bytes/token GPT-2: 73,728 (fp32), 36,864 (fp16); measured == formula.
- 2026-11-20 — Sequential baseline: short_poisson (offered 56 tok/s) → 56.6 tok/s, TTFT p50 0.62 s / p99 1.89 s; bimodal_poisson (offered 93 tok/s, above capacity) → 68.5 tok/s, TTFT p50 10.9 s / p99 17.5 s. TPOT ≈13–14 ms.
- 2026-10-17 — MPS sync demo (20 chained 2048×2048 matmuls, warmed up): timer without sync ≈ 1.1–1.4 ms, with `timed()` ≈ 125–128 ms → unsynced timing under-reports by ~100×.
- 2026-10-17 — Fixture check: GPT-2 generation_config has no extra processors (plain greedy); EOS never appears in the 5 outputs; CPU fp32 greedy == MPS fp32 greedy for all 5 prompts.