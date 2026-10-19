# Progress

**Current step:** 1.1

Claude Code: work only on the current step. When it's done, tick it, set "Current step" to the next
one, and log any decisions or measured numbers below.

## Steps
- [x] 0.1 Environment
- [x] 0.2 Skeleton
- [x] 0.3 Timing utility
- [x] 0.4 Reference fixture
- [ ] 1.1 Naive loop (no cache)
- [ ] 1.2 Cached loop [You]
- [ ] 1.3 Cache inspector
- [ ] 1.4 Sequential baseline
- [ ] 2.1 Request dataclass
- [ ] 2.2 Trace generator
- [ ] 2.3 Trace plots
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

## Results log
<!-- date — experiment — key numbers (e.g. "baseline: X tok/s on M4, GPT-2, fp16") -->
- 2026-10-16 — env check — `torch.backends.mps.is_available()` == True; 1024×1024 matmul on MPS correct.
- 2026-10-17 — MPS sync demo (20 chained 2048×2048 matmuls, warmed up): timer without sync ≈ 1.1–1.4 ms, with `timed()` ≈ 125–128 ms → unsynced timing under-reports by ~100×.
- 2026-10-17 — Fixture check: GPT-2 generation_config has no extra processors (plain greedy); EOS never appears in the 5 outputs; CPU fp32 greedy == MPS fp32 greedy for all 5 prompts.