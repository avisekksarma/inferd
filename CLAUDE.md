# inferd — LLM Serving Engine

An LLM inference serving engine written from scratch on top of PyTorch, with its own prefill/decode
loop. The first milestone covers the core serving problems: continuous batching, KV-cache memory
management under a fixed budget (block-based allocation), and scheduling policies (FCFS, SJF,
priority, and an adaptive policy) that trade throughput against tail latency. More components will
be added on top of this core later.

Every design decision should be simple, explicit, and explainable; the author must understand and
be able to justify each one.

**Current research question:** Under a limited KV-cache memory budget and a mixed, bursty workload, which
scheduling policy best balances throughput and tail latency, and can a policy that adapts to live
queue/memory signals beat any single fixed policy? When does it fail to?

Full step-by-step plan: `docs/PLAN.md`. Current status: `docs/PROGRESS.md`. Read both at the start of a session.

---

## How to work with me (most important section)

I delegate boilerplate, but the core logic must come from me.
Every step in `docs/PLAN.md` carries a label. Follow it strictly:

- **[AI]** — Boilerplate. You write it. Afterwards, explain each function in 1–2 lines and tell me
  how to run the tests.
- **[Pair]** — I sketch the logic (comments/pseudocode) first. If I haven't, ask me for my sketch
  before writing code. Then implement it, and point out anywhere you deviated from my sketch and why.
- **[You]** — I write the first version. **Do not write the implementation**, even if I ask casually.
  You may: explain concepts, give hints, write tests, review my code, and point out bugs with an
  explanation. If I explicitly say "I'm stuck, show me", show the smallest piece that unblocks me.

General rules:
- **I run the commands myself.** After you write code, you may run it to check it works, but then
  give me the terminal commands to run myself (scripts, tests, experiments, plots, commits) so I
  follow the flow: what to run, what to look for, and a prediction to make first.
- **One step at a time.** Do only the current step in `docs/PROGRESS.md`. Don't build ahead, don't
  add features from later phases, don't "also refactor" unrelated files.
- Before writing code for a step, state in 2–4 lines what you'll build and which files you'll touch.
- After a step: run the tests, update `docs/PROGRESS.md` (tick the step, log any decision or
  measured number), and ask me the step's checkpoint question from `docs/PLAN.md`.
- Prefer simple, readable code over clever code. I need to be able to read every line.
- If something in the plan looks wrong or there's a better approach, say so before deviating.
  Don't silently change the design.
- **Keep the docs in sync with the project.** Whenever something actually done in the project differs
  from CLAUDE.md or docs/ (versions, tools, file names, interfaces, design), update those files in
  the same step and log the change in the Decisions log of `docs/PROGRESS.md`.
- Never fake or estimate a measured result. Benchmarks come from running code.

---

## Hardware & stack

- Apple M4 Mac, PyTorch **MPS** backend (`torch.device("mps")`, CPU fallback if unavailable).
- Python 3.13, environment managed by **uv** (`pyproject.toml` + `uv.lock`; run everything via `uv run`).
  Hugging Face `transformers` for model loading only.
- Models: **GPT-2 small (124M)** to get the pipeline working; later a ~0.5–1.5B instruct model
  (small Qwen/Llama/SmolLM) for real experiments.
- pytest, matplotlib, pandas, pyyaml. No other dependencies without asking.
- **torch and transformers versions are pinned exactly** in `pyproject.toml`. The HF KV-cache API
  (`DynamicCache` internals) changes between versions; never upgrade casually.
  Current pins: torch 2.11.0, transformers 5.20.1 (v5: no legacy tuple `past_key_values`).

---

## Architecture rules (non-negotiable)

1. **Deciding is separate from executing.**
   - `serve/schedulers/` and `serve/block_manager.py` are **pure Python. They never import torch.**
     The scheduler looks at requests + memory state and returns a decision (admit / preempt / decode).
   - A **runner** executes that decision: `ModelRunner` (real model) or `SimRunner` (cost model,
     virtual clock). Both expose the same interface; the engine must not care which one it has.
2. **All Hugging Face / KV-tensor code lives in `serve/model_runner.py` and `serve/batch_state.py`.**
   Nowhere else touches `past_key_values`.
3. **Blocks are logical, tensors are physical.** The BlockManager does accounting (which request owns
   which blocks, budget enforcement). Physical KV lives in padded HF cache tensors. We do NOT
   implement PagedAttention kernels. This simplification is stated openly in the report.
4. **Never use `model.generate()`** in the engine. We write our own prefill/decode loop so we can
   intercept every step. (`generate()` is used once, only to create reference outputs for tests.)
5. **Clock abstraction.** The engine reads time only through `clock.now()`. `RealClock` uses
   `time.perf_counter`; `VirtualClock` is advanced by `SimRunner`.

Core engine loop (target shape):
```
loop:
    now = clock.now()
    move requests with arrival_time <= now into waiting
    decision = scheduler.schedule(waiting, running, block_mgr, now)
    for r in decision.preempt: runner.evict(r); block_mgr.free(r); waiting.push_front(r)
    runner.prefill(decision.admit)      # newcomer's first token -> TTFT
    runner.decode_step(running)         # one token per running request
    retire finished -> free blocks, stamp finish_time
    if nothing running or waiting: clock.advance_to(next_arrival)
```

---

## Simplifying assumptions (keep them consistent; they go in the report)

- **Fixed-length outputs:** each request generates exactly `max_new_tokens`, ignoring EOS
  (as vLLM's benchmarks do). This makes the SJF length oracle exact.
- **SJF uses an oracle** for output length.
- Prompts are synthetic token ids. Content doesn't matter for scheduling.
- **Preemption is recompute-style:** free the victim's blocks, and on readmission re-prefill
  prompt + already generated tokens.
- The memory budget is **artificial** (e.g. 200 blocks × 16 tokens), enforced by the BlockManager,
  independent of the Mac's real RAM.

---

## Conventions & gotchas

- **Timing on MPS:** always call `torch.mps.synchronize()` before reading the clock (use
  `with serve.timing.timed(device) as t:` → `t.seconds`), and warm up the model first. MPS is async; without the sync,
  timings are wrong.
- **Batched decoder-only inputs:** left padding, explicit `attention_mask`, and explicit
  `position_ids = attention_mask.cumsum(-1) - 1` (pad positions masked). Missing `position_ids` is
  the classic silent bug.
- **Correctness tests run in fp32** (fp16 numerics can occasionally flip a greedy token).
  Experiments may use fp16.
- **Correctness standard:** any batched/continuous run must produce, per request, exactly the same
  greedy tokens as that request run alone. Reference fixtures live in `tests/fixtures/`.
- **Reproducibility:** every random thing takes a seed. Workload traces are saved to JSON, and every
  policy runs on the same saved trace. Each experiment result row records its config + git commit hash.
- Results go to `results/` (gitignored) as JSONL. Plots are produced by `analysis/plots.py` from
  results files only, never from in-memory runs.
- KV bytes per token = `2 × n_layers × n_kv_heads × head_dim × bytes_per_elem`
  (n_kv_heads < n_heads for GQA models).
- Use type hints and dataclasses. Keep functions small. Write a test with every module.
- **Comments:** every function (tests included) gets a docstring saying what it does and why. Add a
  short comment above any non-obvious block of logic (what it's doing and why). Don't comment the
  obvious, e.g. simple assignments, loops or one-line helpers whose name already says it all.

---

## Repo layout

```
inferd/
├── CLAUDE.md
├── pyproject.toml
├── docs/            PLAN.md, PROGRESS.md, notes, design rationale for the adaptive policy
├── serve/
│   ├── request.py        Request dataclass + State enum
│   ├── clock.py          RealClock, VirtualClock
│   ├── timing.py         sync(), timed(device) with MPS sync, warmup()
│   ├── workload.py       trace generator (Poisson / bursty arrivals, length mixes, priorities)
│   ├── model_runner.py   prefill/decode on the real model (only HF code here + batch_state)
│   ├── batch_state.py    live batched KV cache: add / remove rows / trim padding
│   ├── sim_runner.py     cost-model runner with virtual clock
│   ├── block_manager.py  [You] logical KV block allocator
│   ├── engine.py         the main loop
│   ├── metrics.py
│   └── schedulers/       base.py, fcfs.py, sjf.py, priority.py, adaptive.py   [You]
├── experiments/     one script per experiment + YAML configs
├── analysis/        plots.py
├── scripts/         one-off utilities (make_fixture.py: the only use of model.generate())
├── tests/           + tests/fixtures/
└── results/         (gitignored)
```

## Commands

- Install / rebuild env: `uv sync`
- Run tests: `uv run pytest -q`
- Lint: `uv run ruff check .`
- Regenerate the GPT-2 reference fixture: `uv run python scripts/make_fixture.py`
- (Add experiment commands here as they're created.)
