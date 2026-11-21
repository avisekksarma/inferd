"""Plot #1 data: per-step latency of the naive (no cache) loop vs. the KV-cached loop.

Each loop is run twice on the same prompt; only the second run is recorded. The first run
compiles MPS kernels for every shape the loop will see (the naive loop sees a new sequence length
every step), so the recorded run measures steady-state compute, not one-time compilation.

Run:  uv run python -m experiments.exp_cache_vs_naive
Then: uv run python -m analysis.plots cache_vs_naive
"""

import random

from experiments.common import write_jsonl
from serve.model_runner import cached_generate, load_model, naive_generate
from serve.timing import warmup

CONFIG = {"model": "gpt2", "dtype": "float32", "prompt_len": 64, "max_new_tokens": 256, "seed": 0}


def main() -> None:
    """Measure both loops on one synthetic prompt and save one row per (loop, step)."""
    model, tokenizer = load_model(CONFIG["model"])
    warmup(model, tokenizer)

    # Synthetic prompt: content doesn't affect cost, only length does.
    rng = random.Random(CONFIG["seed"])
    prompt = [rng.randrange(model.config.vocab_size) for _ in range(CONFIG["prompt_len"])]

    rows = []
    outputs = {}
    for name, generate in [("naive", naive_generate), ("cached", cached_generate)]:
        generate(model, prompt, CONFIG["max_new_tokens"])  # untimed pass: compile every shape
        tokens, step_times = generate(model, prompt, CONFIG["max_new_tokens"])
        for step, seconds in enumerate(step_times):
            rows.append({"loop": name, "step": step, "seq_len": CONFIG["prompt_len"] + step,
                         "ms": seconds * 1000})
        print(f"{name:>6}: total {sum(step_times):.2f} s, step 0 {step_times[0] * 1000:.1f} ms, "
              f"last step {step_times[-1] * 1000:.1f} ms")
        outputs[name] = tokens

    # Sanity check: the cache changes the cost, never the tokens.
    assert outputs["naive"] == outputs["cached"], "cached and naive loops disagree"
    path = write_jsonl("cache_vs_naive", rows, CONFIG)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
