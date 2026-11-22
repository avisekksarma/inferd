"""Run the sequential baseline on a saved trace and record per-request metrics.

Run:  uv run python -m experiments.exp_baseline [trace_name]     (default: short_poisson)
Needs the traces first:  uv run python -m experiments.make_traces
"""

import sys

from experiments.common import write_jsonl
from experiments.make_traces import TRACES_DIR
from serve.baseline import run_sequential
from serve.clock import RealClock
from serve.metrics import summarize
from serve.model_runner import load_model
from serve.timing import warmup
from serve.workload import load_trace


def main(trace_name: str) -> None:
    """Replay the trace through the one-at-a-time server and save one row per request."""
    requests, wl_cfg, seed = load_trace(TRACES_DIR / f"{trace_name}.json")
    model, tokenizer = load_model("gpt2")
    warmup(model, tokenizer)

    run_sequential(model, requests, RealClock())

    rows = [
        {"id": r.id, "prompt_len": r.prompt_len, "output_len": r.num_generated,
         "priority": r.priority, "arrival": r.arrival_time, "queue_time": r.queue_time,
         "ttft": r.ttft, "tpot": r.tpot, "e2e_latency": r.e2e_latency}
        for r in requests
    ]
    config = {"system": "sequential", "model": "gpt2", "dtype": "float32",
              "trace": trace_name, "trace_seed": seed}
    path = write_jsonl(f"baseline_{trace_name}", rows, config)

    s = summarize(requests)
    print(f"throughput {s['throughput_tok_s']:.1f} tok/s | "
          f"TTFT p50 {s['ttft_p50']:.2f} s p99 {s['ttft_p99']:.2f} s | "
          f"e2e p50 {s['e2e_latency_p50']:.2f} s p99 {s['e2e_latency_p99']:.2f} s | "
          f"TPOT p50 {s['tpot_p50'] * 1000:.1f} ms")
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "short_poisson")
