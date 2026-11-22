"""Generate the named workload traces every experiment replays.

Run:  uv run python -m experiments.make_traces
Writes results/traces/<name>.json. Same names + seed always give identical traces.
"""

from experiments.common import RESULTS_DIR
from serve.workload import WorkloadConfig, generate_trace, save_trace

TRACES_DIR = RESULTS_DIR / "traces"
SEED = 0

# A small fixed set; experiment sweeps (rate, budget) build on these.
TRACES = {
    "short_poisson": WorkloadConfig(num_requests=60, arrival="poisson", rate=1.5,
                                    length_mix="short"),
    "bimodal_poisson": WorkloadConfig(num_requests=60, arrival="poisson", rate=1.5,
                                      length_mix="bimodal"),
    "bimodal_bursty": WorkloadConfig(num_requests=60, arrival="bursty", rate=0.5,
                                     burst_rate=5.0, period_s=8.0, length_mix="bimodal"),
}


def main() -> None:
    """Generate and save every trace in TRACES, printing a one-line summary of each."""
    for name, cfg in TRACES.items():
        trace = generate_trace(cfg, SEED)
        save_trace(trace, cfg, SEED, TRACES_DIR / f"{name}.json")
        out_tokens = sum(r.max_new_tokens for r in trace)
        span = trace[-1].arrival_time
        print(f"{name:>16}: {len(trace)} requests over {span:.1f} s, "
              f"{out_tokens} output tokens (avg {out_tokens / len(trace):.0f}/request), "
              f"offered load {out_tokens / span:.0f} tok/s")


if __name__ == "__main__":
    main()
