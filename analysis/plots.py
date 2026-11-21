"""All plots, made from results/*.jsonl files only (never from in-memory runs).

Run:  uv run python -m analysis.plots <name> [arg]   e.g.  cache_vs_naive  |  trace bimodal_bursty
Plots are written to results/plots/<name>.png.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.common import RESULTS_DIR, read_jsonl  # noqa: E402

PLOTS_DIR = RESULTS_DIR / "plots"


def load(name: str) -> pd.DataFrame:
    """Load results/<name>.jsonl into a DataFrame."""
    return pd.DataFrame(read_jsonl(RESULTS_DIR / f"{name}.jsonl"))


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure to results/plots/<name>.png and print where it went."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


def plot_cache_vs_naive() -> None:
    """Plot #1: per-step latency vs. step index, naive (no cache) vs. KV-cached loop."""
    df = load("cache_vs_naive")
    cfg = df["config"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    for loop, color in [("naive", "tab:red"), ("cached", "tab:blue")]:
        part = df[df["loop"] == loop]
        ax.plot(part["step"], part["ms"], label=loop, color=color, linewidth=1.2)
    ax.set_xlabel("decode step (step 0 = prefill)")
    ax.set_ylabel("step latency (ms)")
    ax.set_title(
        f"Per-step latency, {cfg['model']} {cfg['dtype']}, prompt {cfg['prompt_len']} tokens"
    )
    ax.set_ylim(bottom=0)
    ax.legend()
    save(fig, "cache_vs_naive")


def plot_trace(trace_name: str) -> None:
    """Step 2.3: a trace's prompt/output length histograms and its arrival timeline."""
    data = json.loads((RESULTS_DIR / "traces" / f"{trace_name}.json").read_text())
    reqs = pd.DataFrame(data["requests"])
    reqs["prompt_len"] = reqs["prompt_ids"].apply(len)

    fig, (ax_len, ax_arr) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax_len.hist(reqs["prompt_len"], bins=30, alpha=0.6, label="prompt")
    ax_len.hist(reqs["max_new_tokens"], bins=30, alpha=0.6, label="output")
    ax_len.set_xlabel("tokens")
    ax_len.set_ylabel("requests")
    ax_len.set_title("Length distribution")
    ax_len.legend()

    # Timeline: one dot per arrival (urgent ones highlighted), plus cumulative count.
    ax_arr.plot(reqs["arrival_time"], range(1, len(reqs) + 1), color="gray", linewidth=1)
    urgent = reqs[reqs["priority"] == 1]
    ax_arr.scatter(urgent["arrival_time"], urgent.index + 1, s=12, color="tab:red",
                   label="latency-sensitive", zorder=3)
    ax_arr.set_xlabel("arrival time (s)")
    ax_arr.set_ylabel("cumulative requests")
    ax_arr.set_title("Arrivals")
    ax_arr.legend()
    cfg = data["config"]
    fig.suptitle(f"Trace {trace_name}: {cfg['arrival']} arrivals, {cfg['length_mix']} lengths")
    save(fig, f"trace_{trace_name}")


PLOTS = {"cache_vs_naive": plot_cache_vs_naive, "trace": plot_trace}

if __name__ == "__main__":
    name, *args = sys.argv[1:]
    PLOTS[name](*args)
