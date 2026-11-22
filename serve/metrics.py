"""Latency/throughput summaries over finished requests. Pure Python (no torch).

Covers the core latency metrics; starvation and memory metrics are added with the scheduler
experiments.
"""

from serve.request import Request


def percentile(values: list[float], p: float) -> float:
    """p-th percentile (0-100) with linear interpolation between the two closest ranks."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(requests: list[Request]) -> dict:
    """Aggregate metrics over a run: throughput and p50/p90/p99 of TTFT, TPOT, end-to-end, queueing.

    Throughput = output tokens / makespan, where makespan runs from the first arrival to the
    last finish (so idle gaps in the trace count against it, as they would for a real server).
    """
    done = [r for r in requests if r.finish_time is not None]
    summary: dict = {"num_finished": len(done), "num_requests": len(requests)}
    if not done:
        return summary
    makespan = max(r.finish_time for r in done) - min(r.arrival_time for r in requests)
    total_tokens = sum(r.num_generated for r in done)
    summary["makespan_s"] = makespan
    summary["throughput_tok_s"] = total_tokens / makespan
    for metric in ("ttft", "tpot", "e2e_latency", "queue_time"):
        values = [getattr(r, metric) for r in done if getattr(r, metric) is not None]
        for p in (50, 90, 99):
            summary[f"{metric}_p{p}"] = percentile(values, p)
    summary["preemptions"] = sum(r.num_preemptions for r in requests)
    return summary
