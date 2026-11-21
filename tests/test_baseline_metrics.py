"""Tests for step 1.4 (sequential baseline) and the metric helpers."""

import pytest

from serve.baseline import run_sequential
from serve.clock import RealClock
from serve.metrics import percentile, summarize
from serve.request import Request, State


def test_percentile_interpolates() -> None:
    """p50 of 1..5 is 3; p100 is the max; p25 sits between ranks."""
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 100) == 5
    assert percentile([1, 2, 3, 4, 5], 25) == pytest.approx(2.0)


def test_summarize_throughput_uses_makespan() -> None:
    """Throughput = total output tokens / (last finish - first arrival)."""
    a = Request(0, [1], 4, arrival_time=0.0, output_ids=[1, 2, 3, 4])
    b = Request(1, [1], 6, arrival_time=1.0, output_ids=[1] * 6)
    a.first_scheduled, a.first_token_time, a.finish_time = 0.0, 0.5, 2.0
    b.first_scheduled, b.first_token_time, b.finish_time = 2.0, 2.5, 5.0
    s = summarize([a, b])
    assert s["throughput_tok_s"] == pytest.approx(10 / 5.0)
    assert s["queue_time_p50"] == pytest.approx(0.5)  # (0 + 1) / 2


def test_sequential_baseline_serves_in_order_with_reference_tokens(
    gpt2: tuple, gpt2_reference: dict
) -> None:
    """All requests get the reference tokens; later ones queue behind earlier ones."""
    model, _ = gpt2
    cases = gpt2_reference["cases"][:3]
    requests = [
        Request(i, c["prompt_ids"], 30, arrival_time=0.0) for i, c in enumerate(cases)
    ]
    run_sequential(model, requests, RealClock())

    for req, case in zip(requests, cases, strict=True):
        assert req.state is State.FINISHED
        assert req.output_ids == case["output_ids"]
        assert req.first_scheduled <= req.first_token_time <= req.finish_time
    # All arrived at t=0, served one at a time: each starts after the previous one finished.
    assert requests[1].first_scheduled >= requests[0].finish_time
    assert requests[2].queue_time > requests[1].queue_time > 0
