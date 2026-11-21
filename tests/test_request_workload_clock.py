"""Tests for Phase 2: Request lifecycle fields, trace generation/saving, and the clocks."""

import pytest

from serve.clock import RealClock, VirtualClock
from serve.request import Request, State
from serve.workload import WorkloadConfig, generate_trace, load_trace, save_trace


def test_request_derived_fields() -> None:
    """Remaining tokens, context length and latencies follow from the raw fields."""
    r = Request(id=0, prompt_ids=[1, 2, 3], max_new_tokens=4, arrival_time=1.0)
    assert r.state is State.WAITING
    r.output_ids = [7, 8]
    assert (r.remaining_tokens, r.context_len, r.is_done) == (2, 5, False)
    assert r.ttft is None
    r.first_scheduled, r.first_token_time, r.finish_time = 1.5, 2.0, 5.0
    r.output_ids = [7, 8, 9, 10]
    assert r.is_done
    assert (r.queue_time, r.ttft, r.e2e_latency) == (0.5, 1.0, 4.0)
    assert r.tpot == pytest.approx(1.0)  # 3 s for the 3 tokens after the first


def test_same_seed_gives_identical_trace() -> None:
    """Reproducibility: same config + seed -> exactly the same requests."""
    cfg = WorkloadConfig(num_requests=50, length_mix="bimodal", arrival="bursty")
    a = [r.to_trace_dict() for r in generate_trace(cfg, seed=1)]
    b = [r.to_trace_dict() for r in generate_trace(cfg, seed=1)]
    c = [r.to_trace_dict() for r in generate_trace(cfg, seed=2)]
    assert a == b
    assert a != c


def test_trace_respects_config() -> None:
    """Arrivals increase, lengths stay in range, and the bimodal mix contains both kinds."""
    cfg = WorkloadConfig(num_requests=500, length_mix="bimodal", long_fraction=0.2)
    trace = generate_trace(cfg, seed=0)
    arrivals = [r.arrival_time for r in trace]
    assert arrivals == sorted(arrivals)
    long = [r for r in trace if r.prompt_len >= cfg.long_prompt[0]]
    assert 0.1 < len(long) / len(trace) < 0.3
    for r in trace:
        assert cfg.short_prompt[0] <= r.prompt_len <= cfg.long_prompt[1]
        assert cfg.short_output[0] <= r.max_new_tokens <= cfg.long_output[1]
    # Mean arrival rate of a Poisson process should be close to the configured rate.
    assert abs(len(trace) / arrivals[-1] - cfg.rate) / cfg.rate < 0.2


def test_save_load_round_trip(tmp_path) -> None:
    """A saved trace loads back to the same requests, config and seed."""
    cfg = WorkloadConfig(num_requests=20, arrival="bursty")
    trace = generate_trace(cfg, seed=3)
    path = tmp_path / "trace.json"  # tmp_path: a fresh temporary folder pytest provides
    save_trace(trace, cfg, seed=3, path=path)
    loaded, cfg2, seed2 = load_trace(path)
    assert [r.to_trace_dict() for r in loaded] == [r.to_trace_dict() for r in trace]
    assert (cfg2, seed2) == (cfg, 3)


def test_virtual_clock_only_moves_forward() -> None:
    """VirtualClock advances by steps and jumps, never backwards."""
    clock = VirtualClock()
    clock.advance(0.5)
    clock.advance_to(2.0)
    clock.advance_to(1.0)  # in the past: ignored
    assert clock.now() == 2.0
    with pytest.raises(ValueError):
        clock.advance(-1)


def test_real_clock_advance_to_waits() -> None:
    """RealClock.advance_to sleeps until the target time."""
    clock = RealClock()
    clock.advance_to(0.05)
    assert clock.now() >= 0.05
