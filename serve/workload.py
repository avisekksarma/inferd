"""Synthetic workload traces: when requests arrive, how long they are, and which are urgent.

Every policy is compared on the SAME saved trace, so traces are generated from a seed and saved
to JSON. Pure Python (no torch).
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from serve.request import Request


@dataclass
class WorkloadConfig:
    """Everything that defines a trace. Saved alongside it, so a trace is self-describing."""

    num_requests: int = 100
    # Arrivals: "poisson" = exponential gaps at `rate` req/s.
    # "bursty" = alternating periods of `burst_rate` and `rate`, each `period_s` seconds long.
    arrival: str = "poisson"
    rate: float = 2.0
    burst_rate: float = 10.0
    period_s: float = 5.0
    # Lengths: "short", "long", or "bimodal" (long_fraction long, the rest short).
    length_mix: str = "short"
    long_fraction: float = 0.2
    short_prompt: tuple[int, int] = (16, 64)  # inclusive (min, max) token counts
    short_output: tuple[int, int] = (16, 64)
    long_prompt: tuple[int, int] = (128, 256)
    long_output: tuple[int, int] = (128, 256)
    priority_fraction: float = 0.2  # share of latency-sensitive requests (priority=1)
    vocab_size: int = 50257  # GPT-2; prompt token ids are random (content doesn't matter)


def _arrival_times(cfg: WorkloadConfig, rng: random.Random) -> list[float]:
    """Arrival timestamps (seconds). A Poisson process = independent exponential gaps."""
    times, t = [], 0.0
    for _ in range(cfg.num_requests):
        if cfg.arrival == "poisson":
            rate = cfg.rate
        elif cfg.arrival == "bursty":
            # Even-numbered periods are bursts, odd-numbered ones are calm.
            in_burst = int(t // cfg.period_s) % 2 == 0
            rate = cfg.burst_rate if in_burst else cfg.rate
        else:
            raise ValueError(f"unknown arrival pattern: {cfg.arrival}")
        t += rng.expovariate(rate)
        times.append(t)
    return times


def _is_long(cfg: WorkloadConfig, rng: random.Random) -> bool:
    """Decide whether the next request is long, according to the length mix."""
    if cfg.length_mix == "short":
        return False
    if cfg.length_mix == "long":
        return True
    if cfg.length_mix == "bimodal":
        return rng.random() < cfg.long_fraction
    raise ValueError(f"unknown length mix: {cfg.length_mix}")


def generate_trace(cfg: WorkloadConfig, seed: int) -> list[Request]:
    """Build a reproducible list of requests (same cfg + seed -> identical trace)."""
    rng = random.Random(seed)
    requests = []
    for i, arrival in enumerate(_arrival_times(cfg, rng)):
        long = _is_long(cfg, rng)
        prompt_len = rng.randint(*(cfg.long_prompt if long else cfg.short_prompt))
        output_len = rng.randint(*(cfg.long_output if long else cfg.short_output))
        requests.append(
            Request(
                id=i,
                prompt_ids=[rng.randrange(cfg.vocab_size) for _ in range(prompt_len)],
                max_new_tokens=output_len,
                arrival_time=arrival,
                priority=1 if rng.random() < cfg.priority_fraction else 0,
            )
        )
    return requests


def save_trace(requests: list[Request], cfg: WorkloadConfig, seed: int, path: Path) -> None:
    """Save a trace (config + seed + static request fields) as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"config": asdict(cfg), "seed": seed, "requests": [r.to_trace_dict() for r in requests]}
    path.write_text(json.dumps(data))


def load_trace(path: Path) -> tuple[list[Request], WorkloadConfig, int]:
    """Load a saved trace as fresh requests, plus the config and seed that produced it."""
    data = json.loads(path.read_text())
    cfg_dict = data["config"]
    # JSON turns tuples into lists; convert back so the config compares equal.
    for key in ("short_prompt", "short_output", "long_prompt", "long_output"):
        cfg_dict[key] = tuple(cfg_dict[key])
    requests = [Request.from_trace_dict(d) for d in data["requests"]]
    return requests, WorkloadConfig(**cfg_dict), data["seed"]
