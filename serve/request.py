"""A single inference request and its lifecycle. Pure Python (no torch): schedulers use this."""

from dataclasses import dataclass, field
from enum import Enum


class State(Enum):
    """Where a request is in its lifecycle."""

    WAITING = "waiting"  # arrived, not yet admitted (or readmission pending after preemption)
    RUNNING = "running"  # in the running batch, generating tokens
    PREEMPTED = "preempted"  # evicted to free KV memory; will be re-prefilled on readmission
    FINISHED = "finished"  # generated all max_new_tokens
    REJECTED = "rejected"  # can never fit, or waited too long


@dataclass
class Request:
    """Static fields come from the workload trace; runtime fields are filled in by the engine.

    All times are in seconds on the engine's clock (real or virtual).
    """

    id: int
    prompt_ids: list[int]
    max_new_tokens: int
    arrival_time: float
    priority: int = 0  # 1 = latency-sensitive, 0 = normal

    # Runtime fields
    state: State = State.WAITING
    output_ids: list[int] = field(default_factory=list)
    first_scheduled: float | None = None  # first time it was admitted (queueing ends here)
    first_token_time: float | None = None
    finish_time: float | None = None
    num_preemptions: int = 0
    recomputed_tokens: int = 0  # tokens re-prefilled because of preemption

    @property
    def prompt_len(self) -> int:
        """Number of prompt tokens."""
        return len(self.prompt_ids)

    @property
    def num_generated(self) -> int:
        """Number of output tokens generated so far."""
        return len(self.output_ids)

    @property
    def remaining_tokens(self) -> int:
        """Output tokens still to generate (the SJF oracle uses this)."""
        return self.max_new_tokens - self.num_generated

    @property
    def context_len(self) -> int:
        """Tokens whose KV must be held right now: prompt + generated so far."""
        return self.prompt_len + self.num_generated

    @property
    def is_done(self) -> bool:
        """True once every output token has been generated (fixed-length outputs)."""
        return self.num_generated >= self.max_new_tokens

    # Latency metrics: None until the relevant events have happened.
    @property
    def ttft(self) -> float | None:
        """Time to first token: arrival -> first output token (includes queueing)."""
        return None if self.first_token_time is None else self.first_token_time - self.arrival_time

    @property
    def e2e_latency(self) -> float | None:
        """End-to-end latency: arrival -> last output token."""
        return None if self.finish_time is None else self.finish_time - self.arrival_time

    @property
    def queue_time(self) -> float | None:
        """Time spent waiting before first admission."""
        return None if self.first_scheduled is None else self.first_scheduled - self.arrival_time

    @property
    def tpot(self) -> float | None:
        """Time per output token after the first (the decode-phase speed this request saw)."""
        if self.finish_time is None or self.first_token_time is None or self.max_new_tokens < 2:
            return None
        return (self.finish_time - self.first_token_time) / (self.max_new_tokens - 1)

    def to_trace_dict(self) -> dict:
        """Static fields only: what goes into a saved workload trace."""
        return {
            "id": self.id,
            "prompt_ids": self.prompt_ids,
            "max_new_tokens": self.max_new_tokens,
            "arrival_time": self.arrival_time,
            "priority": self.priority,
        }

    @classmethod
    def from_trace_dict(cls, d: dict) -> "Request":
        """Rebuild a fresh (never-run) request from a trace entry."""
        return cls(
            id=d["id"],
            prompt_ids=list(d["prompt_ids"]),
            max_new_tokens=d["max_new_tokens"],
            arrival_time=d["arrival_time"],
            priority=d["priority"],
        )
