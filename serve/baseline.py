"""The sequential baseline -- serve requests strictly one at a time, in arrival order.

This is the 1x reference every batching/scheduling result is compared against. It replays a
trace in real time: if the next request hasn't arrived yet, the server idles until it does.
"""

import torch

from serve.clock import RealClock
from serve.model_runner import cached_generate
from serve.request import Request, State


def run_sequential(model: torch.nn.Module, requests: list[Request], clock: RealClock) -> None:
    """Serve each request alone with the KV-cached loop, stamping its timing fields in place."""
    for req in sorted(requests, key=lambda r: r.arrival_time):
        clock.advance_to(req.arrival_time)  # idle until it arrives (no-op if already late)
        req.state = State.RUNNING
        req.first_scheduled = clock.now()

        start = clock.now()
        tokens, step_times = cached_generate(model, req.prompt_ids, req.max_new_tokens)
        # step_times[0] is the prefill, which produces the first token.
        req.first_token_time = start + step_times[0]
        req.output_ids = tokens
        req.finish_time = clock.now()
        req.state = State.FINISHED
