"""Create the reference greedy outputs that every later generation loop is tested against.

This is the ONLY place in the repo that calls `model.generate()`. Each prompt is generated alone
(batch size 1, no padding), in fp32, with EOS ignored so every output has exactly
MAX_NEW_TOKENS tokens (our fixed-length-output assumption).

Run:  uv run python scripts/make_fixture.py
"""

import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"
MAX_NEW_TOKENS = 30
OUT_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "gpt2_greedy.json"

# Different lengths on purpose: batched tests later need prompts that require padding.
PROMPTS = [
    "Hello",
    "The capital of France is",
    "In a distant galaxy, a small robot woke up and",
    "def fibonacci(n):",
    "The history of the printing press begins in the fifteenth century, when a goldsmith named",
]


def get_device() -> torch.device:
    """Use the Apple GPU (MPS) if available, otherwise fall back to CPU."""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


@torch.no_grad()
def greedy_reference(model, tokenizer, prompt: str, device: torch.device) -> dict:
    """Greedy-generate exactly MAX_NEW_TOKENS tokens for one prompt; return ids as plain lists."""
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # eos_token_id=None (not min_new_tokens): min_new_tokens would *forbid* EOS and change the
    # argmax, while our own loops simply keep taking the argmax. This matches them exactly.
    out = model.generate(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        do_sample=False,  # greedy: always take the argmax token
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=None,  # never stop early, even if EOS is the argmax
        pad_token_id=tokenizer.eos_token_id,  # only silences a warning; batch of 1 has no padding
    )

    # generate() returns prompt + new tokens; slice off the prompt to keep only what was generated.
    output_ids = out[0, prompt_ids.shape[1] :].tolist()
    assert len(output_ids) == MAX_NEW_TOKENS, f"expected {MAX_NEW_TOKENS}, got {len(output_ids)}"
    return {"prompt": prompt, "prompt_ids": prompt_ids[0].tolist(), "output_ids": output_ids}


def main() -> None:
    """Load GPT-2 (fp32), generate every prompt alone, and save the answer key as JSON."""
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    model.eval()  # no dropout -> deterministic outputs

    cases = [greedy_reference(model, tokenizer, p, device) for p in PROMPTS]

    # Store the setup alongside the answers, so a future mismatch can be traced to a version change.
    fixture = {
        "model": MODEL_NAME,
        "dtype": "float32",
        "device": device.type,
        "max_new_tokens": MAX_NEW_TOKENS,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cases": cases,
    }
    OUT_PATH.write_text(json.dumps(fixture, indent=2) + "\n")

    # Human-readable summary so you can eyeball that the outputs look sane.
    print(f"wrote {len(cases)} cases to {OUT_PATH}")
    for c in cases:
        print(f"  {len(c['prompt_ids']):>2} prompt tokens | {c['prompt']!r} -> "
              f"{tokenizer.decode(c['output_ids'])!r}")


if __name__ == "__main__":
    main()
