"""Shared helpers for experiment scripts: where results go, and how each row is stamped."""

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def git_commit() -> str:
    """Short hash of the current commit (+ '-dirty' if there are uncommitted changes)."""
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    commit = run("rev-parse", "--short", "HEAD") or "unknown"
    return commit + ("-dirty" if run("status", "--porcelain") else "")


def write_jsonl(name: str, rows: list[dict], config: dict) -> Path:
    """Write rows to results/<name>.jsonl, stamping each with the run config and git commit.

    Overwrites the file: one file = one run. Plots are made from these files only.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.jsonl"
    stamp = {"config": config, "git_commit": git_commit()}
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps({**row, **stamp}) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    """Read a results file back as a list of dicts."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
