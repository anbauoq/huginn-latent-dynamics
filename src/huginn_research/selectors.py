"""Token position selectors for trajectory capture.

All functions are pure and operate on plain Python/NumPy data so they are
testable without loading Huginn.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Scope = Literal["input", "output"]
Alignment = Literal["token", "prediction"]


@dataclass
class TokenInfo:
    """Metadata for one absolute token position in a prompt+generation sequence."""

    position: int
    token_id: int
    text: str
    is_special: bool
    scope: Scope


def _cheap_periodicity(x: np.ndarray, min_lag: int = 3) -> float:
    """Max normalized autocorrelation at lag >= min_lag. 0.0 if undefined."""
    n = len(x)
    if n < min_lag + 2:
        return 0.0
    centered = x - x.mean()
    denom = float(np.dot(centered, centered))
    if denom < 1e-12:
        return 0.0
    best = 0.0
    for lag in range(min_lag, n):
        ac = float(np.dot(centered[:-lag], centered[lag:])) / denom
        best = max(best, ac)
    return max(best, 0.0)


def interesting_scores(states: np.ndarray) -> np.ndarray:
    """Cheap per-position score combining persistent late movement and periodicity.

    `states` has shape [num_steps + 1, num_positions, hidden]. This score is used
    only to pick which positions to save; it must never be reused as a
    classification metric.
    """
    diffs = np.diff(states, axis=0)
    step_norms = np.linalg.norm(diffs, axis=-1)  # [num_steps, num_positions]
    num_steps = step_norms.shape[0]
    late_start = max(num_steps - max(num_steps // 3, 1), 0)
    late_movement = step_norms[late_start:].mean(axis=0)
    periodicity = np.array([_cheap_periodicity(step_norms[:, s]) for s in range(step_norms.shape[1])])
    spread = step_norms.std(axis=0)
    return late_movement + periodicity * spread


def _parse_selector(selector: str) -> tuple[str, Optional[str]]:
    if ":" in selector:
        name, arg = selector.split(":", 1)
        return name, arg
    return selector, None


def select_positions(
    selector: str,
    tokens: list[TokenInfo],
    interesting_top_k: int = 5,
    states: Optional[np.ndarray] = None,
) -> list[int]:
    """Resolve a selector string to a sorted list of absolute token positions.

    `tokens` must be ordered by position, one entry per sequence position.
    `states` (shape [num_steps + 1, num_positions, hidden]) is required only
    for the `interesting:k` selector.
    """
    name, arg = _parse_selector(selector)

    if name == "all":
        positions = [t.position for t in tokens]
    elif name == "input":
        positions = [t.position for t in tokens if t.scope == "input"]
    elif name == "output" and arg == "first":
        output_positions = [t.position for t in tokens if t.scope == "output"]
        positions = output_positions[:1]
    elif name == "output" and arg == "last":
        output_positions = [t.position for t in tokens if t.scope == "output"]
        positions = output_positions[-1:]
    elif name == "output":
        positions = [t.position for t in tokens if t.scope == "output"]
    elif name == "numeric":
        positions = [t.position for t in tokens if any(ch.isdigit() for ch in t.text)]
    elif name == "content":
        positions = [t.position for t in tokens if not t.is_special and t.text.strip() != ""]
    elif name == "indices":
        if not arg:
            raise ValueError("indices selector requires a comma-separated list, e.g. 'indices:3,8,12'")
        valid = {t.position for t in tokens}
        requested = [int(x) for x in arg.split(",") if x.strip() != ""]
        out_of_range = [i for i in requested if i not in valid]
        if out_of_range:
            raise ValueError(f"indices selector has out-of-range positions: {out_of_range}")
        positions = requested
    elif name == "contains":
        if arg is None:
            raise ValueError("contains selector requires a substring, e.g. 'contains:42'")
        positions = [t.position for t in tokens if arg in t.text]
    elif name == "interesting":
        top_k = int(arg) if arg else interesting_top_k
        if states is None:
            raise ValueError("interesting selector requires trajectory states for scoring")
        scores = interesting_scores(states)
        order = np.argsort(-scores, kind="stable")[:top_k]
        positions = sorted(int(tokens[i].position) for i in order)
    else:
        raise ValueError(f"unknown token selector: {selector!r}")

    return sorted(set(positions))


def apply_alignment(positions: list[int], alignment: Alignment) -> tuple[list[int], list[bool]]:
    """Shift selected positions for causal 'prediction' alignment.

    Returns (aligned_positions, clamped_flags). Position 0 has no predecessor,
    so under 'prediction' alignment it is clamped to itself and flagged.
    """
    if alignment == "token":
        return list(positions), [False] * len(positions)

    aligned = []
    clamped = []
    for p in positions:
        if p == 0:
            aligned.append(0)
            clamped.append(True)
        else:
            aligned.append(p - 1)
            clamped.append(False)
    return aligned, clamped
