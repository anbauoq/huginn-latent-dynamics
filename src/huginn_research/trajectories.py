"""Token metadata construction, selection, alignment, and slicing of captured states.

Pure orchestration over already-captured NumPy arrays -- no model or torch
dependency, so it is directly testable.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from huginn_research.selectors import TokenInfo, apply_alignment, select_positions


def build_token_infos(
    prompt_ids: list[int],
    generated_ids: list[int],
    decode_token: Callable[[int], str],
    special_ids: set[int],
) -> list[TokenInfo]:
    """Build one TokenInfo per absolute position across prompt + generated tokens."""
    tokens = [
        TokenInfo(position=i, token_id=tid, text=decode_token(tid), is_special=tid in special_ids, scope="input")
        for i, tid in enumerate(prompt_ids)
    ]
    offset = len(prompt_ids)
    tokens.extend(
        TokenInfo(
            position=offset + j, token_id=tid, text=decode_token(tid), is_special=tid in special_ids, scope="output"
        )
        for j, tid in enumerate(generated_ids)
    )
    return tokens


@dataclass
class TrajectorySelection:
    selected_states: np.ndarray  # [num_steps + 1, num_selected, hidden]
    selected_positions: list[int]  # positions being analyzed, sorted ascending
    aligned_positions: list[int]  # positions actually indexed into full_states
    clamped: list[bool]  # True where prediction alignment had no predecessor


def select_and_slice(
    full_states: np.ndarray,
    tokens: list[TokenInfo],
    selector: str,
    alignment: str,
    interesting_top_k: int,
) -> TrajectorySelection:
    """Resolve `selector` to token positions, apply alignment, and slice `full_states`.

    `full_states` has shape [num_steps + 1, sequence_length, hidden_size].
    """
    positions = select_positions(selector, tokens, interesting_top_k=interesting_top_k, states=full_states)
    if not positions:
        raise ValueError(f"selector {selector!r} matched no token positions")

    aligned, clamped = apply_alignment(positions, alignment)
    selected_states = full_states[:, aligned, :]
    return TrajectorySelection(
        selected_states=selected_states,
        selected_positions=positions,
        aligned_positions=aligned,
        clamped=clamped,
    )


def build_tokens_metadata(tokens: list[TokenInfo], selection: TrajectorySelection, alignment: str) -> dict[str, Any]:
    """Build the JSON-serializable token metadata saved alongside each NPZ trajectory."""
    by_position = {t.position: t for t in tokens}
    entries = []
    for pos, aligned_pos, clamped in zip(
        selection.selected_positions, selection.aligned_positions, selection.clamped
    ):
        token = by_position[pos]
        aligned_token = by_position[aligned_pos]
        entries.append(
            {
                "position": pos,
                "token_id": token.token_id,
                "token_text": token.text,
                "scope": token.scope,
                "aligned_position": aligned_pos,
                "aligned_token_id": aligned_token.token_id,
                "aligned_token_text": aligned_token.text,
                "alignment_clamped": clamped,
            }
        )
    return {"alignment": alignment, "tokens": entries}
