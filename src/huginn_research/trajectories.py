"""Token metadata construction, selection, alignment, and slicing of captured states.

Pure orchestration over already-captured NumPy arrays -- no model or torch
dependency, so it is directly testable. Two capture modes feed into this
module:

* teacher-forced -- `full_states` covers every position in the prompt +
  generated sequence, so a selected token's position is shifted per
  `--alignment` and used directly as a column index.
* generation -- `output_states` covers only the generated tokens, one
  prediction-aligned trajectory per token in generation order, so a selected
  token's absolute position is translated to its generation-order column.
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


def build_generated_token_infos(
    generated_ids: list[int],
    input_length: int,
    decode_token: Callable[[int], str],
    special_ids: set[int],
) -> list[TokenInfo]:
    """Build one TokenInfo per generated token for generation-mode capture.

    Each position is the token's absolute index in the full prompt +
    generated sequence (`input_length + generation_index`), matching the
    teacher-forced convention even though no prompt tokens were captured.
    """
    return [
        TokenInfo(
            position=input_length + j, token_id=tid, text=decode_token(tid), is_special=tid in special_ids, scope="output"
        )
        for j, tid in enumerate(generated_ids)
    ]


@dataclass
class TrajectorySelection:
    selected_states: np.ndarray  # [num_steps + 1, num_selected, hidden]
    selected_positions: list[int]  # positions being analyzed, sorted ascending
    aligned_positions: list[int]  # positions whose recurrent state was actually captured/analyzed
    clamped: list[bool]  # True where prediction alignment had no predecessor


def select_and_slice(
    full_states: np.ndarray,
    tokens: list[TokenInfo],
    selector: str,
    alignment: str,
    interesting_top_k: int,
) -> TrajectorySelection:
    """Resolve `selector` to token positions, apply alignment, and slice `full_states`.

    `full_states` has shape [num_steps + 1, sequence_length, hidden_size], one
    column per absolute sequence position.
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


def select_and_slice_generated(
    output_states: np.ndarray,
    tokens: list[TokenInfo],
    selector: str,
    interesting_top_k: int,
) -> TrajectorySelection:
    """Resolve `selector` to generated-token positions and slice `output_states`.

    `output_states` has shape [num_steps + 1, num_generated_tokens, hidden_size],
    one column per generated token in generation order -- each column is
    already the prediction-aligned trajectory for that token, so there is no
    separate alignment shift to apply, only a translation from absolute
    position to generation-order column.
    """
    positions = select_positions(selector, tokens, interesting_top_k=interesting_top_k, states=output_states)
    if not positions:
        raise ValueError(f"selector {selector!r} matched no token positions")

    position_to_column = {t.position: i for i, t in enumerate(tokens)}
    columns = [position_to_column[p] for p in positions]
    selected_states = output_states[:, columns, :]
    return TrajectorySelection(
        selected_states=selected_states,
        selected_positions=positions,
        aligned_positions=[p - 1 for p in positions],
        clamped=[False] * len(positions),
    )


def build_tokens_metadata(tokens: list[TokenInfo], selection: TrajectorySelection, alignment: str) -> dict[str, Any]:
    """Build the JSON-serializable token metadata for teacher-forced capture."""
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
    return {"capture_mode": "teacher-forced", "alignment": alignment, "tokens": entries}


def build_generation_tokens_metadata(
    tokens: list[TokenInfo], selection: TrajectorySelection, input_length: int
) -> dict[str, Any]:
    """Build the JSON-serializable token metadata for generation-mode capture.

    Every entry's stored state is the recurrent trajectory at the causal
    position that predicted it -- generation mode is prediction-aligned by
    construction, so there is no separate alignment shift to record.
    """
    by_position = {t.position: t for t in tokens}
    entries = []
    for position in selection.selected_positions:
        token = by_position[position]
        entries.append(
            {
                "generation_index": position - input_length,
                "position": position,
                "predictor_position": position - 1,
                "token_id": token.token_id,
                "token_text": token.text,
                "scope": token.scope,
            }
        )
    return {"capture_mode": "generation", "alignment": "prediction", "tokens": entries}
