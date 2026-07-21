import numpy as np
import pytest

from huginn_research import trajectories
from huginn_research.selectors import TokenInfo


def _decode(token_id: int) -> str:
    return f"[{token_id}]"


def test_build_token_infos_positions_and_scope():
    tokens = trajectories.build_token_infos([10, 11, 12], [20, 21], _decode, special_ids=set())
    assert [t.position for t in tokens] == [0, 1, 2, 3, 4]
    assert [t.scope for t in tokens] == ["input", "input", "input", "output", "output"]
    assert tokens[3].token_id == 20


def test_build_generated_token_infos_offsets_by_input_length():
    tokens = trajectories.build_generated_token_infos([20, 21, 22], input_length=5, decode_token=_decode, special_ids=set())
    assert [t.position for t in tokens] == [5, 6, 7]
    assert all(t.scope == "output" for t in tokens)
    assert [t.token_id for t in tokens] == [20, 21, 22]


def test_select_and_slice_teacher_forced_token_alignment():
    num_steps, seq_len, hidden = 4, 6, 3
    full_states = np.arange(num_steps * seq_len * hidden, dtype=np.float32).reshape(num_steps, seq_len, hidden)
    tokens = trajectories.build_token_infos([1, 2, 3], [4, 5, 6], _decode, special_ids=set())

    selection = trajectories.select_and_slice(full_states, tokens, "output", "token", interesting_top_k=5)
    assert selection.selected_positions == [3, 4, 5]
    assert selection.aligned_positions == [3, 4, 5]
    assert selection.clamped == [False, False, False]
    assert np.array_equal(selection.selected_states, full_states[:, [3, 4, 5], :])


def test_select_and_slice_teacher_forced_prediction_alignment_clamps_first_position():
    num_steps, seq_len, hidden = 4, 6, 3
    full_states = np.arange(num_steps * seq_len * hidden, dtype=np.float32).reshape(num_steps, seq_len, hidden)
    tokens = trajectories.build_token_infos([1, 2, 3], [4, 5, 6], _decode, special_ids=set())

    selection = trajectories.select_and_slice(full_states, tokens, "indices:0,3", "prediction", interesting_top_k=5)
    assert selection.selected_positions == [0, 3]
    assert selection.aligned_positions == [0, 2]
    assert selection.clamped == [True, False]
    assert np.array_equal(selection.selected_states, full_states[:, [0, 2], :])


def test_select_and_slice_raises_on_empty_selection():
    full_states = np.zeros((4, 3, 2), dtype=np.float32)
    tokens = trajectories.build_token_infos([1, 2, 3], [], _decode, special_ids=set())
    with pytest.raises(ValueError):
        trajectories.select_and_slice(full_states, tokens, "output", "token", interesting_top_k=5)


def test_select_and_slice_generated_maps_absolute_position_to_column():
    num_steps, num_generated, hidden = 4, 3, 2
    input_length = 10
    output_states = np.arange(num_steps * num_generated * hidden, dtype=np.float32).reshape(
        num_steps, num_generated, hidden
    )
    tokens = trajectories.build_generated_token_infos([100, 101, 102], input_length, _decode, special_ids=set())

    selection = trajectories.select_and_slice_generated(output_states, tokens, "output:last", interesting_top_k=5)
    assert selection.selected_positions == [12]
    # Position 12 is generation index 2 -> column 2 of output_states.
    assert np.array_equal(selection.selected_states, output_states[:, [2], :])
    assert selection.aligned_positions == [11]
    assert selection.clamped == [False]


def test_select_and_slice_generated_indices_use_absolute_positions():
    num_steps, num_generated, hidden = 3, 4, 2
    input_length = 20
    output_states = np.arange(num_steps * num_generated * hidden, dtype=np.float32).reshape(
        num_steps, num_generated, hidden
    )
    tokens = trajectories.build_generated_token_infos([1, 2, 3, 4], input_length, _decode, special_ids=set())

    selection = trajectories.select_and_slice_generated(output_states, tokens, "indices:20,23", interesting_top_k=5)
    assert selection.selected_positions == [20, 23]
    assert np.array_equal(selection.selected_states, output_states[:, [0, 3], :])


def test_select_and_slice_generated_rejects_input_scope_indices():
    output_states = np.zeros((3, 2, 2), dtype=np.float32)
    tokens = trajectories.build_generated_token_infos([1, 2], input_length=10, decode_token=_decode, special_ids=set())
    with pytest.raises(ValueError):
        trajectories.select_and_slice_generated(output_states, tokens, "indices:0", interesting_top_k=5)


def test_build_tokens_metadata_marks_teacher_forced_capture_mode():
    tokens = trajectories.build_token_infos([1, 2], [3, 4], _decode, special_ids=set())
    selection = trajectories.select_and_slice(
        np.zeros((2, 4, 2), dtype=np.float32), tokens, "output", "token", interesting_top_k=5
    )
    meta = trajectories.build_tokens_metadata(tokens, selection, "token")
    assert meta["capture_mode"] == "teacher-forced"
    assert meta["alignment"] == "token"
    assert meta["tokens"][0]["scope"] == "output"
    assert "aligned_position" in meta["tokens"][0]


def test_build_generation_tokens_metadata_fields():
    input_length = 7
    tokens = trajectories.build_generated_token_infos([9, 8], input_length, _decode, special_ids=set())
    output_states = np.zeros((5, 2, 3), dtype=np.float32)
    selection = trajectories.select_and_slice_generated(output_states, tokens, "all", interesting_top_k=5)

    meta = trajectories.build_generation_tokens_metadata(tokens, selection, input_length)
    assert meta["capture_mode"] == "generation"
    assert meta["alignment"] == "prediction"

    first = meta["tokens"][0]
    assert first["position"] == 7
    assert first["generation_index"] == 0
    assert first["predictor_position"] == 6
    assert first["token_id"] == 9
    assert first["scope"] == "output"

    second = meta["tokens"][1]
    assert second["position"] == 8
    assert second["generation_index"] == 1
    assert second["predictor_position"] == 7
