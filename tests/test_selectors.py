import numpy as np
import pytest

from huginn_research.selectors import TokenInfo, apply_alignment, interesting_scores, select_positions


def _make_tokens() -> list[TokenInfo]:
    # 3 input tokens ("The", " answer", " is"), 3 output tokens ("42", "!", "<eos>")
    return [
        TokenInfo(0, 100, "The", False, "input"),
        TokenInfo(1, 101, " answer", False, "input"),
        TokenInfo(2, 102, " is", False, "input"),
        TokenInfo(3, 200, " 42", False, "output"),
        TokenInfo(4, 201, "!", False, "output"),
        TokenInfo(5, 202, "<eos>", True, "output"),
    ]


def test_select_all():
    tokens = _make_tokens()
    assert select_positions("all", tokens) == [0, 1, 2, 3, 4, 5]


def test_select_input_and_output():
    tokens = _make_tokens()
    assert select_positions("input", tokens) == [0, 1, 2]
    assert select_positions("output", tokens) == [3, 4, 5]


def test_select_output_first_last():
    tokens = _make_tokens()
    assert select_positions("output:first", tokens) == [3]
    assert select_positions("output:last", tokens) == [5]


def test_select_numeric():
    tokens = _make_tokens()
    assert select_positions("numeric", tokens) == [3]


def test_select_content_excludes_special():
    tokens = _make_tokens()
    assert select_positions("content", tokens) == [0, 1, 2, 3, 4]


def test_select_indices():
    tokens = _make_tokens()
    assert select_positions("indices:0,2,4", tokens) == [0, 2, 4]


def test_select_indices_out_of_range_raises():
    tokens = _make_tokens()
    with pytest.raises(ValueError):
        select_positions("indices:0,99", tokens)


def test_select_contains():
    tokens = _make_tokens()
    assert select_positions("contains:answer", tokens) == [1]


def test_select_unknown_selector_raises():
    tokens = _make_tokens()
    with pytest.raises(ValueError):
        select_positions("bogus", tokens)


def test_select_interesting_requires_states():
    tokens = _make_tokens()
    with pytest.raises(ValueError):
        select_positions("interesting:2", tokens)


def test_select_interesting_picks_top_k_by_score():
    tokens = _make_tokens()
    num_steps, hidden = 20, 4
    rng = np.random.default_rng(0)
    states = rng.normal(scale=0.01, size=(num_steps + 1, len(tokens), hidden))
    # Position 4 keeps moving substantially in the late steps; others stay flat.
    states[10:, 4, :] += np.linspace(0, 5, num_steps + 1 - 10)[:, None]

    positions = select_positions("interesting:1", tokens, states=states)
    assert positions == [4]

    scores = interesting_scores(states)
    assert scores[4] == scores.max()


def test_apply_alignment_token_is_identity():
    positions = [0, 3, 5]
    aligned, clamped = apply_alignment(positions, "token")
    assert aligned == positions
    assert clamped == [False, False, False]


def test_apply_alignment_prediction_shifts_and_clamps_first():
    positions = [0, 3, 5]
    aligned, clamped = apply_alignment(positions, "prediction")
    assert aligned == [0, 2, 4]
    assert clamped == [True, False, False]
