import argparse

import pytest

from huginn_research.cli import _validate_trajectory_args, build_parser
from huginn_research.model import MODEL_NAME_DEFAULT


def test_generate_parses_defaults():
    parser = build_parser()
    args = parser.parse_args(["generate", "data.jsonl", "--output-dir", "out"])
    assert args.command == "generate"
    assert args.data == "data.jsonl"
    assert args.model == MODEL_NAME_DEFAULT
    assert args.num_steps == 64
    assert args.max_new_tokens == 256
    assert args.task == "auto"
    assert args.seed == 0
    assert args.limit is None
    assert args.func.__name__ == "cmd_generate"


def test_generate_requires_output_dir():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "data.jsonl"])


def test_trajectory_parses_token_selector_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "trajectory",
            "data.jsonl",
            "--output-dir",
            "out",
            "--tokens",
            "interesting:5",
            "--alignment",
            "prediction",
            "--interesting-top-k",
            "7",
        ]
    )
    assert args.command == "trajectory"
    assert args.tokens == "interesting:5"
    assert args.alignment == "prediction"
    assert args.interesting_top_k == 7
    assert args.func.__name__ == "cmd_trajectory"


def test_trajectory_alignment_rejects_invalid_choice():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["trajectory", "data.jsonl", "--output-dir", "out", "--alignment", "bogus"])


def test_trajectory_capture_mode_rejects_invalid_choice():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["trajectory", "data.jsonl", "--output-dir", "out", "--capture-mode", "bogus"])


def test_trajectory_defaults_to_generation_mode_and_prediction_alignment():
    parser = build_parser()
    args = parser.parse_args(["trajectory", "data.jsonl", "--output-dir", "out"])
    assert args.capture_mode == "generation"
    assert args.alignment == "prediction"
    assert args.tokens == "output"


def test_trajectory_accepts_teacher_forced_with_token_alignment():
    parser = build_parser()
    args = parser.parse_args(
        [
            "trajectory",
            "data.jsonl",
            "--output-dir",
            "out",
            "--capture-mode",
            "teacher-forced",
            "--tokens",
            "input",
            "--alignment",
            "token",
        ]
    )
    _validate_trajectory_args(args)  # should not raise


def test_validate_trajectory_args_rejects_token_alignment_with_generation_mode():
    args = argparse.Namespace(capture_mode="generation", alignment="token", tokens="output")
    with pytest.raises(SystemExit):
        _validate_trajectory_args(args)


def test_validate_trajectory_args_rejects_input_tokens_with_generation_mode():
    args = argparse.Namespace(capture_mode="generation", alignment="prediction", tokens="input")
    with pytest.raises(SystemExit):
        _validate_trajectory_args(args)


def test_validate_trajectory_args_allows_generation_mode_with_output_tokens():
    args = argparse.Namespace(capture_mode="generation", alignment="prediction", tokens="interesting:5")
    _validate_trajectory_args(args)  # should not raise


def test_metrics_parses_run_directory_and_no_plots():
    parser = build_parser()
    args = parser.parse_args(["metrics", "outputs/run1", "--no-plots"])
    assert args.command == "metrics"
    assert args.run_directory == "outputs/run1"
    assert args.no_plots is True
    assert args.func.__name__ == "cmd_metrics"


def test_metrics_no_plots_defaults_false():
    parser = build_parser()
    args = parser.parse_args(["metrics", "outputs/run1"])
    assert args.no_plots is False


def test_task_choice_validated():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "data.jsonl", "--output-dir", "out", "--task", "bogus"])


def test_no_command_is_error():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
