"""Command-line entry point: huginn-research generate|trajectory|metrics."""

import argparse
import os
from typing import Optional

import numpy as np
from tqdm import tqdm

from huginn_research import answers, datasets, metrics, plotting, storage, trajectories
from huginn_research.model import MODEL_NAME_DEFAULT, HuginnAdapter


def _add_common_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data", help="Path to a JSONL file of benchmark questions")
    parser.add_argument("--model", default=MODEL_NAME_DEFAULT, help="HF model id (default: %(default)s)")
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (default: cuda if available, else cpu)")
    parser.add_argument("--num-steps", type=int, default=64, help="Recurrent steps passed directly to the model")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N examples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", choices=["auto", "numeric", "multiple-choice"], default="auto")
    parser.add_argument("--system-prompt", default=datasets.DEFAULT_SYSTEM_PROMPT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="huginn-research", description="Run and analyze Huginn-0125 recurrent inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Run inference on JSONL questions and check correctness.")
    _add_common_inference_args(generate_parser)
    generate_parser.set_defaults(func=cmd_generate)

    trajectory_parser = subparsers.add_parser("trajectory", help="Run inference and capture recurrent latent trajectories.")
    _add_common_inference_args(trajectory_parser)
    trajectory_parser.add_argument(
        "--tokens",
        default="output",
        help=(
            "Token selector: all|input|output|output:first|output:last|numeric|content|"
            "indices:3,8,12|contains:substring|interesting:5 (default: %(default)s)"
        ),
    )
    trajectory_parser.add_argument("--alignment", choices=["token", "prediction"], default="token")
    trajectory_parser.add_argument("--interesting-top-k", type=int, default=5)
    trajectory_parser.set_defaults(func=cmd_trajectory)

    metrics_parser = subparsers.add_parser("metrics", help="Compute trajectory metrics and classify saved tokens.")
    metrics_parser.add_argument("run_directory", help="A directory previously produced by the `trajectory` command")
    metrics_parser.add_argument("--no-plots", action="store_true", help="Skip generating figures")
    metrics_parser.set_defaults(func=cmd_metrics)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def _run_config(args: argparse.Namespace, command: str) -> dict:
    return {
        "command": command,
        "model": args.model,
        "device": args.device,
        "num_steps": args.num_steps,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "seed": args.seed,
        "task": args.task,
        "system_prompt": args.system_prompt,
        "data": args.data,
        "limit": args.limit,
    }


def _generate_example(adapter: HuginnAdapter, example: datasets.Example, args: argparse.Namespace):
    user_text = datasets.build_prompt(example.question)
    prompt_ids = adapter.build_prompt_ids(user_text, args.system_prompt)
    result = adapter.generate(prompt_ids, num_steps=args.num_steps, max_new_tokens=args.max_new_tokens, seed=args.seed)
    extracted = datasets.extract_answer(example.task, result.generated_text, example.options)
    correct = answers.check_correct(example.task, extracted, example.expected_answer)
    record = {
        "id": example.id,
        "question": example.question,
        "expected_answer": example.expected_answer,
        "generated_text": result.generated_text,
        "extracted_answer": extracted,
        "correct": correct,
        "prompt_token_count": len(result.prompt_ids),
        "generated_token_count": len(result.generated_ids),
    }
    return record, result


def _tally(record: dict, counts: dict) -> None:
    counts["total"] += 1
    if record["correct"]:
        counts["correct"] += 1
    if record["extracted_answer"] == "no_final_answer":
        counts["missing"] += 1


def _build_summary(counts: dict, args: argparse.Namespace, adapter: HuginnAdapter) -> dict:
    total = counts["total"]
    return {
        "example_count": total,
        "correct_count": counts["correct"],
        "accuracy": counts["correct"] / total if total else 0.0,
        "missing_answer_count": counts["missing"],
        "model_settings": {"model": args.model, "device": adapter.device, "dtype": str(adapter.dtype)},
        "runtime_settings": {
            "num_steps": args.num_steps,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "seed": args.seed,
            "task": args.task,
        },
    }


def _require_data_file(path: str) -> None:
    if not os.path.isfile(path):
        raise SystemExit(f"JSONL data file not found: {path}")


def cmd_generate(args: argparse.Namespace) -> None:
    _require_data_file(args.data)
    examples = datasets.load_examples(args.data, args.task, args.limit)
    paths = storage.make_run_paths(args.output_dir, with_trajectories=False)
    storage.save_json(paths.run_json, _run_config(args, "generate"))

    completed_ids = storage.load_completed_ids(paths.predictions_jsonl)
    counts = {"total": 0, "correct": 0, "missing": 0}
    for record in storage.read_jsonl(paths.predictions_jsonl):
        _tally(record, counts)

    adapter = HuginnAdapter(args.model, args.device)

    for example in tqdm(examples, desc="generate"):
        if example.id in completed_ids:
            continue
        record, _ = _generate_example(adapter, example, args)
        storage.append_jsonl(paths.predictions_jsonl, record)
        _tally(record, counts)

    storage.save_json(paths.summary_json, _build_summary(counts, args, adapter))


def cmd_trajectory(args: argparse.Namespace) -> None:
    _require_data_file(args.data)
    examples = datasets.load_examples(args.data, args.task, args.limit)
    paths = storage.make_run_paths(args.output_dir, with_trajectories=True)

    config = _run_config(args, "trajectory")
    config.update(
        {
            "tokens_selector": args.tokens,
            "alignment": args.alignment,
            "interesting_top_k": args.interesting_top_k,
            "trajectory_convention": (
                "states[0] is the initial recurrent state before any core-block application; "
                "states[i] for i >= 1 is the state after the i-th application of the recurrent "
                "core block, so there are num_steps + 1 states in total."
            ),
        }
    )
    storage.save_json(paths.run_json, config)

    completed_ids = storage.load_completed_ids(paths.predictions_jsonl)
    counts = {"total": 0, "correct": 0, "missing": 0}
    for record in storage.read_jsonl(paths.predictions_jsonl):
        _tally(record, counts)

    example_number = sum(1 for _ in storage.iter_trajectory_examples(paths))

    adapter = HuginnAdapter(args.model, args.device)

    for example in tqdm(examples, desc="trajectory"):
        if example.id in completed_ids:
            continue
        record, result = _generate_example(adapter, example, args)

        full_ids = result.prompt_ids + result.generated_ids
        full_states = adapter.capture_trajectory(full_ids, args.num_steps)
        tokens = trajectories.build_token_infos(
            result.prompt_ids, result.generated_ids, adapter.decode_token, adapter.special_token_ids()
        )
        selection = trajectories.select_and_slice(full_states, tokens, args.tokens, args.alignment, args.interesting_top_k)

        example_number += 1
        storage.save_trajectory_npz(
            paths.trajectory_path(example_number),
            selection.selected_states,
            np.array(selection.selected_positions, dtype=np.int64),
            input_length=len(result.prompt_ids),
            sequence_length=len(full_ids),
            num_steps=args.num_steps,
        )
        tokens_meta = trajectories.build_tokens_metadata(tokens, selection, args.alignment)
        tokens_meta["example_id"] = example.id
        storage.save_tokens_json(paths.tokens_path(example_number), tokens_meta)

        # Predictions are appended last so a completed record always implies its
        # matching trajectory/tokens files were written -- safe to resume on.
        storage.append_jsonl(paths.predictions_jsonl, record)
        _tally(record, counts)

    storage.save_json(paths.summary_json, _build_summary(counts, args, adapter))


def cmd_metrics(args: argparse.Namespace) -> None:
    paths = storage.RunPaths(args.run_directory)
    if not os.path.isdir(paths.trajectories_dir):
        raise SystemExit(f"{args.run_directory} has no trajectories/ directory; run `trajectory` first.")

    if os.path.exists(paths.metrics_jsonl):
        os.remove(paths.metrics_jsonl)

    verdict_counts = {"converging": 0, "looping": 0, "drifting": 0, "uncertain": 0}
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}

    example_pairs = list(storage.iter_trajectory_examples(paths))
    for npz_path, tokens_path in tqdm(example_pairs, desc="metrics"):
        trajectory = storage.load_trajectory_npz(npz_path)
        tokens_meta = storage.load_tokens_json(tokens_path)
        example_id = tokens_meta.get("example_id", os.path.basename(npz_path))
        alignment = tokens_meta["alignment"]

        for i, token_entry in enumerate(tokens_meta["tokens"]):
            token_states = trajectory["states"][:, i, :]
            results, classification = metrics.analyze_trajectory(token_states)

            verdict_counts[classification.verdict] += 1
            for name, value in classification.metric_values.items():
                if value is not None:
                    metric_sums[name] = metric_sums.get(name, 0.0) + value
                    metric_counts[name] = metric_counts.get(name, 0) + 1

            record = {
                "example_id": example_id,
                "token_index": token_entry["position"],
                "token_text": token_entry["token_text"],
                "scope": token_entry["scope"],
                "alignment": alignment,
                "verdict": classification.verdict,
                "confidence": classification.confidence,
                "vote_margin": classification.vote_margin,
                "metric_values": classification.metric_values,
                "metric_votes": classification.metric_votes,
            }
            storage.append_jsonl(paths.metrics_jsonl, record)

            if not args.no_plots and results:
                features = metrics.compute_features(
                    token_states, metrics.THRESHOLDS["tail_fraction"], metrics.THRESHOLDS["early_fraction"]
                )
                fig_dir = paths.figure_dir(example_id, token_entry["position"])
                plotting.save_all_figures(features, fig_dir)

    total_tokens = sum(verdict_counts.values())
    summary = {
        "token_count": total_tokens,
        "verdict_counts": verdict_counts,
        "metric_means": {name: metric_sums[name] / metric_counts[name] for name in metric_sums},
    }
    storage.save_json(paths.metrics_summary_json, summary)
