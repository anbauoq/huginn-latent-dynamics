"""Run-directory layout and all file I/O (JSON, JSONL, NPZ). No pickled tensors."""

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

RUN_JSON = "run.json"
PREDICTIONS_JSONL = "predictions.jsonl"
SUMMARY_JSON = "summary.json"
TRAJECTORIES_DIR = "trajectories"
TOKENS_DIR = "tokens"
METRICS_JSONL = "metrics.jsonl"
METRICS_SUMMARY_JSON = "metrics_summary.json"
FIGURES_DIR = "figures"


@dataclass
class RunPaths:
    root: str

    @property
    def run_json(self) -> str:
        return os.path.join(self.root, RUN_JSON)

    @property
    def predictions_jsonl(self) -> str:
        return os.path.join(self.root, PREDICTIONS_JSONL)

    @property
    def summary_json(self) -> str:
        return os.path.join(self.root, SUMMARY_JSON)

    @property
    def trajectories_dir(self) -> str:
        return os.path.join(self.root, TRAJECTORIES_DIR)

    @property
    def tokens_dir(self) -> str:
        return os.path.join(self.root, TOKENS_DIR)

    @property
    def metrics_jsonl(self) -> str:
        return os.path.join(self.root, METRICS_JSONL)

    @property
    def metrics_summary_json(self) -> str:
        return os.path.join(self.root, METRICS_SUMMARY_JSON)

    @property
    def figures_dir(self) -> str:
        return os.path.join(self.root, FIGURES_DIR)

    def trajectory_path(self, example_number: int) -> str:
        return os.path.join(self.trajectories_dir, f"example_{example_number:06d}.npz")

    def tokens_path(self, example_number: int) -> str:
        return os.path.join(self.tokens_dir, f"example_{example_number:06d}.json")

    def figure_dir(self, example_id: str, position: int) -> str:
        return os.path.join(self.figures_dir, str(example_id), f"token_{position}")


def make_run_paths(output_dir: str, with_trajectories: bool = False) -> RunPaths:
    paths = RunPaths(output_dir)
    os.makedirs(paths.root, exist_ok=True)
    if with_trajectories:
        os.makedirs(paths.trajectories_dir, exist_ok=True)
        os.makedirs(paths.tokens_dir, exist_ok=True)
    return paths


def save_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_completed_ids(predictions_path: str) -> set[str]:
    """IDs already present in predictions.jsonl, for resuming an interrupted run."""
    return {record["id"] for record in read_jsonl(predictions_path)}


def save_trajectory_npz(
    path: str,
    states: np.ndarray,
    selected_indices: np.ndarray,
    input_length: int,
    sequence_length: int,
    num_steps: int,
) -> None:
    """Save one example's trajectory. `states` has shape [num_steps + 1, num_selected, hidden]."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        states=states.astype(np.float32),
        selected_indices=np.asarray(selected_indices, dtype=np.int64),
        input_length=np.int64(input_length),
        sequence_length=np.int64(sequence_length),
        num_steps=np.int64(num_steps),
    )


def load_trajectory_npz(path: str) -> dict[str, Any]:
    with np.load(path) as data:
        return {
            "states": data["states"],
            "selected_indices": data["selected_indices"],
            "input_length": int(data["input_length"]),
            "sequence_length": int(data["sequence_length"]),
            "num_steps": int(data["num_steps"]),
        }


def save_tokens_json(path: str, tokens_meta: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_json(path, tokens_meta)


def load_tokens_json(path: str) -> dict[str, Any]:
    return load_json(path)


def iter_trajectory_examples(paths: RunPaths) -> Iterator[tuple[str, str]]:
    """Yield (npz_path, tokens_path) pairs sorted by example number."""
    if not os.path.isdir(paths.trajectories_dir):
        return
    for name in sorted(os.listdir(paths.trajectories_dir)):
        if not name.endswith(".npz"):
            continue
        stem = name[: -len(".npz")]
        npz_path = os.path.join(paths.trajectories_dir, name)
        tokens_path = os.path.join(paths.tokens_dir, f"{stem}.json")
        yield npz_path, tokens_path
