#!/usr/bin/env python3
"""Standalone script: hidden-space recurrence loop detection for saved trajectories.

Reads the `trajectories/*.npz` + `tokens/*.json` produced by
`huginn-research trajectory` (either capture mode) and, for every analyzed
token, computes only:

  * recurrence_score  -- the primary loop-detection metric. Built from the
    pairwise distance matrix D_ij = ||h_i - h_j|| in the ORIGINAL hidden
    space (no PCA):

        recurrence(k) = 1 - mean_t ||h_{t+k} - h_t|| / median_{i != j} ||h_i - h_j||

    taken as the best score over nontrivial lags k >= --min-lag. The best
    lag is also reported (an interpretable loop period).

  * late_activity_ratio -- required guard. A trajectory that has simply
    stopped moving also has many nearby states, but only because it
    converged, not because it loops. This is the ratio of late-window to
    early-window step norms; recurrence only counts as a loop when this
    stays non-negligible.

  * path_efficiency_ratio -- supporting confirmation. Low values (little net
    displacement despite a long path) support looping over ordinary drift.

`loop_detected` = recurrence_score >= --recurrence-threshold AND
                  late_activity_ratio >= --late-activity-threshold

This intentionally does not touch huginn_research/metrics.py, cli.py, or any
existing output file -- it only reads the trajectory/token files that
`trajectory` already wrote and writes its own outputs under
RUN_DIRECTORY/loop_metrics/:

    loop_metrics.jsonl                        one record per analyzed token
    distance_matrices/<example>__token_<pos>.npy   the full D_ij matrix per token
    figures/<metric>_panel.png                histogram + metric-vs-token-index

Persistent homology (H1) is not computed here: a real implementation needs an
extra dependency (e.g. ripser/gudhi) outside this project's minimal
dependency set, so it is left out of this first pass.

Usage:
    python scripts/loop_recurrence.py outputs/custom_traj
    python scripts/loop_recurrence.py outputs/custom_traj --plot-metric path_efficiency_ratio
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from huginn_research import storage  # noqa: E402
from huginn_research.metrics import THRESHOLDS  # noqa: E402

EPS = 1e-8

METRIC_LABELS = {
    "recurrence_score": "nontrivial recurrence score",
    "late_activity_ratio": "late/early step-norm ratio",
    "path_efficiency_ratio": "path efficiency ratio",
}


def pairwise_distance_matrix(states: np.ndarray) -> np.ndarray:
    """D_ij = ||h_i - h_j|| in the original hidden space. `states` is [T, H]."""
    diffs = states[:, None, :] - states[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def recurrence_score(distance_matrix: np.ndarray, min_lag: int) -> tuple[Optional[int], Optional[float]]:
    """Best recurrence(k) over nontrivial lags k >= min_lag. Returns (best_lag, best_score)."""
    num_states = distance_matrix.shape[0]
    if num_states <= min_lag:
        return None, None
    nonzero = distance_matrix[distance_matrix > 0]
    if nonzero.size == 0:
        return None, None
    scale = float(np.median(nonzero))

    best_lag, best_score = None, -np.inf
    for lag in range(min_lag, num_states):
        lag_distances = np.array([distance_matrix[t + lag, t] for t in range(num_states - lag)])
        score = 1.0 - float(lag_distances.mean()) / (scale + EPS)
        if score > best_score:
            best_score, best_lag = score, lag
    return best_lag, float(best_score)


def late_activity_ratio(states: np.ndarray, early_fraction: float = 0.25, late_fraction: float = 0.25) -> float:
    """Late-window / early-window mean step norm. Near 0 means motion has stalled (converged, not looping)."""
    step_norms = np.linalg.norm(np.diff(states, axis=0), axis=-1)
    n = len(step_norms)
    if n < 2:
        return 1.0
    early_count = min(max(int(round(n * early_fraction)), 1), n)
    late_count = min(max(int(round(n * late_fraction)), 1), n)
    early = step_norms[:early_count].mean()
    late = step_norms[-late_count:].mean()
    return float(late / (early + EPS))


def path_efficiency_ratio(states: np.ndarray) -> Optional[float]:
    """||h_T - h_0|| / sum_t ||h_{t+1} - h_t||. Low values support looping over drift."""
    step_norms = np.linalg.norm(np.diff(states, axis=0), axis=-1)
    total = float(step_norms.sum())
    if total < EPS:
        return None
    displacement = float(np.linalg.norm(states[-1] - states[0]))
    return displacement / total


def compute_for_run(
    run_directory: str,
    min_lag: int,
    recurrence_threshold: float,
    late_activity_threshold: float,
) -> tuple[list[dict], str]:
    """Compute the three metrics + D_ij for every analyzed token in `run_directory`.

    Returns (rows, out_dir). Writes loop_metrics.jsonl and one .npy per token
    under RUN_DIRECTORY/loop_metrics/.
    """
    paths = storage.RunPaths(run_directory)
    if not os.path.isdir(paths.trajectories_dir):
        raise SystemExit(f"{run_directory} has no trajectories/ directory; run `huginn-research trajectory` first.")

    out_dir = os.path.join(run_directory, "loop_metrics")
    matrices_dir = os.path.join(out_dir, "distance_matrices")
    os.makedirs(matrices_dir, exist_ok=True)

    jsonl_path = os.path.join(out_dir, "loop_metrics.jsonl")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    rows: list[dict] = []
    for npz_path, tokens_path in storage.iter_trajectory_examples(paths):
        trajectory = storage.load_trajectory_npz(npz_path)
        tokens_meta = storage.load_tokens_json(tokens_path)
        example_id = tokens_meta.get("example_id", os.path.basename(npz_path))

        for i, token_entry in enumerate(tokens_meta["tokens"]):
            states = trajectory["states"][:, i, :].astype(np.float64)  # [num_steps + 1, hidden]
            distance_matrix = pairwise_distance_matrix(states)

            best_lag, score = recurrence_score(distance_matrix, min_lag)
            activity = late_activity_ratio(states)
            efficiency = path_efficiency_ratio(states)
            loop_detected = bool(
                score is not None and score >= recurrence_threshold and activity >= late_activity_threshold
            )

            position = token_entry["position"]
            matrix_name = f"{example_id}__token_{position}.npy"
            np.save(os.path.join(matrices_dir, matrix_name), distance_matrix.astype(np.float32))

            row = {
                "example_id": example_id,
                "token_index": position,
                "token_text": token_entry["token_text"],
                "scope": token_entry["scope"],
                "recurrence_score": score,
                "best_recurrence_lag": best_lag,
                "late_activity_ratio": activity,
                "path_efficiency_ratio": efficiency,
                "loop_detected": loop_detected,
                "distance_matrix_path": os.path.join("distance_matrices", matrix_name),
            }
            storage.append_jsonl(jsonl_path, row)
            rows.append(row)

    return rows, out_dir


def plot_metric_panel(
    rows: list[dict],
    out_path: Path,
    metric_key: str,
    metric_label: str,
    threshold: Optional[float] = None,
    title: str = "",
) -> Optional[Path]:
    """One figure, two axes: metric histogram + metric vs token_index (matches the
    jacobian-panel layout, generalized to any metric key computed by this script)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(r["token_index"], r[metric_key]) for r in rows if r.get(metric_key) is not None]
    if not pts:
        return None
    pts.sort(key=lambda p: p[0])
    idxs = np.array([p[0] for p in pts], dtype=float)
    values = np.array([p[1] for p in pts], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.hist(values, bins=min(24, max(8, len(values) // 3)), color="#264653", edgecolor="white", alpha=0.9)
    if threshold is not None:
        ax.axvline(threshold, color="#e76f51", ls="--", lw=1.5, label=f"threshold={threshold:.3f}")
    ax.axvline(float(values.mean()), color="#2a9d8f", ls=":", lw=1.5, label=f"mean={values.mean():.3f}")
    ax.set_xlabel(metric_label)
    ax.set_ylabel("token count")
    ax.set_title("Distribution")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(idxs, values, color="#adb5bd", lw=0.8, zorder=1)
    if threshold is not None:
        above = values >= threshold
        ax2.scatter(idxs[~above], values[~above], c="#264653", s=18, zorder=2, label=f"< {threshold:.2f}")
        if np.any(above):
            ax2.scatter(idxs[above], values[above], c="#e76f51", s=36, zorder=3, label=f">= {threshold:.2f}")
        ax2.axhline(threshold, color="#e76f51", ls="--", lw=1.2)
        ax2.legend(fontsize=8, loc="best")
    else:
        ax2.scatter(idxs, values, c="#264653", s=18, zorder=2)
    ax2.set_xlabel("token index")
    ax2.set_ylabel(metric_label)
    ax2.set_title(f"{metric_label} vs token index")

    fig.suptitle((title + f"  ·  n={len(pts)}").strip(" ·"), fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _default_plot_threshold(metric_key: str, recurrence_threshold: float, late_activity_threshold: float) -> Optional[float]:
    return {
        "recurrence_score": recurrence_threshold,
        "late_activity_ratio": late_activity_threshold,
        "path_efficiency_ratio": None,  # no single natural cutoff; pass --plot-threshold to set one
    }[metric_key]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hidden-space recurrence loop detection for saved trajectories.")
    parser.add_argument("run_directory", help="A directory previously produced by `huginn-research trajectory`")
    parser.add_argument(
        "--min-lag",
        type=int,
        default=THRESHOLDS["looping"]["min_lag"],
        help="Smallest lag k considered nontrivial (default: %(default)s)",
    )
    parser.add_argument(
        "--recurrence-threshold",
        type=float,
        default=THRESHOLDS["looping"]["recurrence_score"]["strong"],
        help="recurrence_score at/above which a token is a strong loop candidate (default: %(default)s)",
    )
    parser.add_argument(
        "--late-activity-threshold",
        type=float,
        default=THRESHOLDS["looping"]["late_activity_min"],
        help="late_activity_ratio at/above which late motion counts as non-negligible (default: %(default)s)",
    )
    parser.add_argument(
        "--plot-metric",
        choices=list(METRIC_LABELS),
        default="recurrence_score",
        help="Which computed metric to render as a histogram + vs-token-index panel (default: %(default)s)",
    )
    parser.add_argument(
        "--plot-threshold",
        type=float,
        default=None,
        help="Override the reference line / color-split threshold used in the plot",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip generating the panel figure")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    rows, out_dir = compute_for_run(
        args.run_directory,
        min_lag=args.min_lag,
        recurrence_threshold=args.recurrence_threshold,
        late_activity_threshold=args.late_activity_threshold,
    )

    print(f"wrote {len(rows)} token records to {os.path.join(out_dir, 'loop_metrics.jsonl')}")
    print(f"saved {len(rows)} pairwise distance matrices under {os.path.join(out_dir, 'distance_matrices')}")
    loop_count = sum(1 for r in rows if r["loop_detected"])
    print(f"loop_detected: {loop_count}/{len(rows)} tokens")

    if not args.no_plot and rows:
        threshold = args.plot_threshold
        if threshold is None:
            threshold = _default_plot_threshold(args.plot_metric, args.recurrence_threshold, args.late_activity_threshold)
        fig_path = Path(out_dir) / "figures" / f"{args.plot_metric}_panel.png"
        result = plot_metric_panel(
            rows,
            fig_path,
            args.plot_metric,
            METRIC_LABELS[args.plot_metric],
            threshold=threshold,
            title=os.path.basename(os.path.normpath(args.run_directory)),
        )
        if result:
            print(f"saved panel plot to {result}")


if __name__ == "__main__":
    main()
