"""Simple diagnostic plots for one token's recurrent trajectory.

The PCA path plot is a visualization aid only -- classification always uses
the original hidden-dimension states (see huginn_research.metrics).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from huginn_research.metrics import TrajectoryFeatures


def plot_pca_path(features: TrajectoryFeatures, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    xy = features.pca_2d
    colors = np.arange(xy.shape[0])
    ax.plot(xy[:, 0], xy[:, 1], color="0.7", linewidth=1, zorder=1)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=colors, cmap="viridis", s=25, zorder=2)
    ax.scatter(xy[0, 0], xy[0, 1], marker="s", color="black", s=60, label="start", zorder=3)
    ax.scatter(xy[-1, 0], xy[-1, 1], marker="*", color="red", s=120, label="end", zorder=3)
    fig.colorbar(sc, ax=ax, label="recurrent step")
    ax.set_title("PCA path (visualization only)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_step_norm(features: TrajectoryFeatures, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(1, len(features.step_norms) + 1), features.step_norms, marker="o", markersize=3)
    ax.set_title("Step norm ||s_t - s_{t-1}||")
    ax.set_xlabel("recurrent step")
    ax.set_ylabel("norm")
    fig.tight_layout()
    _save(fig, path)


def plot_distance_to_tail_center(features: TrajectoryFeatures, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(len(features.dist_to_tail)), features.dist_to_tail, marker="o", markersize=3, color="tab:orange")
    ax.set_title("Distance to tail-center")
    ax.set_xlabel("recurrent step")
    ax.set_ylabel("distance")
    fig.tight_layout()
    _save(fig, path)


def plot_recurrence(features: TrajectoryFeatures, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(features.pairwise_dist, cmap="magma", origin="upper")
    fig.colorbar(im, ax=ax, label="distance")
    ax.set_title("Pairwise-distance recurrence matrix")
    ax.set_xlabel("recurrent step")
    ax.set_ylabel("recurrent step")
    fig.tight_layout()
    _save(fig, path)


def save_all_figures(features: TrajectoryFeatures, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plot_pca_path(features, os.path.join(out_dir, "pca_path.png"))
    plot_step_norm(features, os.path.join(out_dir, "step_norm.png"))
    plot_distance_to_tail_center(features, os.path.join(out_dir, "distance_to_tail_center.png"))
    plot_recurrence(features, os.path.join(out_dir, "recurrence.png"))


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=120)
    plt.close(fig)
