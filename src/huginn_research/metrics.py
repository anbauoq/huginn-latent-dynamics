"""Trajectory metrics, classification, and an extension point for future metrics.

All metrics are computed on the ORIGINAL hidden-dimension states. A 2D PCA
projection is used only for the `winding_number` diagnostic (and for plotting
elsewhere) -- it never contributes to the classification vote.

Thresholds live in a single `THRESHOLDS` dict below so they can be retuned
without touching the metric logic.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

EPS = 1e-8

Family = Literal["convergence", "looping", "drift"]
Vote = Literal["converging", "looping", "drifting", "abstain"]

# ---------------------------------------------------------------------------
# Thresholds -- the single place to retune classification behavior.
# For each (weak, strong) pair, `strong` is the value at which a metric votes
# with full confidence (1.0) and `weak` is the value at which confidence hits
# 0. Whether higher or lower values are "stronger" is inferred from whether
# strong > weak or strong < weak.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "min_steps_for_metrics": 4,
    "tail_fraction": 0.25,
    "early_fraction": 0.25,
    "convergence": {
        "late_early_step_norm_ratio": {"weak": 0.6, "strong": 0.3},
        "rolling_var_late_early_ratio": {"weak": 0.6, "strong": 0.3},
        "late_early_tail_center_ratio": {"weak": 0.6, "strong": 0.3},
        "mean_log_step_ratio": {"weak": -0.05, "strong": -0.15},
        "cosine_consecutive_states": {"weak": 0.9, "strong": 0.99},
    },
    "looping": {
        "recurrence_score": {"weak": 0.5, "strong": 0.7},
        "autocorr_peak": {"weak": 0.5, "strong": 0.7},
        "spectral_peak_fraction": {"weak": 0.3, "strong": 0.5},
        "path_closedness": {"weak": 0.5, "strong": 0.7},
        "radius_cv": {"weak": 0.3, "strong": 0.15},
        "radius_relative_scale_min": 0.05,
        "late_activity_min": 0.2,
        "min_lag": 3,
    },
    "drift": {
        "path_efficiency_ratio": {"weak": 0.4, "strong": 0.6},
        "path_efficiency_loop_ceiling": 0.15,
        "late_step_norm_ratio": {"weak": 0.5, "strong": 0.7},
        "directional_persistence": {"weak": 0.3, "strong": 0.5},
        "growth_slope": {"weak": 0.0, "strong": 0.05},
    },
    "classification": {
        "min_voting_metrics": 4,
        "min_top_score": 0.35,
        "min_margin": 0.15,
    },
}


@dataclass
class MetricResult:
    name: str
    value: Optional[float]
    vote: Vote
    confidence: float
    reason: str
    family: Family = field(default="convergence")


@dataclass
class ClassificationResult:
    verdict: str
    confidence: float
    vote_margin: float
    metric_values: dict[str, Optional[float]]
    metric_votes: dict[str, str]


@dataclass
class TrajectoryFeatures:
    states: np.ndarray  # [T, H] float64
    step_vectors: np.ndarray  # [T - 1, H]
    step_norms: np.ndarray  # [T - 1]
    tail_center: np.ndarray  # [H]
    dist_to_tail: np.ndarray  # [T], "radius" around the tail-center
    pairwise_dist: np.ndarray  # [T, T]
    pca_2d: np.ndarray  # [T, 2], visualization/diagnostic only
    late_activity_ratio: float  # late/early step-norm ratio; near 0 means motion has stalled


def _ramp(value: float, weak: float, strong: float) -> float:
    """Linear confidence ramp from 0 at `weak` to 1 at `strong`, clipped to [0, 1]."""
    if strong == weak:
        return 1.0
    frac = (value - weak) / (strong - weak)
    return float(min(max(frac, 0.0), 1.0))


def _windows(n: int, early_fraction: float, late_fraction: float) -> tuple[slice, slice]:
    early_count = min(max(int(round(n * early_fraction)), 1), n)
    late_count = min(max(int(round(n * late_fraction)), 1), n)
    return slice(0, early_count), slice(n - late_count, n)


def _pca_2d(states: np.ndarray) -> np.ndarray:
    centered = states - states.mean(axis=0, keepdims=True)
    if states.shape[0] < 2:
        return np.zeros((states.shape[0], 2))
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    comps = vt[: min(2, vt.shape[0])]
    projected = centered @ comps.T
    if projected.shape[1] < 2:
        projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
    return projected


def compute_features(states: np.ndarray, tail_fraction: float, early_fraction: float) -> TrajectoryFeatures:
    """Precompute the shared quantities every metric function reads from."""
    states = np.asarray(states, dtype=np.float64)
    step_vectors = np.diff(states, axis=0)
    step_norms = np.linalg.norm(step_vectors, axis=-1)
    tail_count = min(max(int(round(states.shape[0] * tail_fraction)), 1), states.shape[0])
    tail_center = states[-tail_count:].mean(axis=0)
    dist_to_tail = np.linalg.norm(states - tail_center[None, :], axis=-1)
    pairwise_dist = np.linalg.norm(states[:, None, :] - states[None, :, :], axis=-1)
    pca_2d = _pca_2d(states)
    if len(step_norms) >= 2:
        early_sl, late_sl = _windows(len(step_norms), early_fraction, tail_fraction)
        late_activity_ratio = float(step_norms[late_sl].mean() / (step_norms[early_sl].mean() + EPS))
    else:
        late_activity_ratio = 1.0
    return TrajectoryFeatures(
        states, step_vectors, step_norms, tail_center, dist_to_tail, pairwise_dist, pca_2d, late_activity_ratio
    )


# ---------------------------------------------------------------------------
# Convergence family
# ---------------------------------------------------------------------------


def metric_step_norms(f: TrajectoryFeatures) -> MetricResult:
    value = float(f.step_norms.mean()) if len(f.step_norms) else 0.0
    return MetricResult("step_norms_mean", value, "abstain", 0.0, "mean per-step displacement norm (diagnostic)", "convergence")


def metric_late_early_step_norm_ratio(f: TrajectoryFeatures) -> MetricResult:
    n = len(f.step_norms)
    if n < 2:
        return MetricResult("late_early_step_norm_ratio", None, "abstain", 0.0, "insufficient steps", "convergence")
    early_sl, late_sl = _windows(n, THRESHOLDS["early_fraction"], THRESHOLDS["tail_fraction"])
    ratio = float(f.step_norms[late_sl].mean() / (f.step_norms[early_sl].mean() + EPS))
    t = THRESHOLDS["convergence"]["late_early_step_norm_ratio"]
    conf = _ramp(ratio, t["weak"], t["strong"])
    vote = "converging" if conf > 0 else "abstain"
    return MetricResult(
        "late_early_step_norm_ratio", ratio, vote, conf, "late-window step norms shrunk relative to early window", "convergence"
    )


def _rolling_var(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return np.array([float(x.var())]) if len(x) else np.array([0.0])
    return np.array([x[i : i + window].var() for i in range(len(x) - window + 1)])


def metric_rolling_variance_ratio(f: TrajectoryFeatures) -> MetricResult:
    n = len(f.step_norms)
    if n < 4:
        return MetricResult("rolling_var_late_early_ratio", None, "abstain", 0.0, "insufficient steps", "convergence")
    window = max(n // 5, 2)
    rv = _rolling_var(f.step_norms, window)
    early_sl, late_sl = _windows(len(rv), THRESHOLDS["early_fraction"], THRESHOLDS["tail_fraction"])
    ratio = float(rv[late_sl].mean() / (rv[early_sl].mean() + EPS))
    t = THRESHOLDS["convergence"]["rolling_var_late_early_ratio"]
    conf = _ramp(ratio, t["weak"], t["strong"])
    vote = "converging" if conf > 0 else "abstain"
    return MetricResult(
        "rolling_var_late_early_ratio", ratio, vote, conf, "rolling variance of step norms shrunk late vs early", "convergence"
    )


def metric_tail_center_distance(f: TrajectoryFeatures) -> MetricResult:
    value = float(f.dist_to_tail.mean())
    return MetricResult("tail_center_distance_mean", value, "abstain", 0.0, "mean distance to tail-center (diagnostic)", "convergence")


def metric_late_early_tail_center_ratio(f: TrajectoryFeatures) -> MetricResult:
    n = len(f.dist_to_tail)
    if n < 2:
        return MetricResult("late_early_tail_center_ratio", None, "abstain", 0.0, "insufficient steps", "convergence")
    early_sl, late_sl = _windows(n, THRESHOLDS["early_fraction"], THRESHOLDS["tail_fraction"])
    ratio = float(f.dist_to_tail[late_sl].mean() / (f.dist_to_tail[early_sl].mean() + EPS))
    t = THRESHOLDS["convergence"]["late_early_tail_center_ratio"]
    conf = _ramp(ratio, t["weak"], t["strong"])
    vote = "converging" if conf > 0 else "abstain"
    return MetricResult(
        "late_early_tail_center_ratio", ratio, vote, conf, "distance to tail-center shrunk late vs early", "convergence"
    )


def metric_mean_log_step_ratio(f: TrajectoryFeatures) -> MetricResult:
    if len(f.step_norms) < 3:
        return MetricResult("mean_log_step_ratio", None, "abstain", 0.0, "insufficient steps", "convergence")
    a, b = f.step_norms[:-1], f.step_norms[1:]
    valid = (a > EPS) & (b > EPS)
    if not np.any(valid):
        return MetricResult("mean_log_step_ratio", None, "abstain", 0.0, "step norms too small to compare", "convergence")
    mean_log_ratio = float(np.log(b[valid] / a[valid]).mean())
    t = THRESHOLDS["convergence"]["mean_log_step_ratio"]
    conf = _ramp(mean_log_ratio, t["weak"], t["strong"])
    vote = "converging" if conf > 0 else "abstain"
    return MetricResult(
        "mean_log_step_ratio", mean_log_ratio, vote, conf, "consecutive step norms shrink exponentially on average", "convergence"
    )


def metric_cosine_consecutive_states(f: TrajectoryFeatures) -> MetricResult:
    if f.states.shape[0] < 2:
        return MetricResult("cosine_consecutive_states", None, "abstain", 0.0, "insufficient steps", "convergence")
    a, b = f.states[:-1], f.states[1:]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + EPS
    cos = (a * b).sum(axis=1) / denom
    mean_cos = float(cos.mean())
    t = THRESHOLDS["convergence"]["cosine_consecutive_states"]
    conf = _ramp(mean_cos, t["weak"], t["strong"])
    vote = "converging" if conf > 0 else "abstain"
    return MetricResult(
        "cosine_consecutive_states", mean_cos, vote, conf, "consecutive states point in nearly the same direction", "convergence"
    )


# ---------------------------------------------------------------------------
# Looping / recurrence family
# ---------------------------------------------------------------------------


def _best_recurrence(pairwise_dist: np.ndarray, min_lag: int) -> tuple[Optional[int], Optional[float]]:
    """Find the lag with a genuine *return* -- an interior local minimum in the
    lag/mean-distance curve. Nearby-in-time points are always close for any
    smooth trajectory (open or closed), so the smallest available lag is never
    accepted on its own; only a dip that is flanked by larger distances on
    both sides counts as recurrence. This avoids flagging monotonic drift
    (whose lag/distance curve only ever increases) as looping.
    """
    T = pairwise_dist.shape[0]
    if T <= min_lag + 3:
        return None, None
    nonzero = pairwise_dist[pairwise_dist > 0]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0

    lag_dist = {
        lag: float(np.array([pairwise_dist[i, i + lag] for i in range(T - lag)]).mean()) for lag in range(min_lag, T)
    }
    best_lag, best_mean_dist = None, np.inf
    for lag in range(min_lag + 1, T - 1):
        if lag_dist[lag] < lag_dist[lag - 1] and lag_dist[lag] <= lag_dist[lag + 1] and lag_dist[lag] < best_mean_dist:
            best_mean_dist, best_lag = lag_dist[lag], lag

    if best_lag is None:
        return None, None
    score = 1.0 - min(best_mean_dist / (scale + EPS), 1.0)
    return best_lag, score


def _stalled(f: TrajectoryFeatures) -> bool:
    """True once late-window motion has decayed away -- a settled trajectory
    cannot be orbiting, no matter how its (now mostly noise) shape scores on
    recurrence/periodicity metrics."""
    return f.late_activity_ratio <= THRESHOLDS["looping"]["late_activity_min"]


def metric_pairwise_recurrence_matrix(f: TrajectoryFeatures) -> MetricResult:
    nonzero = f.pairwise_dist[f.pairwise_dist > 0]
    value = float(np.median(nonzero)) if nonzero.size else 0.0
    return MetricResult(
        "pairwise_distance_median", value, "abstain", 0.0, "median pairwise state distance (see recurrence.png)", "looping"
    )


def metric_recurrence(f: TrajectoryFeatures) -> MetricResult:
    if _stalled(f):
        return MetricResult("recurrence_score", None, "abstain", 0.0, "trajectory has effectively stopped moving", "looping")
    min_lag = THRESHOLDS["looping"]["min_lag"]
    best_lag, score = _best_recurrence(f.pairwise_dist, min_lag)
    if best_lag is None:
        return MetricResult("recurrence_score", None, "abstain", 0.0, "trajectory too short for lag search", "looping")
    t = THRESHOLDS["looping"]["recurrence_score"]
    conf = _ramp(score, t["weak"], t["strong"])
    vote = "looping" if conf > 0 else "abstain"
    return MetricResult(
        "recurrence_score",
        score,
        vote,
        conf,
        f"best nontrivial recurrence at lag {best_lag} with normalized distance {1 - score:.3f}",
        "looping",
    )


def metric_best_recurrence_lag(f: TrajectoryFeatures) -> MetricResult:
    min_lag = THRESHOLDS["looping"]["min_lag"]
    best_lag, _ = _best_recurrence(f.pairwise_dist, min_lag)
    value = float(best_lag) if best_lag is not None else None
    return MetricResult("best_recurrence_lag", value, "abstain", 0.0, "lag minimizing average revisit distance (diagnostic)", "looping")


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    centered = x - x.mean()
    denom = float(np.dot(centered, centered))
    if denom < EPS:
        return np.zeros(max_lag + 1)
    return np.array(
        [1.0 if lag == 0 else float(np.dot(centered[: len(x) - lag], centered[lag:])) / denom for lag in range(max_lag + 1)]
    )


def _detrend(x: np.ndarray) -> np.ndarray:
    """Remove a linear trend so monotonic (drift-like) series do not masquerade
    as periodic under autocorrelation/spectral analysis."""
    t = np.arange(len(x))
    if len(x) < 2:
        return x - x.mean()
    slope, intercept = np.polyfit(t, x, 1)
    return x - (slope * t + intercept)


def metric_autocorrelation_peak(f: TrajectoryFeatures) -> MetricResult:
    if _stalled(f):
        return MetricResult("autocorr_peak", None, "abstain", 0.0, "trajectory has effectively stopped moving", "looping")
    min_lag = THRESHOLDS["looping"]["min_lag"]
    radius = _detrend(f.dist_to_tail)
    if len(radius) <= min_lag + 1:
        return MetricResult("autocorr_peak", None, "abstain", 0.0, "trajectory too short", "looping")
    acf_vals = _acf(radius, len(radius) - 1)
    candidates = acf_vals[min_lag:]
    if len(candidates) == 0:
        return MetricResult("autocorr_peak", None, "abstain", 0.0, "no lags beyond trivial range", "looping")
    peak_idx = int(np.argmax(candidates))
    peak_lag = peak_idx + min_lag
    peak_val = float(candidates[peak_idx])
    t = THRESHOLDS["looping"]["autocorr_peak"]
    conf = _ramp(peak_val, t["weak"], t["strong"])
    vote = "looping" if conf > 0 else "abstain"
    return MetricResult(
        "autocorr_peak", peak_val, vote, conf, f"autocorrelation of tail-center distance peaks at lag {peak_lag}", "looping"
    )


def metric_spectral_peak(f: TrajectoryFeatures) -> MetricResult:
    if _stalled(f):
        return MetricResult(
            "spectral_peak_fraction", None, "abstain", 0.0, "trajectory has effectively stopped moving", "looping"
        )
    radius = f.dist_to_tail
    n = len(radius)
    if n < 4:
        return MetricResult("spectral_peak_fraction", None, "abstain", 0.0, "trajectory too short", "looping")
    x = _detrend(radius)
    power = np.abs(np.fft.rfft(x)) ** 2
    power[0] = 0.0
    total = float(power[1:].sum())
    if total < EPS:
        return MetricResult("spectral_peak_fraction", None, "abstain", 0.0, "no spectral power beyond DC", "looping")
    peak_bin = int(np.argmax(power[1:])) + 1
    peak_fraction = float(power[peak_bin] / total)
    t = THRESHOLDS["looping"]["spectral_peak_fraction"]
    conf = _ramp(peak_fraction, t["weak"], t["strong"])
    vote = "looping" if conf > 0 else "abstain"
    return MetricResult(
        "spectral_peak_fraction",
        peak_fraction,
        vote,
        conf,
        f"dominant frequency bin {peak_bin}/{n} concentrates {peak_fraction:.2f} of spectral power",
        "looping",
    )


def metric_path_closedness(f: TrajectoryFeatures) -> MetricResult:
    if _stalled(f):
        return MetricResult("path_closedness", None, "abstain", 0.0, "trajectory has effectively stopped moving", "looping")
    # Nearby-in-time points are always close for any smooth trajectory, so a
    # "return" only counts if it is to a point well back in time -- at least a
    # third of the trajectory -- not merely to the last few steps.
    T = f.pairwise_dist.shape[0]
    min_gap = max(THRESHOLDS["looping"]["min_lag"], T // 3)
    if T <= min_gap + 1:
        return MetricResult("path_closedness", None, "abstain", 0.0, "trajectory too short", "looping")
    nonzero = f.pairwise_dist[f.pairwise_dist > 0]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0
    candidates = f.pairwise_dist[-1, : T - min_gap]
    if candidates.size == 0:
        return MetricResult("path_closedness", None, "abstain", 0.0, "no earlier states to compare", "looping")
    min_dist = float(candidates.min())
    value = 1.0 - min(min_dist / (scale + EPS), 1.0)
    t = THRESHOLDS["looping"]["path_closedness"]
    conf = _ramp(value, t["weak"], t["strong"])
    vote = "looping" if conf > 0 else "abstain"
    return MetricResult("path_closedness", value, vote, conf, "final state lies close to an earlier point on the path", "looping")


def metric_radius_cv(f: TrajectoryFeatures) -> MetricResult:
    tail_count = min(max(int(round(len(f.dist_to_tail) * THRESHOLDS["tail_fraction"])), 1), len(f.dist_to_tail))
    tail_window = f.dist_to_tail[-tail_count:]
    mean_r = float(tail_window.mean())
    cv = float(tail_window.std() / (mean_r + EPS))
    overall_scale = float(f.dist_to_tail.max()) if len(f.dist_to_tail) else 0.0
    relative_radius = mean_r / (overall_scale + EPS)
    t = THRESHOLDS["looping"]["radius_cv"]
    min_relative = THRESHOLDS["looping"]["radius_relative_scale_min"]
    if relative_radius <= min_relative:
        return MetricResult(
            "radius_cv", cv, "abstain", 0.0, "tail radius too small relative to path scale to indicate an orbit", "looping"
        )
    conf = _ramp(cv, t["weak"], t["strong"])
    vote = "looping" if conf > 0 else "abstain"
    return MetricResult("radius_cv", cv, vote, conf, "distance from the tail-center stays steady late in the trajectory", "looping")


def metric_winding_number(f: TrajectoryFeatures) -> MetricResult:
    T = f.pca_2d.shape[0]
    if T < 3:
        return MetricResult("winding_number", 0.0, "abstain", 0.0, "PCA diagnostic only; not used for classification", "looping")
    center = f.pca_2d.mean(axis=0)
    vectors = f.pca_2d - center[None, :]
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    dtheta = np.diff(angles)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    winding = float(dtheta.sum() / (2 * np.pi))
    return MetricResult("winding_number", winding, "abstain", 0.0, "PCA diagnostic only; not used for classification", "looping")


# ---------------------------------------------------------------------------
# Drift family
# ---------------------------------------------------------------------------


def metric_total_path_length(f: TrajectoryFeatures) -> MetricResult:
    value = float(f.step_norms.sum()) if len(f.step_norms) else 0.0
    return MetricResult("total_path_length", value, "abstain", 0.0, "sum of per-step displacement norms (diagnostic)", "drift")


def metric_displacement(f: TrajectoryFeatures) -> MetricResult:
    if f.states.shape[0] < 2:
        return MetricResult("displacement", None, "abstain", 0.0, "insufficient steps", "drift")
    value = float(np.linalg.norm(f.states[-1] - f.states[0]))
    return MetricResult("displacement", value, "abstain", 0.0, "distance from first to last state (diagnostic)", "drift")


def metric_path_efficiency(f: TrajectoryFeatures) -> MetricResult:
    if f.states.shape[0] < 2 or f.step_norms.sum() < EPS:
        return MetricResult("path_efficiency_ratio", None, "abstain", 0.0, "insufficient or degenerate path", "drift")
    displacement = float(np.linalg.norm(f.states[-1] - f.states[0]))
    ratio = displacement / float(f.step_norms.sum())
    loop_ceiling = THRESHOLDS["drift"]["path_efficiency_loop_ceiling"]
    if ratio <= loop_ceiling:
        # Displacement negligible next to total path length: the path folds
        # back on itself rather than going anywhere -- evidence for looping,
        # not drift, even though this metric is computed in the drift family.
        conf = _ramp(ratio, loop_ceiling, 0.0)
        return MetricResult(
            "path_efficiency_ratio",
            ratio,
            "looping",
            conf,
            "displacement is negligible relative to total path length (closed path)",
            "drift",
        )
    t = THRESHOLDS["drift"]["path_efficiency_ratio"]
    conf = _ramp(ratio, t["weak"], t["strong"])
    vote = "drifting" if conf > 0 else "abstain"
    return MetricResult(
        "path_efficiency_ratio", ratio, vote, conf, "trajectory displacement is a large fraction of total path length", "drift"
    )


def metric_late_step_norm_ratio(f: TrajectoryFeatures) -> MetricResult:
    n = len(f.step_norms)
    if n < 2 or f.step_norms.max() < EPS:
        return MetricResult("late_step_norm_ratio", None, "abstain", 0.0, "insufficient or degenerate steps", "drift")
    _, late_sl = _windows(n, THRESHOLDS["early_fraction"], THRESHOLDS["tail_fraction"])
    ratio = float(f.step_norms[late_sl].mean() / (f.step_norms.max() + EPS))
    t = THRESHOLDS["drift"]["late_step_norm_ratio"]
    conf = _ramp(ratio, t["weak"], t["strong"])
    vote = "drifting" if conf > 0 else "abstain"
    return MetricResult(
        "late_step_norm_ratio", ratio, vote, conf, "late-window step norms remain close to the peak step norm", "drift"
    )


def metric_directional_persistence(f: TrajectoryFeatures) -> MetricResult:
    if f.step_vectors.shape[0] < 2:
        return MetricResult("directional_persistence", None, "abstain", 0.0, "insufficient steps", "drift")
    a, b = f.step_vectors[:-1], f.step_vectors[1:]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + EPS
    persistence = float(((a * b).sum(axis=1) / denom).mean())
    t = THRESHOLDS["drift"]["directional_persistence"]
    conf = _ramp(persistence, t["weak"], t["strong"])
    vote = "drifting" if conf > 0 else "abstain"
    return MetricResult(
        "directional_persistence", persistence, vote, conf, "consecutive step vectors point in a consistent direction", "drift"
    )


def metric_growth_from_start(f: TrajectoryFeatures) -> MetricResult:
    T = f.states.shape[0]
    if T < 3 or f.step_norms.mean() < EPS:
        return MetricResult("growth_slope", None, "abstain", 0.0, "insufficient or degenerate path", "drift")
    dist_from_start = np.linalg.norm(f.states - f.states[0][None, :], axis=1)
    scale = float(f.step_norms.mean()) + EPS
    _, late_sl = _windows(T, THRESHOLDS["early_fraction"], THRESHOLDS["tail_fraction"])
    ts = np.arange(T)[late_sl]
    ys = dist_from_start[late_sl] / scale
    if len(ts) < 2:
        return MetricResult("growth_slope", None, "abstain", 0.0, "insufficient late-window points", "drift")
    slope = float(np.polyfit(ts, ys, 1)[0])
    t = THRESHOLDS["drift"]["growth_slope"]
    conf = _ramp(slope, t["weak"], t["strong"])
    vote = "drifting" if conf > 0 else "abstain"
    return MetricResult(
        "growth_slope", slope, vote, conf, "distance from the starting state keeps growing late in the trajectory", "drift"
    )


METRIC_REGISTRY: list[Callable[[TrajectoryFeatures], MetricResult]] = [
    metric_step_norms,
    metric_late_early_step_norm_ratio,
    metric_rolling_variance_ratio,
    metric_tail_center_distance,
    metric_late_early_tail_center_ratio,
    metric_mean_log_step_ratio,
    metric_cosine_consecutive_states,
    metric_pairwise_recurrence_matrix,
    metric_recurrence,
    metric_best_recurrence_lag,
    metric_autocorrelation_peak,
    metric_spectral_peak,
    metric_path_closedness,
    metric_radius_cv,
    metric_winding_number,
    metric_total_path_length,
    metric_displacement,
    metric_path_efficiency,
    metric_late_step_norm_ratio,
    metric_directional_persistence,
    metric_growth_from_start,
]

# Extension point for future research metrics (DMD, persistent homology, RQA,
# Jacobian probes, intrinsic dimension, CKA, ...). Append callables with the
# same `(TrajectoryFeatures) -> MetricResult` signature.
EXTRA_METRICS: list[Callable[[TrajectoryFeatures], MetricResult]] = []


def classify(results: list[MetricResult]) -> ClassificationResult:
    """Family-balanced weighted vote: each family contributes total weight 1.0,
    split evenly across its non-abstaining metrics, so a family with many
    correlated metrics does not outweigh a family with few.
    """
    cfg = THRESHOLDS["classification"]
    metric_values = {m.name: (float(m.value) if m.value is not None else None) for m in results}
    metric_votes = {m.name: m.vote for m in results}

    voting = [m for m in results if m.vote != "abstain"]
    scores = {"converging": 0.0, "looping": 0.0, "drifting": 0.0}
    for family in {m.family for m in voting}:
        fam_metrics = [m for m in voting if m.family == family]
        weight = 1.0 / len(fam_metrics)
        for m in fam_metrics:
            scores[m.vote] += weight * m.confidence

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ranked[0]
    margin = top_score - ranked[1][1]

    if len(voting) < cfg["min_voting_metrics"] or top_score < cfg["min_top_score"] or margin < cfg["min_margin"]:
        return ClassificationResult("uncertain", top_score, margin, metric_values, metric_votes)

    return ClassificationResult(top_label, top_score, margin, metric_values, metric_votes)


def analyze_trajectory(states: np.ndarray) -> tuple[list[MetricResult], ClassificationResult]:
    """Compute all registered metrics and classify one token's trajectory.

    `states` has shape [num_recurrent_states, hidden_size] for a single
    selected token.
    """
    if states.shape[0] < THRESHOLDS["min_steps_for_metrics"]:
        return [], ClassificationResult("uncertain", 0.0, 0.0, {}, {})

    features = compute_features(states, THRESHOLDS["tail_fraction"], THRESHOLDS["early_fraction"])
    results = [fn(features) for fn in METRIC_REGISTRY] + [fn(features) for fn in EXTRA_METRICS]
    return results, classify(results)
