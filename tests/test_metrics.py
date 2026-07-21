import numpy as np

from huginn_research import metrics


def make_converging(num_steps=60, hidden=16, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = rng.normal(size=hidden)
    state = target + rng.normal(size=hidden) * 5
    states = [state]
    for _ in range(num_steps):
        state = target + (state - target) * 0.85 + rng.normal(scale=0.01, size=hidden)
        states.append(state)
    return np.array(states)


def make_looping(num_steps=60, hidden=16, seed=0, period=10, radius=3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    center = rng.normal(size=hidden)
    basis1 = rng.normal(size=hidden)
    basis1 /= np.linalg.norm(basis1)
    basis2 = rng.normal(size=hidden)
    basis2 -= basis2.dot(basis1) * basis1
    basis2 /= np.linalg.norm(basis2)
    states = []
    for t in range(num_steps + 1):
        theta = 2 * np.pi * t / period
        states.append(center + radius * (np.cos(theta) * basis1 + np.sin(theta) * basis2) + rng.normal(scale=0.01, size=hidden))
    return np.array(states)


def make_drifting(num_steps=60, hidden=16, seed=0, speed=1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    start = rng.normal(size=hidden)
    direction = rng.normal(size=hidden)
    direction /= np.linalg.norm(direction)
    return np.array([start + speed * t * direction + rng.normal(scale=0.02, size=hidden) for t in range(num_steps + 1)])


def test_converging_trajectory_is_classified_as_converging():
    _, classification = metrics.analyze_trajectory(make_converging())
    assert classification.verdict == "converging"
    assert classification.metric_votes["late_early_step_norm_ratio"] == "converging"


def test_looping_trajectory_is_classified_as_looping():
    _, classification = metrics.analyze_trajectory(make_looping())
    assert classification.verdict == "looping"


def test_drifting_trajectory_is_classified_as_drifting():
    _, classification = metrics.analyze_trajectory(make_drifting())
    assert classification.verdict == "drifting"


def test_too_short_trajectory_is_uncertain():
    states = np.random.default_rng(0).normal(size=(2, 8))
    results, classification = metrics.analyze_trajectory(states)
    assert results == []
    assert classification.verdict == "uncertain"


def test_pure_noise_trajectory_tends_uncertain_or_low_confidence():
    states = np.random.default_rng(1).normal(scale=1.0, size=(60, 16))
    _, classification = metrics.analyze_trajectory(states)
    assert classification.verdict in {"uncertain", "converging", "looping", "drifting"}


def test_winding_number_never_votes():
    results, _ = metrics.analyze_trajectory(make_looping())
    winding = next(r for r in results if r.name == "winding_number")
    assert winding.vote == "abstain"
    assert winding.confidence == 0.0


def test_classify_uncertain_when_too_few_voting_metrics():
    results = [
        metrics.MetricResult("a", 1.0, "converging", 0.9, "x", "convergence"),
    ]
    classification = metrics.classify(results)
    assert classification.verdict == "uncertain"


def test_classify_uncertain_on_close_margin():
    results = [
        metrics.MetricResult("a", 1.0, "converging", 0.5, "x", "convergence"),
        metrics.MetricResult("b", 1.0, "looping", 0.5, "x", "looping"),
        metrics.MetricResult("c", 1.0, "drifting", 0.1, "x", "drift"),
        metrics.MetricResult("d", 1.0, "converging", 0.5, "x", "convergence"),
    ]
    classification = metrics.classify(results)
    assert classification.verdict == "uncertain"


def test_family_balanced_weighting_prevents_family_dominance():
    # Five weak-but-consistent convergence votes vs one strong looping vote:
    # family averaging should stop the convergence family from steamrolling it.
    conv = [metrics.MetricResult(f"c{i}", 1.0, "converging", 0.4, "x", "convergence") for i in range(5)]
    loop = [metrics.MetricResult("l", 1.0, "looping", 0.9, "x", "looping")]
    classification = metrics.classify(conv + loop)
    assert classification.verdict == "looping"
