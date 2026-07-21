import numpy as np

from huginn_research import storage


def test_trajectory_npz_round_trip(tmp_path):
    states = np.random.default_rng(0).normal(size=(9, 4, 8)).astype(np.float32)
    selected_indices = np.array([2, 5, 7, 9], dtype=np.int64)
    path = tmp_path / "trajectories" / "example_000001.npz"

    storage.save_trajectory_npz(
        str(path), states, selected_indices, input_length=6, sequence_length=10, num_steps=8
    )
    loaded = storage.load_trajectory_npz(str(path))

    assert np.allclose(loaded["states"], states)
    assert np.array_equal(loaded["selected_indices"], selected_indices)
    assert loaded["input_length"] == 6
    assert loaded["sequence_length"] == 10
    assert loaded["num_steps"] == 8


def test_tokens_json_round_trip(tmp_path):
    path = tmp_path / "tokens" / "example_000001.json"
    meta = {"alignment": "token", "tokens": [{"position": 3, "token_text": "hi"}]}
    storage.save_tokens_json(str(path), meta)
    assert storage.load_tokens_json(str(path)) == meta


def test_predictions_jsonl_append_and_resume(tmp_path):
    paths = storage.make_run_paths(str(tmp_path))
    storage.append_jsonl(paths.predictions_jsonl, {"id": "1", "correct": True})
    storage.append_jsonl(paths.predictions_jsonl, {"id": "2", "correct": False})

    records = list(storage.read_jsonl(paths.predictions_jsonl))
    assert [r["id"] for r in records] == ["1", "2"]

    completed = storage.load_completed_ids(paths.predictions_jsonl)
    assert completed == {"1", "2"}


def test_read_jsonl_missing_file_yields_nothing(tmp_path):
    assert list(storage.read_jsonl(str(tmp_path / "missing.jsonl"))) == []


def test_iter_trajectory_examples_sorted(tmp_path):
    paths = storage.make_run_paths(str(tmp_path), with_trajectories=True)
    for n in [2, 1, 3]:
        storage.save_trajectory_npz(
            paths.trajectory_path(n),
            np.zeros((2, 1, 4), dtype=np.float32),
            np.array([0]),
            input_length=1,
            sequence_length=2,
            num_steps=1,
        )
        storage.save_tokens_json(paths.tokens_path(n), {"alignment": "token", "tokens": []})

    pairs = list(storage.iter_trajectory_examples(paths))
    names = [p[0].split("/")[-1] for p in pairs]
    assert names == ["example_000001.npz", "example_000002.npz", "example_000003.npz"]


def test_run_json_save_and_load(tmp_path):
    paths = storage.make_run_paths(str(tmp_path))
    storage.save_json(paths.run_json, {"model": "x", "num_steps": 64})
    assert storage.load_json(paths.run_json) == {"model": "x", "num_steps": 64}
