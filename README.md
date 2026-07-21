# huginn-research

Minimal CLI research tools for running and analyzing
[`tomg-group-umd/huginn-0125`](https://huggingface.co/tomg-group-umd/huginn-0125):
run inference on benchmark questions, capture its recurrent latent
trajectories, and classify each token's trajectory as `converging`,
`looping`, `drifting`, or `uncertain`.

## Installation

```bash
pip install -e .
```

Requires `torch`, `transformers`, `numpy`, `scipy`, `matplotlib`, `tqdm`.
Loading the model requires `trust_remote_code=True` (handled internally).

## Supported JSONL schemas

**Numeric:**

```json
{"id": "1", "question": "What is 17 + 25?", "answer": "42"}
```

**Multiple choice:**

```json
{
  "id": "1",
  "context": "...",
  "question": "...",
  "options": ["A) ...", "B) ...", "C) ..."],
  "answer": "B"
}
```

The task is auto-detected per item (presence of `"options"` implies
multiple-choice) unless `--task numeric` or `--task multiple-choice` is
passed explicitly.

## Commands

### `generate` -- run inference and check correctness

```bash
huginn-research generate data/gsm8k.jsonl \
  --task numeric \
  --device cuda:0 \
  --num-steps 64 \
  --output-dir outputs/gsm8k
```

### `trajectory` -- run inference and capture recurrent latent trajectories

`--capture-mode` selects how recurrent states are captured:

- `generation` (default) -- states are captured *while* `model.generate()`
  runs, with no second forward pass. Each generated token's trajectory is the
  recurrent state at the causal position that predicted it, so this mode
  supports generated output tokens only and is always prediction-aligned.
- `teacher-forced` -- a separate forward pass runs over the full
  prompt + generated sequence after generation completes. This is the only
  mode that can analyze prompt tokens, arbitrary sequence positions, or use
  `--alignment token`.

```bash
huginn-research trajectory data/gsm8k.jsonl \
  --task numeric \
  --device cuda:0 \
  --num-steps 64 \
  --tokens output:last \
  --output-dir outputs/gsm8k_trajectories
```

```bash
huginn-research trajectory data/ar_lsat.jsonl \
  --task multiple-choice \
  --device cuda:0 \
  --num-steps 64 \
  --tokens interesting:5 \
  --output-dir outputs/ar_lsat_trajectories
```

```bash
huginn-research trajectory data/gsm8k.jsonl \
  --task numeric \
  --device cuda:0 \
  --num-steps 64 \
  --capture-mode teacher-forced \
  --tokens input \
  --alignment token \
  --output-dir outputs/gsm8k_input_trajectories
```

### `metrics` -- compute trajectory metrics and classify tokens

```bash
huginn-research metrics outputs/gsm8k_trajectories
```

Add `--no-plots` to skip figure generation.

## Token selector syntax (`--tokens`)

| Selector | Meaning |
|---|---|
| `all` | every prompt and generated position |
| `input` | prompt positions only |
| `output` | generated positions only |
| `output:first` | first generated token |
| `output:last` | last generated token |
| `numeric` | positions whose decoded token contains a digit |
| `content` | non-special, non-whitespace positions |
| `indices:3,8,12` | explicit absolute token positions |
| `contains:substring` | positions whose decoded token contains `substring` |
| `interesting:5` | top-5 positions by a cheap pre-scan for persistent late movement and periodicity (this score is never used as a classification metric) |

With `--capture-mode generation`, only generated output tokens have captured
trajectories, so `--tokens input` is rejected and any selector that resolves
to a prompt position (e.g. `indices:` referencing one) fails with a clear
error. Use `--capture-mode teacher-forced` to analyze prompt tokens.

## `token` vs `prediction` alignment

- `token`: analyze the hidden state at the selected token's own position.
  Only available with `--capture-mode teacher-forced`.
- `prediction`: analyze the preceding causal position -- the one whose
  forward pass predicted the selected token. Position 0 has no predecessor;
  under `--capture-mode teacher-forced` it is clamped to itself and flagged
  (`alignment_clamped: true` in the token metadata) rather than silently
  misaligned. Under `--capture-mode generation` every trajectory is
  prediction-aligned by construction (the generated token's own position
  always has a predecessor, since it is never the first token in the
  sequence), so this never happens.

`--capture-mode generation` only supports prediction alignment, since it
never captures the state at a token's own position -- only at the position
that predicted it. `--alignment token` requires `--capture-mode teacher-forced`.

## Output structure

```text
outputs/gsm8k_trajectories/
├── run.json              # exact model/runtime/selector settings for reproducing the run
├── predictions.jsonl     # one record per example (generation + correctness)
├── summary.json          # accuracy, missing-answer count, model/runtime settings
├── trajectories/
│   └── example_000001.npz   # states, selected_indices, input_length, sequence_length, num_steps
├── tokens/
│   └── example_000001.json  # token ids/text, absolute positions, scope, alignment
├── metrics.jsonl         # one record per analyzed token (written by `metrics`)
├── metrics_summary.json  # verdict counts and metric means (written by `metrics`)
└── figures/
    └── EXAMPLE_ID/token_POSITION/
        ├── pca_path.png
        ├── step_norm.png
        ├── distance_to_tail_center.png
        └── recurrence.png
```

`states` in each NPZ has shape `[num_steps + 1, num_selected_tokens,
hidden_size]`: index 0 is the initial recurrent state (before any
core-block application) and index `i >= 1` is the state after the i-th
application of the recurrent core block. `run.json` and every `tokens/*.json`
file record `capture_mode` alongside `alignment`.

Each entry in `tokens/*.json` records, for `--capture-mode teacher-forced`:

```text
position, token_id, token_text, scope,
aligned_position, aligned_token_id, aligned_token_text, alignment_clamped
```

and for `--capture-mode generation`:

```text
generation_index, position, predictor_position, token_id, token_text, scope
```

In both cases `position` is the analyzed token's absolute index in the full
prompt + generated sequence, and the NPZ `states` columns correspond 1:1, in
order, to the `tokens` list -- the `metrics` command reads both the same way
regardless of which mode produced them.

## PCA is for visualization only

All convergence/looping/drift metrics and the `converging` / `looping` /
`drifting` / `uncertain` classification are computed on the **original
hidden-dimension** states. The 2D PCA projection (`pca_path.png`, the
`winding_number` diagnostic) exists purely to visualize a trajectory -- it
never feeds into or overrides any classification decision.
