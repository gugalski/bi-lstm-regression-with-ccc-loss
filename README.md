# BiLSTM + Attention with CCC Loss

Predicts depression severity (PHQ-9 score, 0–27) from voice recordings using a Bidirectional LSTM
with Bahdanau attention, trained with Concordance Correlation Coefficient (CCC) loss.

## Overview

| | |
|---|---|
| **Task** | Regression — continuous PHQ-9 score |
| **Primary metric** | CCC (AVEC challenge standard) |
| **Features** | eGeMAPS v02 — 88-dim frame-level acoustic descriptors |
| **Architecture** | BiLSTM × 2 + Bahdanau attention + regression head |
| **Validation** | 5-fold group-aware cross-validation (no subject leakage) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place recordings in `recordings/` and populate `labels.csv`:

```
filename,subject_id,phq9_score
001.wav,101,12
002.wav,102,5
...
```

One row per recording. `subject_id` ensures the same speaker never appears in both train and
validation splits.

## Usage

### 1. Extract features (run once)

```bash
python scripts/extract_features.py \
    --recordings_dir recordings/ \
    --output_dir outputs/features/
```

Saves `{stem}.npy` (shape `T × 88`) per recording. Training loads from cache — raw WAV is not
read during training.

### 2. Run 5-fold cross-validation

```bash
python scripts/run_cv.py --config configs/default.yaml
```

Trains each fold independently, applies early stopping on validation CCC, and saves the best
checkpoint to `outputs/checkpoints/fold_{k}/best_model.pt`.

### 3. Review results

| Output | Description |
|---|---|
| `outputs/cv_report.txt` | Per-fold metrics + mean ± std |
| `outputs/figures/cv_metrics_boxplot.png` | Metric distribution across folds |
| `outputs/figures/scatter_pred_vs_true.png` | Predictions vs ground truth |
| `outputs/figures/error_by_severity.png` | MAE per PHQ-9 severity band |
| `outputs/figures/training_curves.png` | Loss & CCC per epoch per fold |

## Architecture

```
(T, 88) eGeMAPS frames
    │
Linear projection  88 → 128  +  LayerNorm
    │
BiLSTM × 2  (hidden = 256 per direction)
    │
Bahdanau attention  →  context vector (512-dim)
    │
LayerNorm → Dropout(0.4)
    │
Linear(512, 1)  →  clamp [0, 27]
```

## Loss

CCC Loss directly optimises the evaluation metric:

```
CCC = 2·σ_xy / (σ_x² + σ_y² + (μ_x − μ_y)²)
loss = 1 − CCC
```

This penalises both poor correlation and scale/bias mismatch simultaneously. MSE alone can produce
a well-correlated predictor that scores poorly on CCC due to systematic offset.

Huber and MSE losses are available as alternatives (`loss: huber` / `loss: mse` in config).

## Configuration

Edit `configs/default.yaml` to adjust any hyperparameter. Key options:

```yaml
training:
  loss: ccc          # ccc | huber | mse
  batch_size: 32
  max_epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10

model:
  hidden_size: 256
  num_layers: 2
  dropout: 0.4
  bidirectional: true
```

## Tests

```bash
pytest tests/
```

## Design Decisions

| Decision | Rationale |
|---|---|
| eGeMAPS over MFCC | Designed for clinical/paralinguistic tasks; covers jitter, shimmer, and voice quality absent from MFCC |
| `LowLevelDescriptors` not `Functionals` | Preserves temporal dynamics — LSTM needs a sequence |
| Feature caching | eGeMAPS extraction is slow; extract once, load `.npy` during training |
| CCC loss | Matches the evaluation metric; MSE can yield good correlation but poor CCC |
| `GroupKFold` on `subject_id` | Prevents voice-identity leakage across splits |
| Output clamp `[0, 27]` | Hard constraint — no physically impossible PHQ-9 values |
| Early stopping on val CCC | Consistent with evaluation — stops when the metric we care about stagnates |
