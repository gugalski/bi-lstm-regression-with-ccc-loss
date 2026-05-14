# CLAUDE.md — PHQ-9 Score Prediction from Voice (LSTM + Attention, Regression)

## Goal
Predict the **PHQ-9 depression severity score (0–27)** from voice recordings.
This is a regression task — the model outputs a continuous value, not a binary label.
Primary metric: **Concordance Correlation Coefficient (CCC)**, consistent with AVEC challenge standards.
Validation: **5-Fold Cross-Validation** stratified by PHQ-9 severity band.

---

## Dataset

```
recordings/       # numbered .wav files (e.g. 001.wav, 002.wav, …)
labels.csv        # columns: filename, subject_id, phq9_score (int, 0–27)
```

One row per recording. `subject_id` ensures the same person never appears in both train and
validation fold within a split (group-aware 5-fold).

---

## Repository Structure

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── recordings/
├── labels.csv
├── configs/
│   └── default.yaml
├── src/
│   ├── dataset.py        # AudioDataset, frame segmentation, DataLoader factory
│   ├── features.py       # eGeMAPS extraction via openSMILE
│   ├── model.py          # LSTMWithAttention, regression head
│   ├── train.py          # training loop, early stopping on val CCC
│   ├── evaluate.py       # CCC + full metric suite, plots
│   └── utils.py          # Config dataclass, seed, logging, CCC loss
├── scripts/
│   ├── extract_features.py   # pre-extract eGeMAPS for all recordings → cache
│   └── run_cv.py             # 5-fold CV entry-point
├── outputs/
│   ├── features/             # cached eGeMAPS .npy files
│   ├── checkpoints/          # fold_{k}/best_model.pt
│   ├── logs/
│   └── figures/
└── tests/
    ├── test_dataset.py
    ├── test_features.py
    ├── test_model.py
    └── fixtures/
        └── sample.wav
```

---

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Key packages: `torch>=2.3`, `torchaudio>=2.3`, `opensmile>=2.5`, `scikit-learn>=1.4`,
`pandas>=2.2`, `numpy>=1.26`, `matplotlib>=3.8`, `seaborn>=0.13`, `pyyaml>=6.0`, `tqdm>=4.66`

---

## Features — eGeMAPS (`src/features.py`)

**eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set)** is a standardised 88-dimensional
acoustic feature set designed specifically for paralinguistic and clinical speech research.
It covers frequency, energy, spectral, temporal, and voice quality parameters.

Extraction via `opensmile` Python package (wraps openSMILE binary, no separate install needed):

```python
import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,  # frame-level, not summary stats
)
# returns pd.DataFrame: rows = frames, cols = 88 eGeMAPS features
features = smile.process_file("recordings/001.wav")
```

**`LowLevelDescriptors`** gives one feature vector per 10 ms frame → shape `(T, 88)`.
This preserves temporal dynamics for the LSTM, unlike `Functionals` (which collapse to one vector).

### Feature caching (`scripts/extract_features.py`)

eGeMAPS extraction is slow. Run once before training:

```bash
python scripts/extract_features.py --recordings_dir recordings/ --output_dir outputs/features/
```

Saves one `{filename}.npy` per recording. `dataset.py` loads from cache; raw WAV is not touched
during training.

---

## Architecture (`src/model.py`)

```
(T, 88) eGeMAPS frames
    │
Linear projection  88 → 128   [optional input normalisation layer]
    │
BiLSTM × num_layers  [hidden=256 per direction]
    │
Bahdanau Attention  [α_t = softmax(v · tanh(W · h_t))]
    │
LayerNorm → Dropout(0.4)
    │
Linear(512, 1)  →  scalar PHQ-9 prediction
    │
Clamp [0, 27]   [hard constraint: score must be in valid range]
```

At inference, frame-level predictions are **mean-pooled per recording**.

---

## Loss Function (`src/utils.py`)

**CCC Loss** — directly optimises the primary evaluation metric:

```
CCC = (2 · σ_xy) / (σ_x² + σ_y² + (μ_x − μ_y)²)

CCC_loss = 1 − CCC
```

Optimising `1 − CCC` instead of MSE ensures the model learns both correlation *and* scale
simultaneously. MSE can produce a well-correlated but systematically shifted predictor
(e.g. always predicting mean ± constant), which scores poorly on CCC.

`HuberLoss` is available as a fallback (set `loss: huber` in config) — more robust to PHQ-9
outliers at the scale extremes (scores 0–4 and 22–27 are rare).

---

## Validation — 5-Fold CV (`scripts/run_cv.py`)

`sklearn.model_selection.GroupKFold(n_splits=5)` with `groups=subject_id`.

Group-aware split guarantees the same subject never appears in both train and val fold.
Folds are further **stratified by PHQ-9 band** (0–4, 5–9, 10–14, 15–27) to preserve
severity distribution across folds.

```bash
python scripts/run_cv.py --config configs/default.yaml
```

Each fold trains independently and saves `outputs/checkpoints/fold_{k}/best_model.pt`.
Early stopping monitors **val CCC** (patience=10), not loss.

---

## Config (`configs/default.yaml`)

```yaml
audio:
  sample_rate: 16000

features:
  feature_set: eGeMAPSv02
  feature_level: LowLevelDescriptors   # frame-level, shape (T, 88)
  cache_dir: outputs/features/

model:
  input_size: 88
  projection_size: 128
  hidden_size: 256
  num_layers: 2
  dropout: 0.4
  bidirectional: true

training:
  loss: ccc                   # ccc | huber | mse
  batch_size: 32
  max_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
  seed: 42

cv:
  n_splits: 5
  group_col: subject_id
```

---

## Metrics (`src/evaluate.py`)

Reported **per fold** and as **mean ± std** across all 5 folds.

| Metric | Why |
|--------|-----|
| **CCC** *(primary)* | AVEC standard; penalises both poor correlation and scale/bias mismatch |
| **MAE** | Mean absolute error in PHQ-9 points — clinically interpretable |
| **RMSE** | Penalises large errors more — missing a score of 22 as 10 matters |
| **R²** | Proportion of variance explained |
| **Pearson r** | Linear correlation strength |
| **Spearman ρ** | Rank correlation, robust to outliers |

### Output figures (`outputs/figures/`)

- `cv_metrics_boxplot.png` — distribution of each metric across 5 folds
- `scatter_pred_vs_true.png` — predicted vs. true PHQ-9 scatter, all folds combined,
  with identity line and per-severity-band colouring
- `error_by_severity.png` — MAE binned by true PHQ-9 band (shows where model struggles)
- `attention_heatmap.png` — averaged attention weights for high-error vs. low-error predictions
- `training_curves.png` — CCC & loss per epoch for each fold
- `cv_report.txt` — full per-fold metric table + aggregate stats

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| eGeMAPS over MFCC | Designed for clinical/paralinguistic tasks; 88 standardised features cover voice quality (jitter, shimmer) that MFCC misses |
| `LowLevelDescriptors` not `Functionals` | Preserves temporal dynamics — LSTM needs a sequence, not a single summary vector |
| Feature caching | eGeMAPS extraction via openSMILE is slow; pre-extract once, load `.npy` during training |
| CCC loss | Directly optimises the evaluation metric; MSE can yield good correlation but poor CCC |
| `GroupKFold` on `subject_id` | Same person must not appear in both train and val — prevents voice identity leakage |
| Clamp `[0, 27]` on output | Hard constraint — model cannot predict physically impossible PHQ-9 values |
| Early stopping on val CCC | Consistent with evaluation — stops when the metric we care about stops improving |
