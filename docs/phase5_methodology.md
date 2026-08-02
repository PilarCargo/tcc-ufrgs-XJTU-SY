# Phase 5 methodology: causal LSTM RUL regression

Phase 5 predicts absolute `rul_minutes` for unseen bearings with one compact unidirectional LSTM
family. For acquisition `t` and sequence length `k`, the input is `[x(t-k+1), ..., x(t)]`; no
future acquisition, centered window, padding, cross-bearing state, bidirectionality, attention,
raw signal, health indicator, or PELT value is used. The first `k-1` acquisitions are dropped.

The immutable five Phase 3 folds are verified by physical and logical hashes. Fold-specific
features come from the training-only Phase 3 selection. Heavy-tailed non-negative features use
declared `log1p`; a robust feature scaler and context scaler are fitted on deterministic,
equal-size samples from each training bearing. The target center and scale are weighted training
statistics, with each training bearing contributing equally. Validation and test values never
fit these transforms.

The network contains a unidirectional LSTM, its final hidden state, an optional concatenation of
the current elapsed minutes, speed and load, and a compact scalar regression head. Training uses
AdamW, weighted Smooth L1 loss, gradient clipping, deterministic CPU seeds, and validation macro
MAE early stopping. Curated candidates differ only in sequence length, hidden size, layer count,
dropout, and learning rate. Ranking uses validation macro MAE, macro RMSE, parameter count,
sequence length, then declaration order. One-layer dropout is explicitly zero.

Raw predictions and `max(raw, 0)` are preserved without upper clipping, smoothing, isotonic
correction, or enforced monotonicity. Macro MAE across the 15 unseen bearings remains primary.
Global, bearing, fold, condition, life-stage, and onset-region metrics are secondary. Life stage
and the 15 estimated-onset records are retrospective annotations only and cannot affect model
selection.

Because sequence models drop prefixes, every Phase 4 comparison is recomputed on identical
eligible acquisition IDs. Complete-trajectory Phase 4 artifacts remain unchanged. Feature
occlusion sets one scaled vibration feature to its training baseline (zero after robust scaling)
at all timesteps and measures validation macro-MAE increase without retraining. This is predictive
association, not causality.

Execution records PyTorch/device details, epochs, best epoch, parameter count, sequence counts,
checkpoint size, inference time and peak RSS. CPU is the reproducible default; deterministic
algorithms are requested, but bitwise equality is not claimed on unsupported backends. A negative
result is scientifically acceptable. Conclusions remain limited to the 15 XJTU-SY bearings and
this single LSTM family.
