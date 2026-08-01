# Phase 4 methodology: classical RUL regression

## Task and frozen protocol

The supervised target is absolute remaining useful life in minutes,
`RUL(b,t) = N_b - 1 - t`. It is a target, never an input. Phase 4 consumes the 9,216-row Phase 2
feature table and verifies both the byte-level SHA-256 and logical configuration hash of the
Phase 3 fold manifest. Each of five folds contains nine training, three validation, and three
test bearings, with one validation and one test bearing from every condition. Each bearing is
tested exactly once.

Preprocessing and fitting use training bearings only. Candidate ranking uses validation macro
MAE, followed by macro RMSE, lower declared complexity, and deterministic configuration order.
Only after selection is the frozen candidate applied once to the test bearings. Test results do
not change any experiment choice.

## Inputs and experiments

The experiment matrix contains `dummy_median`, `time_only_ridge`, selected-feature Ridge, Random
Forest, and HistGradientBoosting, their all-52-feature ablations, and a secondary
selected-features-plus-time Ridge ablation. Time-only inputs are elapsed minutes, rotation speed,
and radial load. Vibration-only experiments contain no elapsed-time field. Known operating
conditions are permitted, but bearing identity, paths, RUL, lifetime, normalized life, fold,
subset, PELT output, and every future-derived field are rejected.

The Phase 3 health indicator is retrospective: its initial center uses the complete first 10% of
each trajectory. It is therefore omitted from full-trajectory prediction rather than being
misrepresented as causal. Estimated degradation onset is never an input.

## Training and prediction

For a training bearing with `n_b` acquisitions, raw sample weight is `1/n_b`; weights are scaled
to mean one. Thus each bearing has equal total weight. Ridge uses declared `log1p` transforms for
verified non-negative heavy-tailed vibration features and a training-only `RobustScaler`. Scaler
fitting uses equal-size deterministic interpolation per training bearing so long trajectories do
not dominate. Trees receive finite numeric values without scaling, clipping, winsorization, or
imputation. All supported estimators receive the balanced sample weights.

Models predict RUL directly in minutes. Both raw prediction and `max(raw, 0)` are stored; the
latter is the primary engineering prediction. No test-lifetime upper bound, monotonic correction,
isotonic regression, or retrospective smoothing is applied. Negative raw values and chronological
increases in predicted RUL are reported as diagnostics.

## Evaluation

The primary quantity is the unweighted mean of test-bearing MAEs. Reports distinguish global
acquisition-weighted MAE, RMSE, and R² from bearing-macro MAE, RMSE, and valid R². Bearing tables
also contain signed error, lifetime-normalized descriptive errors, negative predictions, and
monotonicity violations. Fold, operating-condition, life-stage, and computational-cost tables are
separate.

Early `[0,.33)`, intermediate `[.33,.66)`, and late `[.66,1]` stages use complete lifetime only
after prediction for descriptive evaluation. Similarly, exactly one frozen test-fold onset row
per bearing labels `before estimated onset` and `after estimated onset`. This is retrospective
post-hoc analysis: the onset is neither official ground truth nor used in fitting or selection.

Ridge coefficients refer to the scaled model input. Tree permutation importance is computed on
validation bearings with deterministic permutations and macro MAE loss. These associations aid
interpretation but are not causal effects. Fit and inference times, artifact size, observation
and feature counts, effective weighted sample size, and process peak RSS are recorded.

## Reproducibility and limitations

Configuration, input, and artifact hashes guard restarts; outputs use atomic replacement and an
exclusive run lock. A compatible tables-only run can later generate figures without refitting.
Non-finite inputs or predictions fail validation rather than being imputed.

The evidence is limited to 15 run-to-failure XJTU-SY bearings and three documented conditions.
Five folds do not make condition differences causal, estimated onset is not a label, and elapsed
time can encode dataset-specific lifetime structure. Phase 4 contains no deep learning. A future
phase may compare temporal models under the same frozen bearing protocol, while retaining the
classical baselines and leakage checks established here.
