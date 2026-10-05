# XJTU-SY pipeline technical guide

This document contains the detailed operational documentation for the experimental pipeline.
For the project introduction, institutional information, research question, and main results,
see the [main README](../README.md).

The scientific scope is deliberately narrow: **only the XJTU-SY Bearing Dataset may be used**.
The project must not substitute simulated data, add fault-type classification to the primary
scope, or claim generalization to industrial environments outside XJTU-SY.

## Current status

Phases 1 through 8 are implemented, validated, and executed on the real dataset. Phase 8 adds
probabilistic post-detection survival prognosis and its validation artifacts. The repository
contains the complete integrity audit, 52 vibration features, leakage-free degradation analysis,
classical RUL regression, causal unidirectional LSTM experiments, and final bearing-level
statistical consolidation, causal spectral degradation detection, survival-ready cohort
construction, and classical probabilistic survival models. The current automated suite contains
202 passing tests.

The exhaustive audit validated three operating conditions, 15 bearing directories, 9,216
acquisitions, and 301,989,888 signal rows with no critical dataset errors. The per-bearing counts
are documented in [docs/dataset_structure.md](dataset_structure.md). Re-run the audit only
when verifying a new checkout or intentionally checking the immutable source data again.

The primary consolidated result is deliberately retained even though it is negative: under
complete-bearing separation, absolute-RUL prediction, bearing-balanced development, and
bearing-macro MAE evaluation, the evaluated vibration models and causal temporal sequences did
not outperform the training-target median baseline on unseen XJTU-SY bearings. This result does
not imply that vibration, artificial intelligence, or predictive maintenance is ineffective in
other datasets or protocols.

## Local dataset

The dataset is publicly available from the
[official XJTU-SY repository](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets). The
extracted dataset is expected under the uppercase, project-relative `Data/` directory and is
treated as immutable source material:

```text
Data/
├── 35Hz12kN/
├── 37.5Hz11kN/
├── 40Hz10kN/
└── Introduction_to_XJTU-SY_Bearing_Dataset.pdf
```

Do not rename, move, rewrite, clean, or enrich files inside `Data/`. Generated metadata and
reports belong under `outputs/`, never beside the raw acquisitions. The source archive
`XJTU-SY.zip` is also ignored by Git.

Each acquisition CSV is expected to contain 32,768 data rows and these two columns, in order:

```text
Horizontal_vibration_signals,Vertical_vibration_signals
```

CSV filenames are numeric acquisition identifiers. Chronology must be established by parsing the
numeric stem; lexicographic sorting is invalid because it places `10.csv` before `2.csv`.

See [docs/dataset_structure.md](dataset_structure.md) for the observed acquisition counts,
nominal operating values, provenance boundary, and validation rules.

## Installation

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required. From the project root,
create the environment and install both runtime and development dependencies:

```bash
uv sync --group dev
```

Runtime dependencies include NumPy, pandas, PyArrow, PyYAML, SciPy, Matplotlib, psutil,
scikit-learn, scikit-survival, ruptures, and PyTorch. pytest and Ruff are development dependencies.
PyTorch is used only by the Phase 5 unidirectional LSTM pipeline; Phase 8 uses scikit-survival for
Kaplan–Meier, regularized Cox, Random Survival Forest, Brier, IBS, and concordance calculations.

For bounded local execution, configuration accepts at most eight inspection workers and each raw
CSV is rejected before numeric parsing if it exceeds 16 MiB. The observed XJTU-SY files are below
1.4 MB, so this guard identifies malformed inputs rather than valid acquisitions.

## Run the dataset audit

Run the exhaustive audit from the project root:

```bash
uv run python -m xjtu_sy_tcc.cli.audit --config configs/data.yaml
```

Phase-1 metadata and audit artifacts are written beneath:

```text
outputs/audits/
```

The audit checks the complete configured dataset rather than relying only on documented counts.
It validates paths, condition and bearing membership, numeric filename ordering,
sequence continuity, CSV schema and shape, numeric values, finite values, uniqueness constraints,
documented per-bearing acquisition counts, RUL invariants, and unexpected files or directories.
Duplicate-content checking is controlled by `detect_duplicate_content` in `configs/data.yaml`.

Critical validation errors gate downstream work. Do not extract features or train models when the
audit reports a critical error. Warnings and any explainable numbering irregularities must be
retained in the generated report rather than silently ignored.

The complete Phase 1 procedure, equations, duplicate-content policy, validated results, and
limitations are documented in
[docs/phase1_methodology.md](phase1_methodology.md). The observed directory structure and
per-bearing acquisition counts remain documented in
[docs/dataset_structure.md](dataset_structure.md).

## Build Phase-2 vibration features

Phase 2 consumes the validated `outputs/audits/metadata.parquet` manifest and reads one raw
acquisition at a time. Run it only after the Phase-1 audit passes:

```bash
uv run xjtu-sy-build-features --config configs/features.yaml
```

Use `--skip-plots` for a tables-only benchmark and `--force` to discard a safe partial extraction
checkpoint. The command logs JSON progress records, exits nonzero on processing or validation
failure, and writes:

```text
outputs/features/features.parquet
outputs/features/feature_validation.json
outputs/features/feature_summary.csv
outputs/figures/exploratory/condition_<id>/<bearing>/
```

The final table has one row per acquisition. Identifier and operating-context columns are kept for
traceability and downstream bearing-level splits. `rul_minutes` is explicitly a supervised target,
not an input feature. The configured production schema contains 52 vibration features: 15 time
features, seven general spectral features, and four band powers for each of two channels.

Time-domain definitions use population moments: standard deviation and variance use `ddof=0`, RMS
is the square root of mean squared amplitude, energy is the sum of squared amplitudes, and
kurtosis is Pearson kurtosis (`m4 / m2²`, not excess kurtosis). Crest, shape, impulse, and
clearance factors follow their standard amplitude-ratio definitions. Frequency features use a
Hann-window Welch PSD with segment length, overlap, and frequency bands declared in
`configs/features.yaml`. Spectral entropy is normalized to `[0, 1]`.

Computations parse signals as `float32` and accumulate scalar statistics in `float64`. A zero
denominator or zero-power spectrum produces `NaN` because the corresponding quantity is
mathematically undefined; it is never silently replaced by zero. Any such non-finite feature in
the experimental output is a critical validation error. Configured band edges may not exceed the
Nyquist frequency.

The pipeline keeps only one acquisition array in memory, while scalar rows and an atomic Parquet
checkpoint support bounded processing and safe restart. The validation report checks manifest
alignment, uniqueness, required columns, finite and plausible feature values, monotonic time,
nonnegative RUL, and final RUL zero. Runtime and peak resident memory are recorded in that report.

The reference full run on the local 9,216-file dataset completed feature extraction in 132.3
seconds with a measured peak resident memory of 340,787,200 bytes (325 MiB). Generating all plots
took approximately another 33 seconds. These are local benchmark observations, not portable
performance guarantees; storage speed, CPU, SciPy version, and an existing Matplotlib font cache
affect runtime.

Each bearing receives unsmoothed RMS, kurtosis, crest-factor, energy, and RUL trajectories, plus
combined time-series/PSD figures for its first, middle, and final acquisitions. PNG files use the
configured DPI and PDF counterparts preserve vector content.

Exact time-domain and spectral formulas, Welch settings, band definitions, checkpoint behavior,
validation rules, results, and limitations are documented in
[docs/phase2_methodology.md](phase2_methodology.md).

## Run Phase 3 degradation analysis

Phase 3 creates complete-bearing folds, evaluates prognostic feature quality, fits leakage-free
fold health indicators, and estimates degradation onset with PELT:

```bash
uv run xjtu-sy-run-phase3 --config configs/phase3.yaml
```

Use `--skip-plots` for tables only. Outputs are separated under `outputs/splits/`,
`outputs/prognostics/`, `outputs/degradation/`, and `outputs/figures/degradation/`.

Feature selection, transformations, robust scaling, PCA, orientation, and parameter selection use
only permitted training or training-plus-validation bearings. Test bearings never affect fitted
preprocessing or configuration selection. Their first 10% may calibrate their own baseline as an
explicit initial calibration period.

Every PELT result is an **estimated degradation onset**, not an official XJTU-SY label. Phase 3
reports stability, persistence, robust effect, and physical plausibility rather than accuracy.
It does not produce RUL predictions. Exact formulas, edge policies, leakage controls, and
limitations are in [docs/phase3_methodology.md](phase3_methodology.md).

The reference exhaustive Phase 3 run evaluated 5,760 sensitivity rows and completed in 409.9
seconds with approximately 556 MB resident memory. A restart using the matching validated
sensitivity artifact completed tables and all 44 PNG/PDF figure files in about 10 seconds. These
local timings are not portable performance guarantees.

## Run Phase 4 classical RUL regression

Phase 4 reuses the immutable five complete-bearing folds and selects every candidate using only
the three validation bearings in its fold:

```bash
uv run xjtu-sy-run-phase4 --config configs/phase4.yaml
```

Use `--skip-plots` for model and table generation only, and `--force-train` to disregard a
compatible validated cache. The command evaluates a median dummy baseline, a time-only Ridge
baseline, and Ridge, Random Forest, and histogram gradient boosting over fold-selected or all
vibration features. A secondary Ridge ablation combines selected vibration features with elapsed
time and known operating conditions.

Absolute `rul_minutes` is the target and is never an input. Sample weights give every training
bearing equal total influence; Ridge transformations and robust scaling are training-only; and
candidate selection minimizes validation-bearing macro MAE. Frozen models are evaluated once on
test bearings. Raw and non-negative predictions are both preserved.

Phase 3 health indicators are omitted as predictive inputs because their baseline uses the full
initial calibration interval and is therefore not available causally for all early acquisitions.
The PELT estimated onset is used only as a retrospective test-result annotation, never for model
training or selection. Outputs are written beneath `outputs/rul/`. Full definitions, the
experiment matrix, leakage controls, metrics, and limitations are documented in
[docs/phase4_methodology.md](phase4_methodology.md).

The validated Phase 4 run evaluated 185 candidates and generated 82,944 test-prediction rows. The
median dummy achieved the best primary macro MAE, 269.63 minutes. Time-only Ridge achieved 290.76
minutes, selected-feature Ridge 337.64 minutes, selected-feature HistGradientBoosting 384.43
minutes, and selected-feature Random Forest 402.92 minutes. Acquisition-weighted and
bearing-macro metrics are intentionally reported separately.

## Run Phase 5 causal LSTM regression

Phase 5 evaluates whether past vibration-feature acquisitions add useful temporal context while
preserving the frozen complete-bearing protocol:

```bash
uv run xjtu-sy-run-phase5 --config configs/phase5.yaml
```

The three experiments are a selected-feature causal LSTM, the same LSTM with current elapsed time
and known operating condition, and a sequence-length-one diagnostic ablation. Windows never cross
bearing boundaries and contain only the current and previous acquisitions. Incomplete prefixes
are dropped. Selection and early stopping use validation bearings; test bearings are evaluated
only after freezing. Phase 4 baselines are recomputed on exactly the eligible LSTM acquisition
IDs. Details are in [docs/phase5_methodology.md](phase5_methodology.md).

The validated Phase 5 run evaluated 65 candidates and generated 27,213 eligible test-prediction
rows. Macro MAE was 280.03 minutes for the single-acquisition LSTM ablation, 285.06 minutes for
the vibration-plus-time LSTM, and 346.74 minutes for the temporal vibration-only LSTM. Temporal
context therefore did not improve over the `k=1` ablation in the evaluated configuration family.

## Run Phase 6 statistical consolidation

Phase 6 performs no training. It harmonizes the immutable Phase 4 and Phase 5 predictions,
constructs native, pairwise-matched, and strict-common supports, and treats the bearing as the
independent unit for uncertainty and paired inference:

```bash
uv run xjtu-sy-run-phase6 --config configs/phase6.yaml
```

The command generates bearing bootstraps, paired tests and effects, Holm-adjusted primary
comparisons, ranking uncertainty, feature and cost consolidation, final tables/figures, and
traceable Portuguese thesis drafts. Non-significance is not interpreted as equivalence. See
[docs/phase6_methodology.md](phase6_methodology.md).

## Run Phase 7 causal degradation and survival datasets

Phase 7 replaces retrospective full-trajectory prediction with an operationally causal gate:

```bash
uv run xjtu-sy-run-phase7 --config configs/phase7.yaml
```

The pipeline compares predeclared RMS, uniform-frequency spectral SKL, and mechanically informed
shaft-order spectral SKL detectors using validation bearings only, freezes one detector per fold,
then applies it to test bearings. It also creates persistent causal alarms and causal
feature trends, and full-event or administratively censored survival cohorts. Detector selection
uses training and validation bearings; the frozen detector is applied once to each test
bearing. PELT remains a retrospective descriptive reference and no survival model is trained.
Details are in [docs/phase7_methodology.md](phase7_methodology.md).

The validated consolidation harmonized 110,157 prediction rows and retained 8,961 acquisitions
per primary model on strict common support. No candidate comparison against the dummy was
significant after Holm correction. The vibration-plus-time LSTM did outperform the vibration-only
LSTM in an exploratory paired contrast, but it still did not outperform the dummy. Phase 6 writes
machine-readable results, tables in four formats, figures, Portuguese manuscript drafts, and a
claims manifest beneath `outputs/consolidation/` without editing the monograph itself.

## Run Phase 8 probabilistic survival prognosis

Phase 8 trains auditable survival models on the causal dynamic landmarks generated by Phase 7:

```bash
uv run xjtu-sy-run-phase8 --config configs/phase8.yaml
```

The command verifies the immutable fold hashes, preserves the frozen Phase 7 detector, filters and
ranks causal attributes using training bearings only, and compares a landmark Kaplan–Meier
baseline, context-only regularized Cox, causal-feature regularized Cox, and Random Survival
Forest. Model selection uses validation-bearing macro integrated Brier score. Test evaluation is
performed only after freezing each model.

Primary scenarios are full event and administrative censoring at 60, 120, and 240 minutes.
Secondary analyses cover 30 minutes and the Phase 7 training-only target-censoring horizons.
Reports distinguish detector coverage from conditional prognostic performance and aggregate
uncertainty by complete bearing rather than treating repeated landmarks as independent. Full
curves, horizon probabilities, median-survival coverage, IBS, Brier, IPCW concordance,
calibration, bearing-clustered Cox PH diagnostics, computational measurements, tables, figures,
and formal Portuguese drafts are
written under `outputs/survival_models/`. See
[docs/phase8_methodology.md](phase8_methodology.md) for formulas, leakage controls, and
limitations. No neural survival model is trained and no monograph or LaTeX source is edited.

## RUL definition and leakage controls

For bearing `b`, acquisition index `t`, total acquisition count `N_b`, and nominal acquisition
interval `delta_t`, absolute RUL in minutes is defined as:

```text
RUL_b,t = (N_b - 1 - t) * delta_t
```

`t` is zero-based after validated numeric ordering. Therefore, the final acquisition has RUL zero.
A normalized life fraction may be used only for visualization and exploratory analysis.

The following leakage controls apply throughout the implemented modeling phases and to any future
extension:

- Never use the final lifetime or total acquisition count of a held-out bearing as an input
  feature.
- Split data by bearing, not by randomly mixing acquisitions from the same bearing across train
  and test sets.
- Fit scalers, imputers, feature selection, health-indicator transformations, and model tuning
  only on training bearings.
- Treat PELT change points as unsupervised estimates. XJTU-SY provides no official exact
  degradation-onset ground truth.
- Use absolute RUL in minutes as the primary supervised target.

## Provenance and known limitations

The nominal 25.6 kHz sampling frequency, 32,768 samples per acquisition, and approximately
one-minute acquisition interval are configuration values whose provenance is recorded as
`Data/Introduction_to_XJTU-SY_Bearing_Dataset.pdf`. The CSV files themselves contain no timestamp,
sampling-frequency field, measurement-unit field, or acquisition-interval field. Consequently,
the audit can verify observed rows and values but cannot independently infer those nominal
properties from CSV content.

Other current limitations include:

- No official label identifies the exact onset of degradation.
- Nominal operating conditions come from dataset documentation and directory labels rather than
  per-acquisition measurements.
- Phase 4 evaluates a deliberately bounded set of classical regressors, and Phase 5 evaluates one
  compact unidirectional LSTM family rather than an unrestricted architecture search.
- Phase 5 temporal models drop incomplete sequence prefixes, requiring support-matched comparison.
- Phase 6 inference uses 15 bearings as independent experimental units; non-significant results do
  not demonstrate equivalence.
- Life-stage and estimated-onset analyses are retrospective, and there is no external-dataset or
  industrial deployment validation.
- Dataset-specific condition differences do not support causal claims about rotation speed or
  radial load.
- Conclusions must remain limited to the 15 XJTU-SY run-to-failure bearings and their three
  operating conditions.
