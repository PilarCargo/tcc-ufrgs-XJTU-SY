# XJTU-SY bearing prognostics thesis project

This repository supports an undergraduate Mechanical Engineering thesis on predictive
maintenance of rolling bearings. Its objective is to build a reproducible experimental pipeline
for degradation analysis, health-indicator construction, unsupervised change-point estimation,
and Remaining Useful Life (RUL) regression.

The scientific scope is deliberately narrow: **only the XJTU-SY Bearing Dataset may be used**.
The project must not substitute simulated data, add fault-type classification to the primary
scope, or claim generalization to industrial environments outside XJTU-SY.

## Current status

Phase 1 provides dataset discovery, loading, metadata construction, and exhaustive audit
reporting. Phase 2 adds incremental time- and frequency-domain feature extraction plus
reproducible exploratory degradation figures. Health indicators, PELT change-point estimation,
classical machine-learning models, temporal deep learning, and model evaluation are not
implemented yet.

A preliminary read-only inspection found the expected three operating-condition directories,
15 bearing directories, and 9,216 acquisition CSV files. The per-bearing counts are documented
in [docs/dataset_structure.md](docs/dataset_structure.md). These observations are not an
exhaustive audit result. Run the audit command below to validate every acquisition before using
the data in any later phase.

## Local dataset

The extracted dataset is already present under the uppercase, project-relative `Data/`
directory. It is treated as immutable source material:

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

See [docs/dataset_structure.md](docs/dataset_structure.md) for the observed acquisition counts,
nominal operating values, provenance boundary, and validation rules.

## Installation

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required. From the project root,
create the environment and install both runtime and development dependencies:

```bash
uv sync --group dev
```

The Phase-1 runtime dependencies are NumPy, pandas, PyArrow, and PyYAML. pytest and Ruff are
development dependencies.

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

The audit checks the complete configured dataset rather than relying on the preliminary counts in
this README. It validates paths, condition and bearing membership, numeric filename ordering,
sequence continuity, CSV schema and shape, numeric values, finite values, uniqueness constraints,
documented per-bearing acquisition counts, RUL invariants, and unexpected files or directories.
Duplicate-content checking is controlled by `detect_duplicate_content` in `configs/data.yaml`.

Critical validation errors gate downstream work. Do not extract features or train models when the
audit reports a critical error. Warnings and any explainable numbering irregularities must be
retained in the generated report rather than silently ignored.

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
traceability and future bearing-level splits. `rul_minutes` is explicitly a supervised target,
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

## RUL definition and leakage controls

For bearing `b`, acquisition index `t`, total acquisition count `N_b`, and nominal acquisition
interval `delta_t`, absolute RUL in minutes is defined as:

```text
RUL_b,t = (N_b - 1 - t) * delta_t
```

`t` is zero-based after validated numeric ordering. Therefore, the final acquisition has RUL zero.
A normalized life fraction may be used only for visualization and exploratory analysis.

The following leakage controls apply to all future modeling phases:

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
- Phases 1 and 2 establish data integrity, scalar vibration features, and exploratory figures;
  they provide no predictive RUL result.
- Conclusions must remain limited to the 15 XJTU-SY run-to-failure bearings and their three
  operating conditions.
