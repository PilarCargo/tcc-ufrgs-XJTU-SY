# XJTU-SY bearing prognostics thesis project

This repository supports an undergraduate Mechanical Engineering thesis on predictive
maintenance of rolling bearings. Its objective is to build a reproducible experimental pipeline
for degradation analysis, health-indicator construction, unsupervised change-point estimation,
and Remaining Useful Life (RUL) regression.

The scientific scope is deliberately narrow: **only the XJTU-SY Bearing Dataset may be used**.
The project must not substitute simulated data, add fault-type classification to the primary
scope, or claim generalization to industrial environments outside XJTU-SY.

## Current status

The current implementation is Phase 1: dataset discovery, loading, metadata construction, and
audit reporting. Feature extraction, health indicators, PELT change-point estimation, classical
machine-learning models, temporal deep learning, and model evaluation are not implemented yet.

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
- Phase 1 establishes data integrity and metadata only; it provides no predictive result.
- Conclusions must remain limited to the 15 XJTU-SY run-to-failure bearings and their three
  operating conditions.
