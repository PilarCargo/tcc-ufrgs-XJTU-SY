# XJTU-SY dataset structure and Phase-1 assumptions

## Scope and evidence level

The project uses the local XJTU-SY Bearing Dataset exclusively. No other dataset and no simulated
replacement data are permitted.

The counts below were initially established by read-only inspection and were subsequently
confirmed by the exhaustive Phase 1 audit of every acquisition. The validated audit parsed all
9,216 CSVs, checked 301,989,888 physical signal rows, and finished with status `PASSED`, zero
critical errors, and no duplicate-content groups. The complete procedure and final evidence are
documented in [phase1_methodology.md](phase1_methodology.md).

## Observed local layout

The configured raw root is the existing project-relative `Data/` directory:

```text
Data/
├── 35Hz12kN/
│   ├── Bearing1_1/  (1.csv ... 123.csv)
│   ├── Bearing1_2/  (1.csv ... 161.csv)
│   ├── Bearing1_3/  (1.csv ... 158.csv)
│   ├── Bearing1_4/  (1.csv ... 122.csv)
│   └── Bearing1_5/  (1.csv ... 52.csv)
├── 37.5Hz11kN/
│   ├── Bearing2_1/  (1.csv ... 491.csv)
│   ├── Bearing2_2/  (1.csv ... 161.csv)
│   ├── Bearing2_3/  (1.csv ... 533.csv)
│   ├── Bearing2_4/  (1.csv ... 42.csv)
│   └── Bearing2_5/  (1.csv ... 339.csv)
├── 40Hz10kN/
│   ├── Bearing3_1/  (1.csv ... 2538.csv)
│   ├── Bearing3_2/  (1.csv ... 2496.csv)
│   ├── Bearing3_3/  (1.csv ... 371.csv)
│   ├── Bearing3_4/  (1.csv ... 1515.csv)
│   └── Bearing3_5/  (1.csv ... 114.csv)
└── Introduction_to_XJTU-SY_Bearing_Dataset.pdf
```

The raw directory also contains `.DS_Store`, which is not a dataset acquisition and must be
reported or explicitly classified as an allowed non-data artifact by the audit policy. Raw files
must remain immutable.

## Observed acquisition counts

| Condition | Directory | Nominal rotation | Nominal load | Bearing | Observed CSVs |
|---:|---|---:|---:|---|---:|
| 1 | `35Hz12kN` | 35 Hz / 2,100 rpm | 12 kN | `Bearing1_1` | 123 |
| 1 | `35Hz12kN` | 35 Hz / 2,100 rpm | 12 kN | `Bearing1_2` | 161 |
| 1 | `35Hz12kN` | 35 Hz / 2,100 rpm | 12 kN | `Bearing1_3` | 158 |
| 1 | `35Hz12kN` | 35 Hz / 2,100 rpm | 12 kN | `Bearing1_4` | 122 |
| 1 | `35Hz12kN` | 35 Hz / 2,100 rpm | 12 kN | `Bearing1_5` | 52 |
| 2 | `37.5Hz11kN` | 37.5 Hz / 2,250 rpm | 11 kN | `Bearing2_1` | 491 |
| 2 | `37.5Hz11kN` | 37.5 Hz / 2,250 rpm | 11 kN | `Bearing2_2` | 161 |
| 2 | `37.5Hz11kN` | 37.5 Hz / 2,250 rpm | 11 kN | `Bearing2_3` | 533 |
| 2 | `37.5Hz11kN` | 37.5 Hz / 2,250 rpm | 11 kN | `Bearing2_4` | 42 |
| 2 | `37.5Hz11kN` | 37.5 Hz / 2,250 rpm | 11 kN | `Bearing2_5` | 339 |
| 3 | `40Hz10kN` | 40 Hz / 2,400 rpm | 10 kN | `Bearing3_1` | 2,538 |
| 3 | `40Hz10kN` | 40 Hz / 2,400 rpm | 10 kN | `Bearing3_2` | 2,496 |
| 3 | `40Hz10kN` | 40 Hz / 2,400 rpm | 10 kN | `Bearing3_3` | 371 |
| 3 | `40Hz10kN` | 40 Hz / 2,400 rpm | 10 kN | `Bearing3_4` | 1,515 |
| 3 | `40Hz10kN` | 40 Hz / 2,400 rpm | 10 kN | `Bearing3_5` | 114 |
|  |  |  |  | **Condition 1 subtotal** | **616** |
|  |  |  |  | **Condition 2 subtotal** | **1,566** |
|  |  |  |  | **Condition 3 subtotal** | **7,034** |
|  |  |  |  | **Total** | **9,216** |

At inspection time, each bearing directory contained one uniquely named CSV for every integer from
`1` through the count shown above. The audit must reproduce this observation and report any later
change or ambiguity.

## Acquisition schema

The exact observed CSV header is:

```text
Horizontal_vibration_signals,Vertical_vibration_signals
```

The configured expected acquisition shape is 32,768 sample rows by two vibration channels. A text
line count is therefore normally 32,769 when the header is included. Horizontal and vertical
channel values must parse as numeric and must contain no NaN or infinite values.

The expected shape is an audit assertion, not permission to truncate, pad, interpolate, or replace
an inconsistent raw file. Any mismatch must be reported clearly.

## Ordering, elapsed time, and RUL

Filenames must be ordered numerically by their integer stem:

```text
1.csv, 2.csv, 3.csv, ..., 10.csv, ...
```

Lexicographic ordering such as `1.csv, 10.csv, 100.csv, 2.csv` is incorrect. After validated
numeric ordering, the zero-based `sequence_index` is `t = 0, 1, ..., N_b - 1`. With nominal
acquisition interval `delta_t` in minutes:

```text
elapsed_minutes = t * delta_t
RUL_b,t = (N_b - 1 - t) * delta_t
```

This definition guarantees non-negative RUL and RUL equal to zero for the final acquisition.
Absolute RUL in minutes is the primary target. A normalized life fraction is permitted only for
exploration and visualization.

The lifetime `N_b` is used to construct the supervised target during dataset preparation. It must
never be supplied as a feature for a held-out bearing or used to normalize its input timeline.
Future data splits must be grouped by bearing, and every learned preprocessing transformation must
be fitted on training bearings only.

## Nominal values and provenance boundary

`configs/data.yaml` records these nominal expectations:

- Sampling frequency: 25,600 Hz.
- Samples per channel and acquisition: 32,768.
- Acquisition interval: 1.0 minute.
- Expected acquisition count for each of the 15 bearings, as listed above.
- Condition-specific rotational frequency, rpm, and radial load as shown in the table above.
- Provenance reference: `Data/Introduction_to_XJTU-SY_Bearing_Dataset.pdf`.

The CSV schema does not contain timestamps, sampling frequency, vibration units, acquisition
intervals, rpm, or radial load. The audit can verify sample counts, channel names, and numeric
content, but it cannot infer or independently prove those nominal quantities from the CSV files.
The `Hz` component of a condition directory is rotational frequency, not sampling frequency.

## Exhaustive Phase-1 audit

Install the project and development dependencies with:

```bash
uv sync --group dev
```

Then run:

```bash
uv run python -m xjtu_sy_tcc.cli.audit --config configs/data.yaml
```

The command performs exhaustive validation and writes Phase-1 metadata, JSON and Markdown audit
reports, and the per-bearing summary beneath `outputs/audits/`. At minimum, it must detect and
report:

- Missing, unexpected, or malformed condition and bearing directories.
- Missing, duplicate, nonnumeric, or ambiguously ordered acquisition filenames.
- Unexpected files and duplicate paths or configured sequence indices.
- Missing, reordered, inconsistent, or extra CSV columns.
- Unexpected sample or channel counts.
- Nonnumeric values, NaN values, and infinite values.
- Negative RUL, nonmonotonic indices, or a nonzero final RUL.
- Duplicate file content when configured.

Critical errors block all downstream feature extraction and modeling. An invalid raw file must
never be silently skipped, repaired in place, or replaced with simulated data.

The validated real-data audit completed with status `PASSED`. Its canonical machine-readable
evidence is `outputs/audits/dataset_audit.json`; the corresponding readable report is
`outputs/audits/dataset_audit.md`. See [phase1_methodology.md](phase1_methodology.md) for the
discovery, duplicate-detection, metadata, atomic-output, and validation methodology.

## Current methodological limitations

- Phase 1 contains no feature-extraction, change-point, health-indicator, or model implementation.
- XJTU-SY does not provide an official exact degradation-onset label; later PELT results are
  estimated change points, not true labels.
- The observed runs cover only 15 bearings under three accelerated test conditions.
- Dataset documentation supplies nominal timing and operating values that the CSVs cannot verify.
- Results from later phases cannot establish generalization to other bearings, machines, sensors,
  datasets, or real industrial environments.
