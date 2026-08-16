# Phase 1 methodology: dataset integrity and acquisition metadata

## Objective and scope

Phase 1 establishes a reproducible integrity boundary for the local XJTU-SY Bearing Dataset.
It discovers the complete directory hierarchy, validates every acquisition, constructs
acquisition-level metadata, and calculates absolute Remaining Useful Life (RUL). It performs no
feature extraction, degradation detection, or predictive modeling. Raw files are treated as
immutable evidence and are never repaired, truncated, interpolated, or replaced.

The only dataset used is XJTU-SY. The configured source is `Data/`, and the nominal operating
values are traced to `Data/Introduction_to_XJTU-SY_Bearing_Dataset.pdf`. Sampling frequency,
acquisition interval, rotation, load, and vibration units cannot be inferred independently from
the amplitude-only CSV schema; the audit distinguishes these documented nominal values from
properties directly observable in the files.

## Configured dataset contract

`configs/data.yaml` declares:

- three operating conditions and five bearings per condition;
- 9,216 expected acquisitions;
- 25,600 Hz nominal sampling frequency;
- 32,768 samples per channel and acquisition;
- two ordered channels, `Horizontal_vibration_signals` and
  `Vertical_vibration_signals`;
- a nominal acquisition interval of 1.0 minute;
- the exact expected acquisition count for every bearing;
- the filename pattern `^[1-9][0-9]*\.csv$`;
- condition-specific rotation and radial load;
- four inspection workers and exact duplicate-content detection.

The three operating conditions are 2,100 rpm/12 kN, 2,250 rpm/11 kN, and 2,400 rpm/10 kN.
These are known operating-context values rather than vibration-derived estimates.

## Dataset discovery and chronological ordering

Discovery validates the configured condition directories, bearing identifiers, file extensions,
and expected file counts. Acquisition chronology is defined by parsing the integer filename stem.
Therefore, `2.csv` precedes `10.csv`; lexicographic ordering is explicitly rejected.

For a bearing with (N_b) acquisitions, the ordered acquisitions receive the zero-based index

\[
t = 0, 1, \ldots, N_b-1.
\]

Missing, duplicated, nonnumeric, or ambiguous acquisition identifiers are critical errors.
Unexpected entries are preserved as structured validation issues instead of being silently
discarded.

## Exhaustive CSV inspection

Every discovered CSV is parsed and checked for:

- the exact channel names and channel order;
- exactly two numeric channels;
- exactly 32,768 signal rows;
- numeric parseability;
- absence of NaN and infinite values;
- file-size, modification-time, device, and inode stability during the audit.

The filesystem-identity checks detect replacement or mutation of an acquisition while the audit
is running. A failure in one file is recorded as a structured issue and does not prevent the
remaining files from being inspected, but any critical issue makes the complete audit fail.

## Exact duplicate-content detection

Duplicate detection uses a deterministic staged procedure to avoid hashing every complete file
unnecessarily:

1. group files by identical byte size;
2. calculate a SHA-256 boundary fingerprint for files in equal-size groups;
3. calculate a full-file SHA-256 only for remaining fingerprint candidates;
4. report groups only after exact full-content confirmation.

This procedure is an optimization of exact equality checking, not an approximate duplicate
criterion. In the validated run, 1,159 equal-size groups led to 2,632 boundary fingerprints and
no file required full-hash confirmation because no boundary-fingerprint collision remained.
No duplicate-content group was detected.

## Metadata, elapsed time, and RUL

For the configured acquisition interval \(\Delta t=1\) minute:

\[
\text{elapsed}_{b,t}=t\Delta t,
\]

\[
\operatorname{RUL}_{b,t}=(N_b-1-t)\Delta t.
\]

This produces nonnegative RUL and guarantees zero RUL at the final acquisition. The generated
metadata contains 15 columns: condition and bearing identifiers, portable file path and name,
acquisition number, sequence index, elapsed minutes, total acquisitions, RUL minutes, rotation,
load, sampling frequency, sample count, channel count, and file size.

`total_acquisitions` is needed to construct the supervised target but is future-derived. It is
retained only in Phase 1 metadata and must never enter a predictive feature matrix for a held-out
bearing. Absolute `rul_minutes` is a target, never a model input.

## Dataset-level invariants

After metadata construction, the audit validates:

- unique file paths;
- continuous zero-based sequence indices;
- strictly increasing acquisition numbers;
- elapsed time equal to sequence index times the configured interval;
- RUL equal to the declared formula;
- finite, nonnegative RUL values;
- final RUL equal to zero;
- a consistent total-acquisition count within each bearing.

Critical errors prevent publication of validated metadata and block all downstream phases.

## Atomic outputs and execution safety

The command is:

```bash
uv run xjtu-sy-audit --config configs/data.yaml
```

An exclusive output lock prevents concurrent audits from writing to the same directory. CSV,
JSON, Markdown, and Parquet artifacts are written atomically. Validated tables are published only
when the audit passes; a failed or interrupted publication invalidates the known final artifacts.
Structured JSON logging records discovery, inspection progress, completion status, counts, and
duration.

The principal outputs are:

- `outputs/audits/metadata.parquet`;
- `outputs/audits/bearing_summary.parquet`;
- `outputs/audits/bearing_summary.csv`;
- `outputs/audits/dataset_audit.json`;
- `outputs/audits/dataset_audit.md`.

## Validated results

The exhaustive real-data audit produced:

- status `PASSED`;
- 3 conditions and 15 bearings;
- 9,216 acquisitions;
- 301,989,888 physical signal rows;
- 12,219,567,144 raw acquisition bytes;
- zero CSV inspection errors;
- zero critical validation errors;
- zero exact duplicate-content groups;
- one non-critical warning for the ignored `Data/.DS_Store` file;
- 136.895 seconds total execution time in the reference run.

Every bearing ended with RUL equal to zero, and downstream processing was explicitly authorized
by the machine-readable audit report.

## Methodological limitations

- The CSV files contain no timestamps or operating-condition fields, so nominal timing, sampling,
  rotation, and load depend on dataset documentation.
- RUL is calculated from the final acquisition count and is therefore suitable as a supervised
  target, not a deployable input.
- The audit verifies integrity, not measurement calibration or sensor accuracy.
- XJTU-SY contains only 15 run-to-failure bearings under accelerated controlled conditions.
- Passing the audit does not demonstrate external or industrial generalization.

