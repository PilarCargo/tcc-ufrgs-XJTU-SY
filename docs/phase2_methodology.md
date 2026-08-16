# Phase 2 methodology: acquisition-level vibration features

## Objective and input boundary

Phase 2 converts each validated two-channel vibration acquisition into a compact, interpretable
feature row. It consumes only `outputs/audits/metadata.parquet` from the passed Phase 1 audit and
the corresponding immutable raw CSV. It does not create experimental folds, fit preprocessing,
select prognostic features, estimate degradation onset, or train a predictive model.

Each acquisition contains 32,768 horizontal and vertical samples. Signals are parsed as
`float32` to control memory, while scalar accumulations and returned descriptors use `float64`.
One acquisition is loaded at a time.

## Output schema

The final table contains 9,216 rows and 65 columns:

- 13 identifier, traceability, context, and target columns;
- 26 horizontal vibration features;
- 26 vertical vibration features.

Each channel contributes 15 time-domain features, seven general frequency-domain features, and
four spectral-band powers. `rul_minutes` is retained for later supervised evaluation but is never
classified as a vibration feature or predictive input by Phase 2.

## Time-domain features

Let an acquisition channel be \(x_1,\ldots,x_n\), with population mean \(\mu\), centered values
\(z_i=x_i-\mu\), and absolute values \(a_i=|x_i|\). Population moments use \(n\), not \(n-1\):

\[
\mu=\frac{1}{n}\sum_i x_i,
\qquad
\sigma^2=\frac{1}{n}\sum_i z_i^2,
\qquad
\sigma=\sqrt{\sigma^2}.
\]

The remaining descriptors are:

\[
\operatorname{RMS}=\sqrt{\frac{1}{n}\sum_i x_i^2},
\qquad
E=\sum_i x_i^2,
\]

\[
\operatorname{skewness}=\frac{n^{-1}\sum_i z_i^3}{(\sigma^2)^{3/2}},
\qquad
\operatorname{kurtosis}=\frac{n^{-1}\sum_i z_i^4}{(\sigma^2)^2}.
\]

Kurtosis is Pearson kurtosis, not excess kurtosis. Minimum, maximum, maximum absolute amplitude,
and peak-to-peak amplitude are also retained. The four amplitude ratios are:

\[
\text{crest factor}=\frac{\max(a_i)}{\operatorname{RMS}},
\]

\[
\text{shape factor}=\frac{\operatorname{RMS}}{n^{-1}\sum_i a_i},
\qquad
\text{impulse factor}=\frac{\max(a_i)}{n^{-1}\sum_i a_i},
\]

\[
\text{clearance factor}=\frac{\max(a_i)}{\left(n^{-1}\sum_i\sqrt{a_i}\right)^2}.
\]

A zero variance makes skewness and kurtosis undefined. A zero denominator makes the associated
ratio undefined. The implementation returns NaN in these cases rather than inventing zero; the
final validation then fails because production features must be finite.

## Welch power spectral density

Frequency-domain descriptors use SciPy's Welch estimator with the exact declared settings:

- sampling frequency: 25,600 Hz from the validated manifest;
- Hann window;
- segment length `nperseg = 2048`;
- overlap `noverlap = 1024`;
- constant detrending per segment;
- density scaling;
- arithmetic averaging of segment periodograms.

Welch averaging reduces the variance of a single periodogram while preserving a reproducible
description of how vibration power is distributed over frequency. The Nyquist frequency is
12,800 Hz; configurations exceeding it are rejected.

Let \(P_k\) be the PSD at frequency \(f_k\), and let
\(p_k=P_k/\sum_jP_j\). The seven general descriptors are:

- dominant frequency: frequency at \(\arg\max_kP_k\);
- dominant spectral power: \(\max_kP_k\);
- spectral centroid: \(\sum_k f_kp_k\);
- spectral spread: \(\sqrt{\sum_k(f_k-\bar f)^2p_k}\);
- RMS frequency: \(\sqrt{\sum_k f_k^2p_k}\);
- normalized spectral entropy:
  \(-\sum_{p_k>0}p_k\log(p_k)/\log(K)\);
- total spectral power: trapezoidal integration of the PSD over frequency.

The normalized entropy lies in `[0, 1]`. Zero total discrete PSD probability makes centroid,
spread, RMS frequency, and entropy undefined and therefore invalid in the production table.

## Spectral-band powers

PSD power is integrated by the trapezoidal rule in four configured bands:

| Band | Frequency interval |
|---|---:|
| Low | 0–1,000 Hz |
| Mid | 1,000–5,000 Hz |
| High | 5,000–10,000 Hz |
| Very high | 10,000–12,800 Hz |

These bands provide a coarse, interpretable representation of spectral redistribution without
claiming that they isolate bearing-defect frequencies. No shaft or bearing-geometry frequency is
inferred beyond the information supplied by the dataset.

## Incremental processing and restartability

The command is:

```bash
uv run xjtu-sy-build-features --config configs/features.yaml
```

The metadata manifest is processed in its validated chronological order. For every acquisition,
the pipeline checks that the resolved file remains inside the project root, verifies its expected
shape and finite numeric values, computes both channels, retains only scalar results, and releases
the signal array before continuing.

An atomic `.features.checkpoint.parquet` is published periodically. Restart is allowed only when
the checkpoint schema and its acquisition-key prefix exactly match the current manifest. The
`--force` option deliberately rejects checkpoint reuse. An exclusive lock prevents concurrent
feature builders from publishing into the same output directory.

Final Parquet, CSV, and JSON files use temporary files followed by atomic replacement. A failed
validation publishes the validation report but never publishes a feature table as valid.

## Validation

The completed table is checked for:

- exact row-count agreement with the Phase 1 manifest;
- one-to-one acquisition-key alignment;
- missing or unexpected acquisitions;
- duplicate acquisition rows;
- presence of every required feature;
- finite values in all 52 vibration features;
- nonnegative energy, power, RMS, standard deviation, variance, and spread;
- entropy within `[0, 1]`;
- frequency descriptors within `[0, Nyquist]`;
- chronological sequence and elapsed-time ordering;
- nonnegative RUL and zero final RUL for every bearing.

## Figures and outputs

The principal outputs are:

- `outputs/features/features.parquet`;
- `outputs/features/feature_validation.json`;
- `outputs/features/feature_summary.csv`;
- `outputs/figures/exploratory/condition_<id>/<bearing>/`.

For every bearing, the exploratory outputs include unsmoothed RMS, kurtosis, crest factor, energy,
and RUL trajectories. Representative first, middle, and final acquisitions receive combined
time-series and PSD figures. PNG uses 200 DPI, and PDF preserves vector graphics where configured.
Smoothing is not applied to these exploratory feature trajectories.

## Validated results

The real-data Phase 2 execution produced:

- validation status `PASSED`;
- 9,216 feature rows;
- 65 total columns;
- 52 vibration features;
- zero duplicate acquisition rows;
- zero missing, NaN, or infinite feature values;
- zero validation errors or warnings;
- 132.316 seconds extraction time;
- 340,787,200-byte peak resident memory (approximately 325 MiB);
- 240 exploratory figures in the validated project state.

The output preserves every Phase 1 acquisition key and is the canonical input to the later
degradation, RUL, causal-detection, and survival phases.

## Methodological limitations

- Handcrafted scalar features compress each 32,768-sample signal and may discard transient or
  phase information.
- The four broad spectral bands are deliberately generic and are not defect-frequency diagnosis.
- Welch parameters trade frequency resolution against estimator stability and were not optimized
  against held-out RUL results.
- Nominal sampling and operating values inherit the Phase 1 provenance limitation.
- Extreme late-life values are retained when finite and physically plausible; Phase 2 does not
  classify them as outliers or remove them.
- Feature validity does not by itself demonstrate prognostic usefulness or external
  generalization; those questions are evaluated in later phases.

