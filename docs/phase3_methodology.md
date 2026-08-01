# Phase 3 methodology: prognostic features and estimated degradation onset

## Scientific status

Phase 3 uses only the 15 XJTU-SY run-to-failure bearings. XJTU-SY has no official timestamp or
label for degradation onset. Every breakpoint is therefore an **estimated degradation onset**,
never ground truth or a supervised class label. Normalized life position is used only for offline
trajectory comparison, balancing, and descriptive figures; it is not a predictive input. Phase 3
does not train an RUL regression model.

## Complete-bearing folds

Five deterministic folds use suffix positions 1 through 5. In fold `i`, suffix `i` is test,
suffix `(i mod 5) + 1` is validation, and the other three suffixes per condition are training.
Every fold therefore has nine train, three validation, and three test bearings, with one
validation and one test bearing per condition. Acquisitions are never randomly split.

## Prognostic metrics

For a feature trajectory with first differences `d`:

- monotonicity is `abs(n(d>0) - n(d<0)) / (n(d>0) + n(d<0))`; exact ties are excluded and counted;
- trend sign is the sign of `n(d>0) - n(d<0)`;
- temporal association is signed Spearman correlation with normalized sequence position;
- fluctuation is median absolute first difference divided by trajectory IQR;
- trendability interpolates without extrapolation onto a configurable `[0,1]` grid, calculates
  absolute Pearson correlation for every valid bearing pair, and reports their median and count;
- prognosability is `exp(-MAD(final medians) / median(abs(final-initial)))`, using robust initial
  and final window medians for each complete trajectory.

Constant-trajectory Spearman correlation and monotonicity are zero. Trendability omits pairs
containing a constant trajectory. Prognosability returns one for identical constant endpoints;
other zero-change cases return zero.

The all-dataset ranking is post-hoc descriptive. Fold rankings recompute every aggregate from the
nine training bearings only. Components are min-max normalized within the fold, combined with
configured weights, and exact ties are resolved by feature name.

## Leakage-free health indicator

Each fold selects ten features from training data. Every training bearing is linearly
interpolated, without extrapolation, to 100 positions, so long trajectories cannot dominate the
fit. These 900 balanced rows fit all transformations, `RobustScaler`, and PCA.

The default applies `log1p` only to selected features whose configured names denote verified
non-negative heavy-tailed quantities. Negative input fails execution. No clipping,
winsorization, or outlier deletion is applied. PCA keeps three components. PC1 orientation is
frozen from the sign of the median training-bearing Spearman correlation between PC1 and temporal
progression, so increasing values indicate worsening condition.

Every bearing, including validation and test, uses its own first 10% as an explicitly declared
initial calibration interval. Its robust median centers PC1. This is an initial calibration
assumption, not future information. Raw, centered, and trailing rolling-median indicators are all
retained. A breakpoint cannot precede calibration.

## PELT and candidate filtering

`ruptures.Pelt` supports `rbf` and `l2`. Penalty, minimum segment size, jump, smoothing,
persistence, and robust effect threshold are declarative. A PELT boundary is valid only when it:

1. follows the calibration interval;
2. has complete pre- and post-change persistence windows;
3. has a positive median level shift divided by the robust dispersion of the full initial
   calibration interval above the threshold;
4. keeps at least 70% of its post-change persistence window above the pre-change median.

The earliest qualifying deterioration is selected. Otherwise status is `not_detected`, with
rejection reasons. No breakpoint is forced.

## Sensitivity and configuration freezing

The parameter grid is evaluated on training and validation bearings only. Configurations are
ranked per fold by:

`0.35*detection_rate + 0.25*grid_median_agreement + 0.15*median_clipped_effect +`
`0.15*condition_coverage + 0.10*(1-late_rate)`.

`grid_median_agreement` is the proportion of onsets within ten minutes of each bearing's median
across the grid; `late_rate` is detection frequency in the final 5% of life. Parameter ID breaks
ties. The chosen
configuration is frozen before test application. Stability outputs include median onset, onset
IQR in minutes, normalized life-fraction IQR, detection proportion, and agreement within minute
and life-fraction tolerances. These are not accuracy metrics because no ground truth exists.

## Main artifacts

- `outputs/splits/folds.json` and `folds.md`;
- `outputs/prognostics/feature_scores_by_bearing.parquet`;
- `outputs/prognostics/feature_scores_by_condition.parquet`;
- `outputs/prognostics/fold_feature_rankings.parquet`;
- `outputs/prognostics/selected_features_by_fold.json`;
- `outputs/degradation/preprocessing/fold_<n>.json`;
- `outputs/degradation/health_indicators.parquet`;
- `outputs/degradation/pelt_sensitivity.parquet`;
- `outputs/degradation/estimated_onsets.parquet` and `.csv`;
- JSON/Markdown validation reports and publication-ready PNG/PDF figures.
