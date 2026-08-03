# Phase 7 methodology: causal degradation and survival datasets

Phase 7 responds to the weak full-trajectory absolute-RUL results by separating monitoring from
post-alarm prognosis. It trains no survival model. The complete-bearing folds and hashes remain
frozen, and all test parameters are selected with training and validation bearings only.

Each raw acquisition is transformed into separate horizontal and vertical Welch PSDs using the
validated Phase 2 settings. Power from 0–12.8 kHz is integrated into 32 deterministic bins,
regularized by a positive epsilon, and normalized to a probability distribution. This cache is
separate from the unchanged 52-feature table.

Each bearing calibrates from a fixed initial count, never a percentage of final life. Channel
references are calibration medians. Both `KL(P||Q)` and symmetric KL are available. Divergence is
combined across channels, aggregated with a strictly trailing window, and robustly normalized by
the calibration median and MAD. Zero MAD uses the configured epsilon.

The causal alarm is the first time persistence can be confirmed from current and past values.
The confirmation time is operational; the earlier threshold crossing is retained separately. An
isolated impulse, insufficient post-alarm follow-up, or insufficient effect returns
`not_detected`; PELT is never a fallback. Parameter ranking rewards validation coverage while
penalizing immediate post-calibration and final-only alarms. Retrospective PELT comparison is
descriptive agreement, not onset accuracy, because no official onset ground truth exists.

Fold-selected Phase 2 features generate only trailing descriptors: calibration differences and
robust ratios, first differences, rolling mean/median/standard deviation, slopes, and EWMA.
Landmarks begin at the confirmed causal alarm and stay within one bearing. Multiple landmarks
from a bearing are correlated and must not be treated as independent subjects.

The full-event cohort uses the observed run-to-failure endpoint. Fixed 30, 60, 120, and 240 minute
horizons create administrative censoring. Target-rate horizons are quantiles of training landmark
durations and are frozen for validation and test. Future truth is isolated under `evaluation/` and
listed in the forbidden-column manifest. Bearing weights equalize total landmark contribution.

Results remain limited to the 15 XJTU-SY bearings, one bearing type, three combined conditions,
and controlled accelerated tests. A Phase 8 survival model may operate only after a causal alarm,
must retain complete-bearing separation, and must never load the evaluation-only truth artifact.
