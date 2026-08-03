# Phase 8: probabilistic post-detection survival prognosis

Phase 8 asks whether probabilistic survival models provide useful conditional prognosis after the frozen Phase 7 causal estimated degradation onset. It does not redefine the onset, modify prior results, train neural survival models, or claim industrial generalization.

## Experimental protocol and leakage prevention

The five immutable complete-bearing folds are verified using both the physical SHA-256 and logical configuration hash. Training-only operations include feature filtering, survival relevance ranking, correlation pruning, scaling, balancing, model fitting, censoring-distribution estimation, evaluation-grid construction, and candidate fitting. Validation bearings select candidates by macro integrated Brier score. Test bearings are evaluated once after freezing. The evaluation-only landmark truth is protected by a runtime guard and can be opened only after every test survival prediction has been frozen.

The bearing is the independent unit. Dynamic landmarks are correlated repeated observations inside one bearing. Cox and Kaplan–Meier fitting therefore use deterministic equal-count sampling by training bearing. Random Survival Forest uses Phase 7 inverse-landmark-frequency weights, whose total is checked for equality across training bearings.

## Cohorts and scenarios

The static onset cohort is restricted to descriptive Kaplan–Meier and warning-time summaries. Predictive models use dynamic landmarks beginning at or after the confirmed causal alarm. Primary scenarios are `full_event`, `fixed_horizon_60`, `fixed_horizon_120`, and `fixed_horizon_240`. Secondary scenarios are `fixed_horizon_30` and the Phase 7 training-only target-censoring scenarios of 25%, 50%, and 75%.

## Feature reduction and models

Training-only filtering removes forbidden, non-numeric, invalid, constant, near-constant, exact-duplicate, and highly correlated columns. Absolute Spearman correlation above 0.95 triggers pruning; the feature with the stronger univariate survival concordance is retained, with feature name as deterministic tie-break. Candidate compact sets contain 5, 10, or 15 causal signal features and never exceed 20. Operational context is kept separate.

The models are a bearing-balanced landmark Kaplan–Meier baseline, context-only L2-regularized Cox, causal-feature L2-regularized Cox, and Random Survival Forest. Cox applies a training-only `RobustScaler`; RSF is not scaled or clipped. Model failures remain explicit candidate failures.

## Predictions and evaluation

Training outcomes construct the survival time grid and the censoring distribution used by `scikit-survival`. Stored outputs include full non-increasing survival curves, probabilities of survival and failure within 30, 60, 120, and 240 minutes, and a median survival time when the curve crosses 0.5. Missing medians remain missing and prediction coverage is reported.

Primary comparison uses bearing-macro integrated Brier score. Reports also contain time-dependent Brier scores, IPCW/Uno and descriptive Harrell concordance, calibration summaries, median-survival point errors and coverage. Cluster bootstrap resamples complete bearings within operating condition; landmarks are never independently resampled. Coverage of the causal detector is shown beside conditional prognostic quality.

The optional conservative detector analysis is explicitly post-hoc and can never replace the primary frozen detector. If the Phase 7 candidates cannot reconstruct causal test trajectories without rerunning Phase 7, it is skipped with a machine-readable reason.

## Limitations

There are only 15 independent bearings, five per controlled accelerated condition, with one bearing type and no official onset ground truth. Some primary Phase 7 alarms are very early, repeated landmarks are correlated, calibration support is limited, and there is no external validation. Feature importance and condition differences are descriptive associations, not causal mechanisms.
