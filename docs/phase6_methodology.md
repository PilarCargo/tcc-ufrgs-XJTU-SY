# Phase 6 methodology: statistical consolidation

Phase 6 is a reporting-only phase. It verifies input hashes and frozen folds, loads predictions
without alteration, and never invokes model training. A canonical acquisition key combines
condition, bearing, and acquisition number. True RUL, fold assignment, uniqueness, finiteness,
and the fifteen-test-bearing mapping are validated before analysis.

Native support preserves each model result. Pairwise support is the acquisition-key intersection
for one pre-specified contrast. Strict common support is the intersection across all primary
models and enables a unified ranking. Support sizes and per-bearing retention remain explicit.

The bearing is the inferential unit. For every pair, Phase 6 calculates one MAE and RMSE per
bearing and defines `delta = candidate MAE - reference MAE`; negative values favor the candidate.
Exact ties use `1e-9` minutes. A five-minute practical band is descriptive, not a formal
equivalence margin.

Uncertainty uses ten thousand paired bearing-level replicates. The primary bootstrap draws five
bearings with replacement within each of the three operating conditions; an unstratified bootstrap
is retained as sensitivity analysis. Intervals do not establish generalization beyond the observed
bearings. Ranking probabilities are descriptive strict-support bootstrap proportions.

Two-sided Wilcoxon signed-rank tests use the `wilcox` zero policy and report zero counts. Exact
sign tests use wins and losses only. Negative rank-biserial effects favor the candidate. Holm
correction applies to the pre-specified dummy-comparison family. Effect sizes, differences, and
confidence intervals receive precedence over p-values. Non-significance is not equivalence.

The condition-specific countdown has a separate predeclared three-hypothesis family: countdown
versus the constant median, selected-features-plus-time Ridge, and the single-acquisition LSTM.
Holm adjustment is performed within this new family. The earlier dummy-comparison family remains
unchanged, so its previously reported adjusted values retain their original multiplicity scope.

Condition results are dataset-specific because each condition contains only five bearings and
speed/load are combined. Life stage and estimated degradation onset are retrospective; onset is
not ground truth or an input. Feature triangulation preserves channel identity and reports rank
agreement without causal interpretation. Missing cost fields remain missing with a reason. Pareto
labels refer only to the recorded axes.

Every final table is exported to CSV, Parquet, Markdown, and LaTeX. Portuguese drafts remain
separate from the monograph, and quantitative claims point to generated tables through a
machine-readable manifest. Conclusions remain limited to the fifteen XJTU-SY bearings, absolute
RUL target, frozen folds, and evaluated representations.
