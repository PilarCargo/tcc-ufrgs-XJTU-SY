"""Complete no-training Phase 6 statistical consolidation pipeline."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from xjtu_sy_tcc.config.phase6 import Phase6Config
from xjtu_sy_tcc.consolidation.evidence import computational_costs, feature_evidence, pareto_front
from xjtu_sy_tcc.consolidation.harmonization import (
    canonicalize,
    pairwise_support,
    strict_common_support,
    support_manifest,
)
from xjtu_sy_tcc.consolidation.manuscript import write_manuscript
from xjtu_sy_tcc.consolidation.plotting import generate_figures
from xjtu_sy_tcc.consolidation.statistics import (
    bootstrap_difference,
    bootstrap_model_metrics,
    bootstrap_model_ranks,
    holm_adjust,
    paired_differences,
    paired_tests,
    summarize_difference,
)
from xjtu_sy_tcc.consolidation.tables import write_table_set
from xjtu_sy_tcc.rul.metrics import bearing_metrics, grouped_metrics
from xjtu_sy_tcc.rul.reporting import atomic_json, atomic_table, atomic_text
from xjtu_sy_tcc.rul.safety import load_frozen_folds


def run_phase6(config: Phase6Config, *, generate_plots=True, generate_manuscript=True, force=False):
    cache = _cache(config)
    if cache and not force:
        return cache
    started = time.perf_counter()
    p4 = pd.read_parquet(config.phase4_predictions)
    p5 = pd.read_parquet(config.phase5_predictions)
    features = p4.drop_duplicates(["condition_id", "bearing_id", "acquisition_number"])
    folds, split = load_frozen_folds(
        config.split_manifest,
        config.expected_split_sha256,
        config.expected_split_configuration_hash,
        features,
    )
    all_models = config.primary_models + config.secondary_models
    canonical = canonicalize(p4, p5, all_models)
    test_onsets = pd.read_parquet(config.test_onsets).query("subset == 'test'")
    if len(test_onsets) != 15 or test_onsets.bearing_id.nunique() != 15:
        raise ValueError("Exactly 15 frozen test-onset records are required")
    root = config.output_directory
    atomic_table(root / "harmonized/canonical_predictions.parquet", canonical)
    native_bearing = bearing_metrics(canonical)
    native = _model_metrics(canonical, native_bearing)
    strict_data = strict_common_support(canonical, config.primary_models)
    atomic_table(root / "harmonized/strict_common_support.parquet", strict_data)
    strict_bearing = bearing_metrics(strict_data)
    strict = _model_metrics(strict_data, strict_bearing)
    support = []
    differences = []
    summaries = []
    bootstraps = []
    tests = []
    all_contrasts = config.primary_contrasts + config.countdown_contrasts
    for index, (candidate, reference) in enumerate(all_contrasts):
        c, r = pairwise_support(canonical, candidate, reference)
        support.append(support_manifest(canonical, candidate, reference))
        diff = paired_differences(
            c, r, config.tie_tolerance_minutes, config.practical_tolerance_minutes
        )
        diff.insert(0, "reference", reference)
        diff.insert(0, "candidate", candidate)
        differences.append(diff)
        summary = {"candidate": candidate, "reference": reference, **summarize_difference(diff)}
        summaries.append(summary)
        for stratified in (True, False):
            bootstraps.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": "paired_mae_difference",
                    **bootstrap_difference(
                        diff,
                        config.bootstrap_replicates,
                        config.bootstrap_seed + index,
                        config.confidence_level,
                        stratified,
                    ),
                }
            )
        tests.append(
            {
                **summary,
                **paired_tests(diff, config.tie_tolerance_minutes),
                "family": (
                    "condition_countdown_predeclared"
                    if (candidate, reference) in config.countdown_contrasts
                    else "dummy_comparisons"
                    if reference == "dummy_median"
                    else "temporal_context"
                ),
            }
        )
    support_table = pd.concat(support, ignore_index=True)
    difference_table = pd.concat(differences, ignore_index=True)
    tests_table = pd.DataFrame(tests)
    tests_table["holm_p_adjusted"] = np.nan
    tests_table["holm_reject"] = False
    for family in ("dummy_comparisons", "condition_countdown_predeclared"):
        mask = tests_table.family.eq(family)
        corrected = holm_adjust(tests_table[mask].reset_index(drop=True), alpha=config.alpha)
        tests_table.loc[mask, "holm_p_adjusted"] = corrected.holm_p_adjusted.to_numpy()
        tests_table.loc[mask, "holm_reject"] = corrected.holm_reject.to_numpy()
    tests_table["family_hypothesis_count"] = tests_table.family.map(
        tests_table.groupby("family").size()
    )
    rank_prob = bootstrap_model_ranks(
        strict_bearing, config.primary_models, config.bootstrap_replicates, config.bootstrap_seed
    )
    bootstraps.extend(
        bootstrap_model_metrics(
            strict_bearing,
            config.primary_models,
            config.bootstrap_replicates,
            config.bootstrap_seed,
            config.confidence_level,
        ).to_dict("records")
    )
    native_bearing["native_support_count"] = native_bearing["acquisition_count"]
    native_bearing["matched_support_count"] = native_bearing["acquisition_count"]
    native_bearing["model_minus_dummy_mae_difference"] = np.nan
    native_bearing["win_tie_loss_status"] = pd.NA
    native_bearing["practical_tolerance_status"] = pd.NA
    native_bearing.loc[
        native_bearing.experiment == "dummy_median", "model_minus_dummy_mae_difference"
    ] = 0.0
    native_bearing.loc[native_bearing.experiment == "dummy_median", "win_tie_loss_status"] = (
        "reference"
    )
    native_bearing.loc[
        native_bearing.experiment == "dummy_median", "practical_tolerance_status"
    ] = "reference"
    dummy_differences = difference_table[difference_table.reference == "dummy_median"]
    dummy_support = support_table[support_table.reference == "dummy_median"]
    for row in dummy_differences.itertuples():
        mask = native_bearing.experiment.eq(row.candidate) & native_bearing.bearing_id.eq(
            row.bearing_id
        )
        native_bearing.loc[mask, "model_minus_dummy_mae_difference"] = row.delta_mae
        native_bearing.loc[mask, "win_tie_loss_status"] = row.exact_status
        native_bearing.loc[mask, "practical_tolerance_status"] = row.practical_status
    for row in dummy_support.itertuples():
        mask = native_bearing.experiment.eq(row.candidate) & native_bearing.bearing_id.eq(
            row.bearing_id
        )
        native_bearing.loc[mask, "matched_support_count"] = row.matched_count
    condition = grouped_metrics(strict_data, ["experiment", "model", "condition_id"])
    life = grouped_metrics(strict_data, ["experiment", "model", "life_stage"])
    onset = grouped_metrics(strict_data, ["experiment", "model", "estimated_onset_region"])
    prognostic = pd.read_parquet(config.phase3_rankings)
    classical = pd.read_csv(config.phase4_importance)
    temporal = pd.read_csv(config.phase5_importance)
    evidence, consensus, agreement = feature_evidence(prognostic, classical, temporal)
    costs = computational_costs(
        pd.read_parquet(config.phase4_costs), pd.read_parquet(config.phase5_costs)
    )
    costs = costs.merge(native[["experiment", "macro_mae"]], on="experiment", how="left")
    costs = pareto_front(costs)
    tables = {
        "primary_native_support_model_comparison": native[
            native.experiment.isin(config.primary_models)
        ],
        "native_support_model_comparison": native,
        "strict_common_support_comparison": strict,
        "pairwise_matched_support_contrasts": pd.DataFrame(summaries),
        "bootstrap_confidence_intervals": pd.DataFrame(bootstraps),
        "paired_tests": tests_table,
        "win_tie_loss": pd.DataFrame(summaries)[
            ["candidate", "reference", "wins", "ties", "losses"]
        ],
        "per_bearing_metrics": native_bearing,
        "condition_metrics": condition,
        "life_stage_metrics": life,
        "onset_region_metrics": onset,
        "feature_evidence_consensus": consensus,
        "computational_cost_comparison": costs,
        "methodological_limitations": pd.DataFrame(
            {
                "limitation": [
                    "Fifteen independent bearings",
                    "Five bearings per operating condition",
                    "No official degradation-onset ground truth",
                    "No external validation",
                    "Temporal models have dropped-prefix support",
                ]
            }
        ),
    }
    for name, table in tables.items():
        write_table_set(root / "tables", name, table, config.decimal_precision)
    outputs = {
        "harmonized/pairwise_support_manifest.parquet": support_table,
        "metrics/native_support_metrics.parquet": native,
        "metrics/matched_support_metrics.parquet": pd.DataFrame(summaries),
        "metrics/strict_common_support_metrics.parquet": strict,
        "metrics/per_bearing_metrics.parquet": native_bearing,
        "metrics/condition_metrics.parquet": condition,
        "metrics/life_stage_metrics.parquet": life,
        "metrics/onset_region_metrics.parquet": onset,
        "statistics/paired_differences.parquet": difference_table,
        "statistics/bootstrap_results.parquet": pd.DataFrame(bootstraps),
        "statistics/paired_tests.parquet": tests_table,
        "statistics/multiple_comparison_results.parquet": tests_table,
        "statistics/model_rank_probabilities.parquet": rank_prob,
        "statistics/win_tie_loss.csv": tables["win_tie_loss"],
        "features/harmonized_feature_evidence.parquet": evidence,
        "features/feature_agreement.parquet": agreement,
        "features/feature_consensus.csv": consensus,
        "computational/computational_costs.parquet": costs,
    }
    for relative, table in outputs.items():
        atomic_table(root / relative, table)
    claim_count = (
        write_manuscript(
            root / "manuscript",
            native,
            strict,
            tests_table,
            condition,
            life,
            onset,
            consensus,
            costs,
        )
        if generate_manuscript
        else 0
    )
    figures = (
        generate_figures(
            config,
            native,
            strict,
            difference_table,
            tests_table,
            native_bearing,
            condition,
            life,
            onset,
            consensus,
            costs,
            canonical,
        )
        if generate_plots
        else 0
    )
    input_hashes = {
        name: _sha(getattr(config, name))
        for name in (
            "split_manifest",
            "phase4_predictions",
            "phase5_predictions",
            "phase3_rankings",
            "phase4_importance",
            "phase5_importance",
            "phase4_costs",
            "phase5_costs",
            "test_onsets",
        )
    }
    manifest = {
        "configuration_hash": config.configuration_hash,
        "input_hashes": input_hashes,
        "models": list(all_models),
        "canonical_key": ["condition_id", "bearing_id", "acquisition_number"],
    }
    atomic_json(root / "inputs/input_manifest.json", manifest)
    validation = _validate(
        config,
        canonical,
        strict_data,
        difference_table,
        tests_table,
        support_table,
        claim_count,
        folds,
        require_claims=generate_manuscript,
    )
    atomic_json(root / "inputs/integrity_report.json", validation)
    report = {
        **validation,
        "configuration_hash": config.configuration_hash,
        "input_hashes": input_hashes,
        "split_configuration_hash": split["configuration_hash"],
        "duration_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "canonical_prediction_rows": len(canonical),
        "strict_common_acquisition_count": strict_data.canonical_key.nunique(),
        "figure_files": figures,
        "manuscript_claims": claim_count,
        "primary_results": tables["primary_native_support_model_comparison"].to_dict("records"),
        "paired_tests": tests_table.to_dict("records"),
    }
    atomic_json(
        root / "configuration/phase6_resolved_config.json",
        asdict(config) | {"configuration_hash": config.configuration_hash},
    )
    atomic_json(root / "reports/phase6_validation.json", report)
    atomic_json(root / "reports/phase6_results.json", report)
    atomic_text(root / "reports/phase6_results.md", _markdown(report))
    return report


def _model_metrics(data, bearing):
    global_ = grouped_metrics(data, ["experiment", "model"]).rename(
        columns={
            "mae": "global_mae",
            "rmse": "global_rmse",
            "r2": "global_r2",
            "signed_error": "global_signed_error",
        }
    )
    macro = bearing.groupby(["experiment", "model"], as_index=False).agg(
        macro_mae=("mae", "mean"),
        macro_rmse=("rmse", "mean"),
        median_bearing_mae=("mae", "median"),
        bearing_mae_q1=("mae", lambda x: x.quantile(0.25)),
        bearing_mae_q3=("mae", lambda x: x.quantile(0.75)),
    )
    return global_.merge(macro).sort_values("macro_mae")


def _validate(
    config, canonical, strict, differences, tests, support, claims, folds, *, require_claims=True
):
    issues = []
    if canonical.bearing_id.nunique() != 15:
        issues.append("not exactly 15 bearings")
    if strict.groupby("experiment").canonical_key.nunique().nunique() != 1:
        issues.append("strict support differs by model")
    if not ((differences.groupby(["candidate", "reference"]).size() == 15).all()):
        issues.append("paired comparisons do not contain 15 bearings")
    if (
        not tests[["wilcoxon_p_raw", "sign_test_p_raw"]]
        .apply(lambda x: x.between(0, 1).all())
        .all()
    ):
        issues.append("invalid p-values")
    if not np.allclose(support.matched_count > 0, True):
        issues.append("empty bearing support")
    if require_claims and claims < 1:
        issues.append("no traceable quantitative claims")
    return {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "model_training_performed": False,
        "test_bearings": canonical.bearing_id.nunique(),
        "fold_count": len(folds),
    }


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache(config):
    path = config.output_directory / "reports/phase6_validation.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    return (
        report
        if report.get("status") == "passed"
        and report.get("configuration_hash") == config.configuration_hash
        else None
    )


def _markdown(report):
    return "\n".join(
        [
            "# Phase 6 statistical consolidation",
            "",
            f"- Status: **{report['status'].upper()}**",
            f"- No model training performed: **{not report['model_training_performed']}**",
            f"- Strict common support: {report['strict_common_acquisition_count']} acquisitions",
            f"- Figure files: {report['figure_files']}",
            f"- Traceable manuscript claims: {report['manuscript_claims']}",
            "",
        ]
    )
