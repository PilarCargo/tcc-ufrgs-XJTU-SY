"""Feature-evidence and computational-cost consolidation without fabrication."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def feature_evidence(prognostic, classical, temporal):
    p = prognostic[["fold", "feature", "rank", "composite_score", "selected"]].copy()
    p["source"] = "Phase 3 prognostic"
    p["score"] = p.composite_score
    counts = p.groupby("fold").feature.transform("count")
    p["normalized_rank"] = 1 - (p["rank"] - 1) / (counts - 1)
    c = classical.copy()
    c["source"] = "Phase 4 classical"
    c["score"] = c.permutation_mean_macro_mae_increase
    c["fold"] = pd.NA
    c["selected"] = c.permutation_fold_count > 0
    c["normalized_rank"] = c.score.rank(ascending=False, method="average").transform(
        lambda x: 1 - (x - 1) / (len(x) - 1)
    )
    t = temporal.groupby("feature", as_index=False).importance_macro_mae_increase.mean()
    t["source"] = "Phase 5 LSTM occlusion"
    t["score"] = t.importance_macro_mae_increase
    t["fold"] = pd.NA
    t["selected"] = True
    t["normalized_rank"] = t.score.rank(ascending=False, method="average").transform(
        lambda x: 1 - (x - 1) / (len(x) - 1)
    )
    evidence = pd.concat(
        [
            p[["fold", "feature", "source", "score", "normalized_rank", "selected"]],
            c[["fold", "feature", "source", "score", "normalized_rank", "selected"]],
            t[["fold", "feature", "source", "score", "normalized_rank", "selected"]],
        ],
        ignore_index=True,
    )
    consensus = (
        evidence.groupby("feature", as_index=False)
        .agg(
            mean_normalized_rank=("normalized_rank", "mean"),
            median_normalized_rank=("normalized_rank", "median"),
            evidence_sources=("source", "nunique"),
            selected_occurrences=("selected", "sum"),
        )
        .sort_values("mean_normalized_rank", ascending=False)
    )
    agreements = []
    sources = sorted(evidence.source.unique())
    ranks = {
        s: evidence[evidence.source == s].groupby("feature").normalized_rank.mean() for s in sources
    }
    for i, a in enumerate(sources):
        for b in sources[i + 1 :]:
            common = ranks[a].index.intersection(ranks[b].index)
            rho = (
                float(spearmanr(ranks[a][common], ranks[b][common]).statistic)
                if len(common) >= 3
                else np.nan
            )
            for k in (5, 10):
                sa = set(ranks[a].nlargest(k).index)
                sb = set(ranks[b].nlargest(k).index)
                agreements.append(
                    {
                        "source_a": a,
                        "source_b": b,
                        "top_k": k,
                        "common_features": len(common),
                        "spearman_rank_agreement": rho,
                        "intersection": len(sa & sb),
                        "jaccard": len(sa & sb) / len(sa | sb),
                    }
                )
    return evidence, consensus, pd.DataFrame(agreements)


def computational_costs(classical, temporal):
    c = classical.groupby("experiment", as_index=False).agg(
        training_seconds=("training_seconds", "sum"),
        preprocessing_seconds=("preprocessing_fit_seconds", "sum"),
        inference_seconds=("test_inference_seconds", "sum"),
        inference_seconds_per_item=("test_inference_seconds_per_acquisition", "mean"),
        artifact_size_bytes=("artifact_size_bytes", "sum"),
        input_feature_count=("input_feature_count", "mean"),
        training_items=("training_observations", "sum"),
    )
    c["trainable_parameters"] = np.nan
    c["sequence_length"] = np.nan
    c["missing_reason"] = "trainable parameter count and sequence length not applicable/recorded"
    t = temporal.groupby("experiment", as_index=False).agg(
        training_seconds=("candidate_training_seconds", "sum"),
        preprocessing_seconds=("preprocessing_fit_seconds", "sum"),
        inference_seconds=("test_inference_seconds", "sum"),
        inference_seconds_per_item=("inference_seconds_per_sequence", "mean"),
        artifact_size_bytes=("checkpoint_size_bytes", "sum"),
        trainable_parameters=("trainable_parameters", "mean"),
        sequence_length=("sequence_length", "mean"),
        training_items=("training_sequences", "sum"),
    )
    t["input_feature_count"] = np.nan
    t["missing_reason"] = "input feature count unavailable in Phase 5 cost artifact"
    return pd.concat([c, t], ignore_index=True)


def pareto_front(table, error="macro_mae"):
    result = table.copy()
    result["pareto_error_training_time"] = False
    for index, row in result.iterrows():
        dominated = (
            (result[error] <= row[error])
            & (result.training_seconds <= row.training_seconds)
            & ((result[error] < row[error]) | (result.training_seconds < row.training_seconds))
        ).any()
        result.loc[index, "pareto_error_training_time"] = not dominated
    return result
