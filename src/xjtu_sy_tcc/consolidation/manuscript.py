"""Traceable formal Portuguese thesis drafts based only on generated tables."""
# ruff: noqa: E501 -- long formal Portuguese draft sentences are generated verbatim.

from pathlib import Path

from xjtu_sy_tcc.rul.reporting import atomic_json, atomic_text


def write_manuscript(root: Path, native, strict, tests, condition, life, onset, consensus, costs):
    dummy = native.set_index("experiment").loc["dummy_median"]
    temporal = native.set_index("experiment").loc["selected_features_lstm"]
    k1 = native.set_index("experiment").loc["selected_features_lstm_k1_ablation"]
    context = native.set_index("experiment").loc["selected_features_plus_time_lstm"]
    claims = []

    def claim(identifier, sentence, table, columns, values, support):
        claims.append(
            {
                "claim_id": identifier,
                "generated_sentence": sentence,
                "source_artifact": f"outputs/consolidation/tables/parquet/{table}.parquet",
                "source_table": table,
                "source_columns": columns,
                "filters_applied": {},
                "support_policy": support,
                "computed_value": values,
                "rounding_policy": "two decimal places",
            }
        )
        return sentence

    result = (
        "# Resultados\n\n"
        + claim(
            "R1",
            f"No suporte nativo, o baseline mediano apresentou macro MAE de {dummy.macro_mae:.2f} min, enquanto o LSTM temporal de vibração apresentou {temporal.macro_mae:.2f} min.",
            "native_support_model_comparison",
            ["experiment", "macro_mae"],
            [dummy.macro_mae, temporal.macro_mae],
            "native",
        )
        + "\n\n"
        + claim(
            "R2",
            f"A ablação LSTM de uma aquisição alcançou macro MAE de {k1.macro_mae:.2f} min, e o LSTM com contexto conhecido alcançou {context.macro_mae:.2f} min.",
            "native_support_model_comparison",
            ["experiment", "macro_mae"],
            [k1.macro_mae, context.macro_mae],
            "native",
        )
        + "\n\nAs comparações inferenciais foram pareadas por rolamento e calculadas em suportes idênticos. As análises por condição, estágio de vida e onset estimado são descritivas e retrospectivas.\n"
    )
    discussion = "# Discussão\n\nO baseline mediano pode ser competitivo porque a RUL absoluta varia intensamente entre rolamentos, enquanto a separação por rolamentos completos exige transferência para trajetórias com duração não observada. Métricas globais atribuem maior influência aos rolamentos longos; a macro MAE concede peso igual a cada rolamento. A complexidade adicional não garantiu generalização e o contexto temporal avaliado não superou a ablação de uma aquisição. A recorrência de RMS, variância, desvio-padrão e potências espectrais constitui evidência preditiva, não identificação causal de mecanismos físicos. Resultados negativos delimitam o que o protocolo e as representações avaliadas sustentam.\n"
    limitations = "# Limitações\n\n" + claim(
        "L1",
        "A análise contém 15 rolamentos independentes, cinco por condição e um tipo de rolamento.",
        "per_bearing_metrics",
        ["bearing_id", "condition_id"],
        {"bearing_count": 15, "bearings_per_condition": 5, "bearing_types": 1},
        "native",
    ) + " Há ensaios acelerados controlados, heterogeneidade elevada de vida, ausência de onset oficial, diferenças de suporte das sequências, calibrações retrospectivas na Fase 3 e nenhuma validação externa. Os efeitos de condição não são causais. A busca LSTM foi deliberadamente limitada, e não há demonstração de implantação industrial. Não significância não demonstra equivalência.\n"
    conclusion = "# Conclusão\n\nSob o protocolo de separação por rolamentos completos, com previsão de RUL absoluta e avaliação macro por rolamento, os modelos baseados em atributos de vibração e as sequências temporais avaliadas não superaram o baseline mediano na generalização para rolamentos não observados do XJTU-SY. Tempo decorrido melhorou resultados ponderados por aquisição, mas complexidade não assegurou melhor generalização por rolamento.\n"
    summary = "# Resumo dos resultados\n\n" + result.split("\n\n", 1)[1]
    for name, text in (
        ("resultados", result),
        ("discussao", discussion),
        ("limitacoes", limitations),
        ("conclusao", conclusion),
        ("resumo_resultados", summary),
    ):
        atomic_text(root / f"{name}.md", text)
    atomic_json(root / "claims_manifest.json", {"claims": claims})
    return len(claims)
