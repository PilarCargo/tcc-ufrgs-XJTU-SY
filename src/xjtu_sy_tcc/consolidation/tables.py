"""Atomic CSV, Parquet, Markdown, and LaTeX scientific tables."""

from pathlib import Path

import pandas as pd

from xjtu_sy_tcc.rul.reporting import atomic_table, atomic_text


def write_table_set(root: Path, name: str, table: pd.DataFrame, precision: int = 2):
    for folder, suffix in (("csv", ".csv"), ("parquet", ".parquet")):
        atomic_table(root / folder / f"{name}{suffix}", table)
    display = table.copy()
    for column in display.select_dtypes("number"):
        display[column] = display[column].map(
            lambda x: f"{x:.{precision}f}" if pd.notna(x) else "--"
        )
    header = "| " + " | ".join(map(str, display.columns)) + " |"
    separator = "|" + "|".join("---" for _ in display.columns) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False)]
    atomic_text(root / "markdown" / f"{name}.md", "\n".join([header, separator, *rows]) + "\n")
    latex = display.to_latex(index=False, na_rep="--", escape=True)
    atomic_text(root / "latex" / f"{name}.tex", f"% Phase 6 table: {name}\n" + latex)
