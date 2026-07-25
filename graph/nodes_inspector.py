"""
Inspector Agent
---------------
Runs df.info(), df.describe(), and missing-value / duplicate counts, and
turns them into a human-readable "Data Health Report" that downstream
agents (Cleaner, Reporter) consume as context.

No LLM call here -- this is pure, deterministic pandas profiling, which is
exactly what you want for numbers you plan to trust.
"""
import io
import pandas as pd
from graph.state import AgentState


def inspector_node(state: AgentState) -> AgentState:
    df: pd.DataFrame = state["df"]
    state["original_shape"] = df.shape

    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()

    with pd.option_context("display.max_columns", None, "display.width", 120):
        describe_str = df.describe(include="all").fillna("").to_string()

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_str = (
        missing.to_string() if not missing.empty else "No missing values detected."
    )

    dup_count = int(df.duplicated().sum())

    dtypes_str = df.dtypes.to_string()

    # Heuristic: flag non-numeric columns that look like dates but aren't parsed yet
    non_numeric_cols = df.select_dtypes(
        exclude=["number", "bool", "datetime", "datetimetz"]
    ).columns
    suspected_date_cols = []
    for col in non_numeric_cols:
        sample = df[col].dropna().astype(str).head(20)
        if len(sample) == 0:
            continue
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() > 0.8:
            suspected_date_cols.append(col)

    report_lines = [
        "## Data Health Report",
        "",
        f"**Shape:** {df.shape[0]} rows x {df.shape[1]} columns",
        f"**Duplicate rows:** {dup_count}",
        f"**Suspected unparsed date columns:** {suspected_date_cols or 'None'}",
        "",
        "### dtypes",
        "```",
        dtypes_str,
        "```",
        "",
        "### Missing values (columns with at least 1 NaN)",
        "```",
        missing_str,
        "```",
        "",
        "### df.info()",
        "```",
        info_str,
        "```",
        "",
        "### df.describe(include='all')",
        "```",
        describe_str,
        "```",
    ]

    state["health_report"] = "\n".join(report_lines)
    state["inspected"] = True
    state.setdefault("messages", []).append(
        f"🔎 Inspector: profiled {df.shape[0]} rows / {df.shape[1]} cols, "
        f"{dup_count} duplicate rows, {int(missing.sum())} total missing cells."
    )
    state["current_node"] = "inspector"
    return state
