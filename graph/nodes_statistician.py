"""
Statistician Agent
------------------
Computes EXACT numbers with pandas/numpy (never guessed by an LLM):
  - descriptive stats (mean/median/std/min/max/quartiles)
  - Pearson correlation matrix
  - skewness / kurtosis per numeric column
  - top correlated pair

The LLM is only used AFTER the numbers exist, to write a short narrative
that references the already-computed figures -- it cannot invent numbers
because the real ones are injected directly into its prompt.
"""
import pandas as pd
from graph.state import AgentState
from utils.llm import get_llm, llm_available
from utils.data_helpers import split_numeric_categorical


def statistician_node(state: AgentState) -> AgentState:
    df: pd.DataFrame = state["df"]
    numeric_df, _categorical_df = split_numeric_categorical(df)

    stats = {}
    stats["describe"] = numeric_df.describe().to_dict()
    stats["shape"] = df.shape

    corr_matrix = numeric_df.corr(numeric_only=True)
    stats["correlation_matrix"] = corr_matrix.round(4).to_dict()

    # Find the single strongest (non-self) correlation pair
    top_pair = None
    top_val = 0.0
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr_matrix.iloc[i, j]
            if pd.notna(v) and abs(v) > abs(top_val):
                top_val = v
                top_pair = (cols[i], cols[j])
    stats["strongest_correlation"] = {
        "pair": top_pair,
        "value": round(float(top_val), 4) if top_pair else None,
    }

    stats["skew"] = numeric_df.skew(numeric_only=True).round(4).to_dict()
    stats["kurtosis"] = numeric_df.kurtosis(numeric_only=True).round(4).to_dict()

    state["stats"] = stats

    # Narrative summary -- LLM narrates, but every number came from `stats` above.
    if not llm_available():
        stats["narrative"] = (
            "_No LLM API key configured (or key invalid) -- narrative generation skipped. "
            "All statistics above were computed directly with pandas/numpy._"
        )
    else:
        try:
            llm = get_llm()
            prompt = (
                "You are a data analyst. Using ONLY the numbers provided below "
                "(do not invent any new numbers), write a concise 4-6 sentence "
                "narrative summary of the key findings for a business audience.\n\n"
                f"Shape: {stats['shape']}\n"
                f"Strongest correlation: {stats['strongest_correlation']}\n"
                f"Skewness per column: {stats['skew']}\n"
                f"Kurtosis per column: {stats['kurtosis']}\n"
            )
            response = llm.invoke(prompt)
            stats["narrative"] = response.content.strip()
        except Exception as e:
            stats["narrative"] = (
                "_(LLM narrative unavailable -- showing raw statistics only. "
                f"Error: {e})_"
            )

    state["analyzed"] = True
    state.setdefault("messages", []).append(
        "📈 Statistician: computed describe(), correlation matrix, skew/kurtosis "
        f"(strongest pair: {stats['strongest_correlation']})."
    )
    state["current_node"] = "statistician"
    return state
