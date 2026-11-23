"""
Executive Explainer Agent
-------------------------
Takes the technical stats and narrative from the Statistician and translates
them into ultra-simple, jargon-free English for non-technical stakeholders.
No skewness, no kurtosis, no standard deviation — just plain business insights.
"""
from graph.state import AgentState
from utils.llm import get_llm, llm_available

EXPLAINER_SYSTEM_PROMPT = """You are a world-class Business Translator.

Your ONLY job is to take highly technical data science findings and translate them into **ultra-simple, plain English** for a CEO or non-technical business stakeholder.

**CRITICAL RULES:**
1. **NO JARGON**: Absolutely NEVER use these words: skewness, kurtosis, p-value, standard deviation, correlation coefficient, multicollinearity, or heteroscedasticity.
2. **Use Plain Words**: Instead of "correlation", say "relationship" or "link". Instead of "skewed distribution", say "Most users are in one group, but a few extreme outliers exist."
3. **Focus on Impact**: Tell the reader what this MEANS for the business or system. Is it healthy? Is there a problem? Where should they look next?
4. **Keep it Short**: Write exactly 2 short paragraphs (4-5 sentences each).
5. **Reference Exact Numbers**: You can use the specific numbers (like percentages, counts, averages) provided below.
6. **Actionable**: End with 1 clear, actionable recommendation.

You will receive the exact technical numbers and a technical narrative. Rewrite it purely for a business audience.
"""


def _relationship_strength(value: float) -> str:
    """Plain-English label for |correlation| -- derived from the real number, never hardcoded."""
    a = abs(value)
    if a < 0.2:
        return "very weak"
    if a < 0.4:
        return "weak"
    if a < 0.6:
        return "moderate"
    if a < 0.8:
        return "strong"
    return "very strong"


def _strongest_relationship(stats: dict):
    """Returns (pair, value) or (None, None) when there are fewer than two numeric columns."""
    sc = stats.get("strongest_correlation") or {}
    pair, val = sc.get("pair"), sc.get("value")
    if not pair or val is None:
        return None, None
    return pair, val


def _technical_narrative(stats: dict) -> str:
    """The Statistician's narrative lives in stats['narrative']. Placeholder strings it
    emits when it could not call the LLM (they start with '_') carry no information."""
    narrative = (stats.get("narrative") or "").strip()
    return "" if (not narrative or narrative.startswith("_")) else narrative


def explainer_node(state: AgentState) -> AgentState:
    df = state.get("df")
    stats = state.get("stats", {})
    total_rows = len(df) if df is not None else 0
    total_cols = len(df.columns) if df is not None else 0
    pair, val = _strongest_relationship(stats)

    # If LLM is NOT available, generate a generic plain-English fallback
    if not llm_available():
        if pair:
            link_line = (
                f"- The link between '{pair[0]}' and '{pair[1]}' is {_relationship_strength(val)} "
                f"({val:.2f})."
            )
        else:
            link_line = (
                "- There were not enough numeric columns to measure relationships "
                "between different data points."
            )

        summary = f"""
**Executive Summary (Fallback Mode - No LLM)**

We analyzed {total_rows} records across {total_cols} data points.
{link_line}
- Actionable Advice: Review the charts and the data-quality notes in this report, and look at factors this dataset does not capture before drawing conclusions.
"""
        state["executive_summary"] = summary
        state["explained"] = True
        state.setdefault("messages", []).append(
            "🗣️ Explainer: Generated fallback executive summary (LLM unavailable)."
        )
        state["current_node"] = "explainer"
        return state

    # --- LLM is available: Generate a beautiful business summary ---
    try:
        llm = get_llm()

        relationship = (
            f"between '{pair[0]}' and '{pair[1]}' (value: {val:.3f}, "
            f"{_relationship_strength(val)})"
            if pair
            else "none measurable (fewer than two numeric columns)"
        )

        prompt = f"""
{EXPLAINER_SYSTEM_PROMPT}

**Technical Data Provided (Do NOT mention these technical names, just interpret them):**
- Total records analyzed: {total_rows}
- Number of columns: {total_cols}
- Numeric measurements available: {stats.get('numeric_columns', [])}
- Strongest statistical link: {relationship}
- Raw technical skewness numbers (to be interpreted as "most users are average, a few are extremes"): {stats.get('skew', {})}
- Technical narrative from the analyst: {_technical_narrative(stats) or '(none available)'}

Now, translate this into pure business English for a CEO. Remember: NO JARGON.
"""
        response = llm.invoke(prompt)
        state["executive_summary"] = response.content.strip()
        state["explained"] = True
        state.setdefault("messages", []).append(
            "🗣️ Explainer: Translated technical findings into executive-friendly summary."
        )
    except Exception as e:
        state["executive_summary"] = (
            f"⚠️ Could not generate executive summary due to a technical glitch ({type(e).__name__}). "
            f"However, the raw stats show we analyzed {total_rows} records successfully."
        )
        state["explained"] = True
        state.setdefault("messages", []).append(
            f"🗣️ Explainer: Failed to generate summary ({type(e).__name__}), but marked as explained."
        )

    state["current_node"] = "explainer"
    return state
