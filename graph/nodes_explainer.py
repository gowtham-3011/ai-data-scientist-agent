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


def explainer_node(state: AgentState) -> AgentState:
    df = state.get("df")
    stats = state.get("stats", {})
    technical_narrative = state.get("analysis_narrative", "")

    # If LLM is NOT available, generate a generic plain-English fallback
    if not llm_available():
        total_rows = len(df) if df is not None else 0
        num_cols = len(df.columns) if df is not None else 0
        
        # Extract key business numbers safely
        strongest_corr = stats.get("strongest_correlation", {})
        pair = strongest_corr.get("pair", ("N/A", "N/A"))
        val = strongest_corr.get("value", 0)
        
        summary = f"""
**Executive Summary (Fallback Mode - No LLM)**

We analyzed {total_rows} records. 
- There were {stats.get('total_rows', total_rows)} total records and {stats.get('total_columns', num_cols)} data points.
- The link between '{pair[0]}' and '{pair[1]}' is very weak ({val:.2f}), meaning they likely happen independently of each other.
- Actionable Advice: Look beyond these metrics. The data suggests we need to examine other factors (like user device or region) to find real issues.
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
        
        # Build a data payload for the LLM with only what's needed
        total_rows = len(df) if df is not None else 0
        strongest_corr = stats.get("strongest_correlation", {})
        pair = strongest_corr.get("pair", ("N/A", "N/A"))
        val = strongest_corr.get("value", 0)
        
        # Pull out simple stats for the prompt
        simple_stats = {
            "total_records": total_rows,
            "numeric_features": stats.get("numeric_columns", []),
            "strongest_relationship": f"between '{pair[0]}' and '{pair[1]}' (value: {val:.3f})",
            "skewness": stats.get("skewness", {}),  # We'll tell the LLM to avoid this word
        }

        prompt = f"""
{EXPLAINER_SYSTEM_PROMPT}

**Technical Data Provided (Do NOT mention these technical names, just interpret them):**
- Total records analyzed: {simple_stats['total_records']}
- Strongest statistical link: {simple_stats['strongest_relationship']}
- Raw technical skewness numbers (to be interpreted as "most users are average, a few are extremes"): {simple_stats['skewness']}
- Technical narrative from the analyst: {technical_narrative}

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
            f"However, the raw stats show we analyzed {len(df) if df is not None else 0} records successfully."
        )
        state["explained"] = True
        state.setdefault("messages", []).append(
            f"🗣️ Explainer: Failed to generate summary ({type(e).__name__}), but marked as explained."
        )

    state["current_node"] = "explainer"
    return state