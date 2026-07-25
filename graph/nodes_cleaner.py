"""
Cleaner Agent
-------------
Asks the LLM to WRITE pandas code that fixes dates, fills NaNs, and drops
duplicates -- then actually EXECUTES that code against a copy of the
dataframe. If it throws, the error is fed back to the LLM so it can correct
itself (bounded retries). If it still fails after `max_clean_attempts`, we
fall back to a deterministic rule-based cleaning pass so the pipeline never
gets stuck.
"""
import pandas as pd
import numpy as np
from graph.state import AgentState
from utils.llm import get_llm, llm_available
from utils.code_executor import safe_exec, extract_code

CLEAN_SYSTEM_PROMPT = """You are a senior data cleaning engineer.

You will be given a pandas DataFrame available as the variable `df`, and a
data health report describing its issues.

Write ONLY executable Python code (no explanations, no markdown fences, no
comments needed) that cleans `df` according to these rules:
- Convert columns that look like dates to proper datetime using
  pd.to_datetime(df[col], errors='coerce')
- Fill missing numeric values with the column median
- Fill missing categorical/object values with the column mode, or 'Unknown'
  if no mode exists
- Drop exact duplicate rows with df.drop_duplicates()
- Never drop columns
- Only use `pd`, `np`, and `df` -- they are already defined, do not import
  anything
- The final cleaned dataframe MUST be assigned back to the variable `df`

Return nothing but the raw Python code.
"""


def _fallback_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic rule-based cleaning used if the LLM-authored code keeps failing."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            continue  # already a proper datetime, nothing to fill/convert
        else:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[col] = parsed
            else:
                mode = df[col].mode()
                fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
    df = df.drop_duplicates()
    return df


def cleaner_node(state: AgentState) -> AgentState:
    df = state["df"]
    attempt = state.get("clean_attempts", 0) + 1
    state["clean_attempts"] = attempt
    max_attempts = state.get("max_clean_attempts", 3)

    # No API key configured -> skip straight to the deterministic fallback
    # instead of burning retries on calls that will never succeed.
    if not llm_available():
        state["df"] = _fallback_clean(df)
        state["cleaned"] = True
        state["cleaning_error"] = None
        state.setdefault("messages", []).append(
            "🧹 Cleaner: no ANTHROPIC_API_KEY configured -> used deterministic "
            "rule-based cleaning (median/mode fill + date parse + dedupe) instead "
            "of LLM-generated code."
        )
        state["current_node"] = "cleaner"
        return state

    error_context = ""
    if state.get("cleaning_error"):
        error_context = (
            f"\n\nYour previous attempt raised this error:\n{state['cleaning_error']}\n"
            f"Previous code was:\n{state.get('cleaning_code', '')}\n"
            f"Fix the bug and return corrected code."
        )

    prompt = (
        f"{CLEAN_SYSTEM_PROMPT}\n\n"
        f"Health report:\n{state['health_report']}\n\n"
        f"df.dtypes:\n{df.dtypes.to_string()}"
        f"{error_context}"
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        code = extract_code(response.content)
    except Exception as e:
        # API-level failure (bad/expired key, rate limit, network, etc.)
        # -- treat exactly like a failed cleaning attempt, don't crash the app.
        code = ""
        state["cleaning_error"] = f"LLM call failed: {type(e).__name__}: {e}"
        state.setdefault("errors", []).append(
            f"Cleaning attempt {attempt} LLM call failed: {state['cleaning_error']}"
        )
        if attempt >= max_attempts:
            state["df"] = _fallback_clean(df)
            state["cleaned"] = True
            state["cleaning_error"] = None
            state.setdefault("messages", []).append(
                f"🧹 Cleaner: LLM call failed {attempt} times -> applied deterministic "
                f"fallback cleaning."
            )
        else:
            state.setdefault("messages", []).append(
                f"⚠️ Cleaner: attempt {attempt} LLM call failed, retrying..."
            )
        state["current_node"] = "cleaner"
        return state

    state["cleaning_code"] = code

    local_env = {"df": df.copy(), "pd": pd, "np": np}
    success, _stdout, error = safe_exec(code, local_env)

    if success and isinstance(local_env.get("df"), pd.DataFrame):
        state["df"] = local_env["df"]
        state["cleaning_error"] = None
        state["cleaned"] = True
        state.setdefault("messages", []).append(
            f"🧹 Cleaner: LLM-generated cleaning code succeeded on attempt {attempt}."
        )
    else:
        state["cleaning_error"] = error or "Unknown execution error"
        state.setdefault("errors", []).append(
            f"Cleaning attempt {attempt} failed: {state['cleaning_error']}"
        )
        if attempt >= max_attempts:
            state["df"] = _fallback_clean(df)
            state["cleaned"] = True
            state["cleaning_error"] = None
            state.setdefault("messages", []).append(
                f"🧹 Cleaner: LLM code failed {attempt} times -> applied deterministic "
                f"fallback cleaning (median/mode fill + date parse + dedupe)."
            )
        else:
            state.setdefault("messages", []).append(
                f"⚠️ Cleaner: attempt {attempt} failed, retrying with error feedback..."
            )

    state["current_node"] = "cleaner"
    return state
