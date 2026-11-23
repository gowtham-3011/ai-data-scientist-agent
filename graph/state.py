"""
Shared state object passed between every node in the LangGraph.

This is the single source of truth the Supervisor, Inspector, Cleaner,
Visualizer, Statistician and Reporter all read from / write to.
"""
from typing import TypedDict, List, Dict, Any, Optional
import pandas as pd


class AgentState(TypedDict, total=False):
    # --- input ---
    file_path: str

    # --- data ---
    df: Optional[pd.DataFrame]          # current (possibly cleaned) dataframe
    original_shape: Optional[tuple]     # shape before cleaning, for the report

    # --- Inspector output ---
    health_report: str                  # textual data health report
    inspected: bool

    # --- Cleaner output ---
    cleaning_code: str                  # last generated pandas code
    cleaning_error: Optional[str]       # last exception (None if success)
    cleaning_method: str                # "llm" | "fallback" | "fallback_no_key" -- what actually cleaned df
    clean_attempts: int
    max_clean_attempts: int
    cleaned: bool

    # --- Visualizer output ---
    plots: List[str]                    # file paths to saved PNGs
    visualized: bool

    # --- Statistician output ---
    stats: Dict[str, Any]                # exact numbers (never LLM-guessed)
    analyzed: bool

    # --- Reporter output ---
    report_md: str
    report_path: str
    pdf_path: Optional[str]
    reported: bool

    # --- bookkeeping / observability ---
    errors: List[str]                    # running list of all errors seen
    messages: List[str]                  # human-readable log, shown in Streamlit
    current_node: str                    # last node the Supervisor dispatched to

    # --- Explainer output ---
    executive_summary: str
    explained: bool


def make_initial_state(file_path: str, max_clean_attempts: int = 3) -> AgentState:
    return AgentState(
        file_path=file_path,
        df=None,
        original_shape=None,
        health_report="",
        inspected=False,
        cleaning_code="",
        cleaning_error=None,
        cleaning_method="",
        clean_attempts=0,
        max_clean_attempts=max_clean_attempts,
        cleaned=False,
        plots=[],
        visualized=False,
        stats={},
        analyzed=False,
        executive_summary="",
        explained=False,
        report_md="",
        report_path="",
        pdf_path=None,
        reported=False,
        errors=[],
        messages=[],
        current_node="start",
    )
