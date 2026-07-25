"""
Supervisor (Router) Agent
-------------------------
Classic LangGraph "supervisor loop": every worker agent hands control back
to the Supervisor, which inspects the shared State and decides which agent
runs next -- including looping the Cleaner back on itself when its
generated code throws, up to `max_clean_attempts` times.

This node does not touch the data; it only reasons about *control flow*,
which is exactly the Supervisor's job in a multi-agent system.
"""
from graph.state import AgentState

NODE_INSPECTOR = "inspector"
NODE_CLEANER = "cleaner"
NODE_VISUALIZER = "visualizer"
NODE_STATISTICIAN = "statistician"
NODE_EXPLAINER = "explainer"     # <-- NEW: Executive Summary Generator
NODE_REPORTER = "reporter"
NODE_END = "__end__"


def supervisor_node(state: AgentState) -> AgentState:
    """No-op pass-through node; conditional edge does the real routing.
    Kept as an explicit node so it shows up in the graph / Streamlit log,
    making the 'Supervisor plans, workers execute' pattern visible.
    """
    plan = _decide_next(state)
    state.setdefault("messages", []).append(f"🧭 Supervisor: routing to '{plan}'")
    state["current_node"] = "supervisor"
    return state


def _decide_next(state: AgentState) -> str:
    """Pure routing logic -- the single place that encodes the workflow plan."""
    if not state.get("inspected"):
        return NODE_INSPECTOR

    if not state.get("cleaned"):
        # Cleaner failed and we haven't hit the retry ceiling yet -> loop back to it
        return NODE_CLEANER

    if not state.get("visualized"):
        return NODE_VISUALIZER

    if not state.get("analyzed"):
        return NODE_STATISTICIAN

    # NEW: After stats are computed, translate them for business users
    if not state.get("explained"):
        return NODE_EXPLAINER

    if not state.get("reported"):
        return NODE_REPORTER

    return NODE_END


def route_after_supervisor(state: AgentState) -> str:
    """Conditional-edge function LangGraph calls after the supervisor node."""
    return _decide_next(state)