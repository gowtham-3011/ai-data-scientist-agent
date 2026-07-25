"""
Wires every node into a LangGraph StateGraph using the classic
Supervisor <-> Worker loop:

    START -> supervisor -> (conditional) -> {inspector|cleaner|visualizer|
                                              statistician|explainer|reporter}
                 ^                                    |
                 |____________________________________|
                                 (loops back to supervisor after every worker)
    supervisor -> END   (once state['reported'] is True)
"""
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from graph.supervisor import (
    supervisor_node,
    route_after_supervisor,
    NODE_INSPECTOR,
    NODE_CLEANER,
    NODE_VISUALIZER,
    NODE_STATISTICIAN,
    NODE_EXPLAINER,          # <-- NEW: Import the Explainer constant
    NODE_REPORTER,
    NODE_END,
)
from graph.nodes_inspector import inspector_node
from graph.nodes_cleaner import cleaner_node
from graph.nodes_visualizer import visualizer_node
from graph.nodes_statistician import statistician_node
from graph.nodes_explainer import explainer_node   # <-- NEW: Import the Explainer function
from graph.nodes_reporter import reporter_node


def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes to the graph
    graph.add_node("supervisor", supervisor_node)
    graph.add_node(NODE_INSPECTOR, inspector_node)
    graph.add_node(NODE_CLEANER, cleaner_node)
    graph.add_node(NODE_VISUALIZER, visualizer_node)
    graph.add_node(NODE_STATISTICIAN, statistician_node)
    graph.add_node(NODE_EXPLAINER, explainer_node)   # <-- NEW: Add the Explainer node
    graph.add_node(NODE_REPORTER, reporter_node)

    # Entry point
    graph.add_edge(START, "supervisor")

    # Conditional routing from the supervisor
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            NODE_INSPECTOR: NODE_INSPECTOR,
            NODE_CLEANER: NODE_CLEANER,
            NODE_VISUALIZER: NODE_VISUALIZER,
            NODE_STATISTICIAN: NODE_STATISTICIAN,
            NODE_EXPLAINER: NODE_EXPLAINER,   # <-- NEW: Route to Explainer
            NODE_REPORTER: NODE_REPORTER,
            NODE_END: END,
        },
    )

    # Every worker hands control back to the supervisor (the "Supervisor Loop")
    # Make sure to include NODE_EXPLAINER in this list!
    for node_name in (
        NODE_INSPECTOR,
        NODE_CLEANER,
        NODE_VISUALIZER,
        NODE_STATISTICIAN,
        NODE_EXPLAINER,   # <-- NEW: Add Explainer to the loop
        NODE_REPORTER,
    ):
        graph.add_edge(node_name, "supervisor")

    return graph.compile()