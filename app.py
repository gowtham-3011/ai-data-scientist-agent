"""
Streamlit front-end for the End-to-End AI Data Scientist Agent.

Upload a CSV -> click "Run Analysis" -> watch the Supervisor dispatch each
agent in real time -> get a downloadable Markdown (and PDF, if available)
report with embedded charts.
"""
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.build_graph import build_graph
from graph.state import make_initial_state
from utils.llm import llm_available

st.set_page_config(page_title="AI Data Scientist Agent", layout="wide")

AGENT_ICONS = {
    "supervisor": "🧭",
    "inspector": "🔎",
    "cleaner": "🧹",
    "visualizer": "📊",
    "statistician": "📈",
    "explainer": "🗣️",   # <-- NEW ICON
    "reporter": "📝",
}

st.title("🤖 End-to-End AI Data Scientist Agent")
st.caption(
    "A LangGraph multi-agent system: Supervisor → Inspector → Cleaner → "
    "Visualizer → Statistician → Reporter"
)

# ============================================================
# FIXED: Generic LLM check using utils.llm.llm_available()
# ============================================================
if llm_available():
    st.success(
        "✅ LLM configured successfully! (Using Groq/OpenAI/Anthropic) "
        "Full cleaning and narrative enabled.",
        icon="✅"
    )
else:
    st.warning(
        "⚠️ No LLM API key found (or key invalid) — running in **fallback mode**. "
        "The Cleaner will use deterministic rule-based cleaning and the "
        "Statistician will skip the narrative summary, but every chart and "
        "statistic still runs normally. Add `GROQ_API_KEY` (or `OPENAI_API_KEY`) "
        "to `.env` to enable LLM-authored cleaning code and narrative summary.",
        icon="⚠️",
    )

uploaded_file = st.file_uploader("Upload a raw CSV file", type=["csv"])

if uploaded_file is not None:
    df_preview = pd.read_csv(uploaded_file)
    st.subheader("Raw data preview")
    st.dataframe(df_preview.head(10), use_container_width=True)
    st.caption(f"{df_preview.shape[0]} rows × {df_preview.shape[1]} columns")

    run_clicked = st.button("🚀 Run Analysis", type="primary")

    if run_clicked:
        os.makedirs("uploads", exist_ok=True)
        save_path = os.path.join("uploads", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        state = make_initial_state(save_path)
        state["df"] = df_preview.copy()

        graph = build_graph()

        progress_placeholder = st.container()
        log_placeholder = st.empty()
        seen_messages = []
        final_state = None

        with st.status("Running multi-agent pipeline...", expanded=True) as status:
            # graph.stream yields the state after each node completes,
            # so we can render progress in real time.
            for step in graph.stream(state, config={"recursion_limit": 50}):
                for node_name, node_state in step.items():
                    icon = AGENT_ICONS.get(node_name, "⚙️")
                    status.write(f"{icon} **{node_name}** completed")
                    new_msgs = node_state.get("messages", [])[len(seen_messages):]
                    for m in new_msgs:
                        log_placeholder.info(m)
                    seen_messages = node_state.get("messages", [])
                    final_state = node_state
            status.update(label="Pipeline complete ✅", state="complete")

        if final_state is not None:
            st.success("Analysis complete!")

            tab_report, tab_charts, tab_stats, tab_log = st.tabs(
                ["📄 Report", "📊 Charts", "📈 Stats (raw)", "🧾 Full Log"]
            )

            with tab_report:
                st.markdown(final_state.get("report_md", "No report generated."))
                report_path = final_state.get("report_path")
                if report_path and os.path.exists(report_path):
                    with open(report_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Markdown Report", f, file_name="report.md"
                        )
                pdf_path = final_state.get("pdf_path")
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF Report", f, file_name="report.pdf"
                        )

            with tab_charts:
                for p in final_state.get("plots", []):
                    if os.path.exists(p):
                        st.image(p, caption=os.path.basename(p), use_container_width=True)

            with tab_stats:
                st.json(final_state.get("stats", {}))

            with tab_log:
                for m in final_state.get("messages", []):
                    st.write(m)
                if final_state.get("errors"):
                    st.warning("Errors encountered (auto-recovered):")
                    for e in final_state["errors"]:
                        st.code(e)
else:
    st.info("Upload a CSV to get started.")


# ============================================================
# (Optional) Hidden debug section — helps you check .env setup
# ============================================================
with st.expander("🔧 Debug: Environment Check (click to expand)"):
    from dotenv import load_dotenv
    import os
    load_dotenv()
    st.write("**LLM_PROVIDER:**", os.getenv("LLM_PROVIDER", "❌ not set"))
    st.write("**GROQ_API_KEY:**", "✅ present" if os.getenv("GROQ_API_KEY") else "❌ missing")
    st.write("**OPENAI_API_KEY:**", "✅ present" if os.getenv("OPENAI_API_KEY") else "❌ missing")
    st.write("**ANTHROPIC_API_KEY:**", "✅ present" if os.getenv("ANTHROPIC_API_KEY") else "❌ missing")
    st.write("**llm_available():**", llm_available())