"""
Reporter Agent
--------------
Compiles the cleaned data preview, chart images, and exact statistics into a
single, professional Markdown report. Optionally renders a PDF too (if
`markdown2` + `weasyprint` are installed -- this is treated as best-effort
since those pull in system-level deps).
"""
import os
import datetime
import pandas as pd
from graph.state import AgentState

REPORTS_DIR = "reports"


def _df_head_markdown(df: pd.DataFrame, n: int = 5) -> str:
    try:
        return df.head(n).to_markdown(index=False)
    except ImportError:
        # tabulate not installed -- fall back to a plain string table
        return "```\n" + df.head(n).to_string(index=False) + "\n```"


def _describe_markdown(stats: dict) -> str:
    describe = stats.get("describe", {})
    if not describe:
        return "_No numeric columns to describe._"
    df = pd.DataFrame(describe).round(3)
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def reporter_node(state: AgentState) -> AgentState:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = state["df"]
    stats = state.get("stats", {})
    plots = state.get("plots", [])
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get the executive summary from the Explainer agent
    executive_summary = state.get("executive_summary", "_No executive summary generated._")

    sections = [
        "# 🤖 Automated Data Science Report",
        f"*Generated {generated_at}*",
        "",
        "## 📌 Executive Summary (For Business Stakeholders)",
        executive_summary,   # <-- THIS IS THE NEW ADDITION
        "",
        "## 1. Data Health Report (pre-cleaning)",
        state.get("health_report", "_not available_"),
        "",
        "## 2. Cleaning Summary",
        f"- Original shape: {state.get('original_shape')}",
        f"- Final shape after cleaning: {df.shape}",
        f"- Cleaning attempts: {state.get('clean_attempts', 0)}",
        (
            "- Cleaning method: LLM-generated pandas code"
            if not state.get("cleaning_error")
            else "- Cleaning method: deterministic fallback (LLM code failed repeatedly)"
        ),
        "",
        "### Cleaned data preview",
        _df_head_markdown(df),
        "",
        "## 3. Visualizations",
    ]

    for p in plots:
        title = os.path.splitext(os.path.basename(p))[0].replace("_", " ").title()
        sections.append(f"### {title}")
        sections.append(f"![{title}]({p})")
        sections.append("")

    sections += [
        "## 4. Statistical Analysis",
        f"**Strongest correlation:** {stats.get('strongest_correlation')}",
        "",
        "### Descriptive statistics",
        _describe_markdown(stats),
        "",
        "### Narrative Summary",
        stats.get("narrative", "_not available_"),
        "",
        "## 5. Pipeline Log",
        "```",
        *state.get("messages", []),
        "```",
    ]

    if state.get("errors"):
        sections += ["", "## Errors Encountered (auto-recovered)", "```", *state["errors"], "```"]

    report_md = "\n".join(sections)
    report_path = os.path.join(REPORTS_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    state["report_md"] = report_md
    state["report_path"] = report_path

    # Best-effort PDF export
    pdf_path = None
    try:
        import markdown2
        from weasyprint import HTML

        html_body = markdown2.markdown(report_md, extras=["tables", "fenced-code-blocks"])
        pdf_path = os.path.join(REPORTS_DIR, "report.pdf")
        HTML(string=html_body, base_url=".").write_pdf(pdf_path)
    except Exception as e:
        state.setdefault("messages", []).append(
            f"ℹ️ Reporter: PDF export skipped ({type(e).__name__}: {e}). "
            f"Markdown report still generated."
        )

    state["pdf_path"] = pdf_path
    state["reported"] = True
    state.setdefault("messages", []).append(
        f"📝 Reporter: report written to {report_path}"
        + (f" and {pdf_path}" if pdf_path else "")
    )
    state["current_node"] = "reporter"
    return state