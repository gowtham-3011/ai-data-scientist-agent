# 🤖 End-to-End AI Data Scientist Agent

A multi-agent, autonomous data science pipeline built with **LangGraph**, **Pandas**,
**Streamlit**, and **Matplotlib/Seaborn**. Upload a raw CSV and a team of six
cooperating agents will profile it, clean it, chart it, analyze it, and hand
you back a polished Markdown (and optional PDF) report — with live progress
in the UI.

## Architecture

```
                     ┌─────────────┐
        ┌───────────▶│ SUPERVISOR  │◀───────────────────────────┐
        │            │  (router)   │                            │
        │            └──────┬──────┘                            │
        │      picks next agent based on shared State            │
        │                   │                                    │
        │   ┌───────┬───────┼────────┬──────────────┬──────────┐ │
        │   ▼        ▼      │        ▼              ▼          ▼ │
        │ Inspector Cleaner  │   Visualizer   Statistician   Reporter
        │  (profile) (fix,   │    (4 PNG        (exact         (final
        │            retry   │     charts)      stats via      .md/.pdf)
        │            on err) │                  code, not
        │                    │                  guesses)
        └────────────────────┴──────────────────────────────────┘
              every worker hands control back to the Supervisor
```

**Shared `State` dict** (`graph/state.py`) is the single source of truth
passed node-to-node: `df`, `plots`, `stats`, `errors`, `messages`, plus
per-stage completion flags (`inspected`, `cleaned`, `visualized`, `analyzed`,
`reported`) that the Supervisor uses to decide the next step.

### The agents

| Agent | File | What it does |
|---|---|---|
| **Supervisor** | `graph/supervisor.py` | Pure routing logic — no data mutation. Decides which worker runs next, and loops the Cleaner back on itself on failure. |
| **Inspector** | `graph/nodes_inspector.py` | Deterministic `df.info()`, `describe()`, missing-value & duplicate counts → a "Data Health Report". |
| **Cleaner** | `graph/nodes_cleaner.py` | Asks an LLM to **write** pandas cleaning code, then actually **executes** it in a sandboxed `exec()`. On exception, feeds the error back to the LLM and retries (`max_clean_attempts`, default 3). If it still fails, falls back to deterministic median/mode-fill + de-dup so the pipeline never gets stuck. |
| **Visualizer** | `graph/nodes_visualizer.py` | Generates a correlation heatmap, distribution plots, boxplots, and a categorical bar chart as PNGs — purely programmatic (reproducible, no LLM). |
| **Statistician** | `graph/nodes_statistician.py` | Computes exact correlation coefficients, skew/kurtosis, and descriptive stats with pandas/numpy. The LLM is only used *after* the numbers exist, to narrate them — it cannot invent figures. |
| **Reporter** | `graph/nodes_reporter.py` | Compiles the health report, cleaned data preview, charts, and stats into one Markdown report (+ best-effort PDF export). |

## Folder structure

```
ai_data_scientist_agent/
├── app.py                      # Streamlit UI
├── graph/
│   ├── state.py                # Shared AgentState TypedDict
│   ├── supervisor.py            # Router logic
│   ├── build_graph.py           # Wires nodes into a LangGraph StateGraph
│   ├── nodes_inspector.py
│   ├── nodes_cleaner.py
│   ├── nodes_visualizer.py
│   ├── nodes_statistician.py
│   └── nodes_reporter.py
├── utils/
│   ├── llm.py                  # Claude client wrapper
│   └── code_executor.py        # Sandboxed exec() + code-fence stripping
├── sample_data/
│   └── sample_sales.csv        # Messy demo dataset (dupes, NaNs, unparsed dates)
├── plots/                       # PNGs written here at runtime
├── reports/                     # report.md / report.pdf written here at runtime
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Clone / unzip the project, then create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   PDF export is optional — if you want it, also run:
   ```bash
   pip install markdown2 weasyprint
   ```
   (weasyprint has some OS-level dependencies; the app works fine without it,
   it just skips PDF export and still writes the Markdown report.)

3. **Add your API key (optional):**
   ```bash
   cp .env.example .env
   # then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
   ```
   **Don't have a key?** The app runs fully without one. The Cleaner falls
   back to deterministic rule-based cleaning (median/mode fill + date
   parsing + de-dup) and the Statistician skips its narrative paragraph —
   every chart and every statistic still runs exactly the same, since those
   never depended on the LLM in the first place. The Streamlit UI shows a
   banner telling you which mode you're in.

## Run

```bash
streamlit run app.py
```

Then in the browser tab that opens:
1. Upload a CSV (or use `sample_data/sample_sales.csv` to try it immediately).
2. Review the raw preview.
3. Click **🚀 Run Analysis**.
4. Watch each agent report in as it finishes (Supervisor → Inspector → Cleaner
   → Visualizer → Statistician → Reporter), then browse the **Report**,
   **Charts**, **Stats**, and **Full Log** tabs, and download the report.

## Design notes / talking points for interviews

- **Supervisor pattern**: routing is a pure function of `State`
  (`graph/supervisor.py::_decide_next`), decoupled from the LLM calls
  themselves — makes the control flow unit-testable without hitting the API.
- **Self-healing Cleaner**: the LLM's code is *executed*, not trusted;
  exceptions are captured and fed back as context for a corrected retry, with
  a deterministic fallback so a bad LLM response can never hang the pipeline.
- **Numbers come from code, not the model**: the Statistician computes every
  statistic with pandas/numpy first; the LLM only narrates already-computed
  values, which are injected into its prompt — it has nothing to hallucinate.
- **Observability**: every node appends to `state["messages"]`, which the
  Streamlit UI streams live via `graph.stream(...)`, so you can *watch* the
  multi-agent system think.
