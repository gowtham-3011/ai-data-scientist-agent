"""
Visualizer Agent
----------------
Generates 3-4 relevant charts from the CLEANED dataframe using
Matplotlib/Seaborn and saves them as PNGs:
  1. Correlation heatmap (numeric columns)
  2. Distribution plots for the top numeric columns
  3. Boxplots for the top numeric columns (outlier detection)
  4. A bar chart of the top categorical column's value counts (if present)

Deterministic and programmatic on purpose: chart generation is exactly the
kind of task you want reproducible, not re-hallucinated by an LLM each run.
"""
import os
import matplotlib
matplotlib.use("Agg")  # headless-safe for Streamlit / servers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from graph.state import AgentState
from utils.data_helpers import split_numeric_categorical

PLOTS_DIR = "plots"
MAX_NUMERIC_COLS_FOR_DIST = 4


def _ensure_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def visualizer_node(state: AgentState) -> AgentState:
    df: pd.DataFrame = state["df"]
    _ensure_dir()
    plots = []
    sns.set_theme(style="whitegrid")

    numeric_df, categorical_df = split_numeric_categorical(df)

    # 1. Correlation heatmap
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = numeric_df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)

    # 2. Distribution plots (histograms + KDE) for top-variance numeric cols
    if numeric_df.shape[1] >= 1:
        top_cols = numeric_df.var().sort_values(ascending=False).head(
            MAX_NUMERIC_COLS_FOR_DIST
        ).index.tolist()
        n = len(top_cols)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, top_cols):
            sns.histplot(numeric_df[col].dropna(), kde=True, ax=ax, color="steelblue")
            ax.set_title(f"Distribution: {col}")
        fig.tight_layout()
        path = os.path.join(PLOTS_DIR, "distributions.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)

    # 3. Boxplots for the same top numeric columns (outlier detection)
    if numeric_df.shape[1] >= 1:
        fig, ax = plt.subplots(figsize=(max(6, 2 * len(top_cols)), 5))
        sns.boxplot(data=numeric_df[top_cols], ax=ax, palette="Set2")
        ax.set_title("Boxplots (Outlier Detection)")
        fig.tight_layout()
        path = os.path.join(PLOTS_DIR, "boxplots.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)

    # 4. Bar chart of the most informative categorical column
    #    (prefer columns with 2-20 unique values -- skips constant columns
    #    and skips high-cardinality free-text/ID columns)
    if categorical_df.shape[1] >= 1:
        nunique = categorical_df.nunique()
        candidates = nunique[(nunique >= 2) & (nunique <= 20)]
        col = (candidates.sort_values().index[0] if not candidates.empty
               else nunique.sort_values().index[0])
        counts = df[col].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, color="darkorange")
        ax.set_title(f"Top Categories: {col}")
        ax.set_xlabel("Count")
        fig.tight_layout()
        path = os.path.join(PLOTS_DIR, "categorical_counts.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)

    state["plots"] = plots
    state["visualized"] = True
    state.setdefault("messages", []).append(
        f"📊 Visualizer: generated {len(plots)} charts -> {plots}"
    )
    state["current_node"] = "visualizer"
    return state
