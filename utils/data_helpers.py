"""
Shared helper for splitting a dataframe into "numeric" vs "categorical"
columns in a way that's actually useful for analysis.

Why this exists: pandas' `select_dtypes(include="number")` excludes `bool`
columns. But boolean flags (e.g. `login_success`, `privileged_action`) are
usually the MOST analytically interesting columns in a dataset (they're
binary 0/1 signals) -- excluding them meant correlation heatmaps and stats
silently dropped the most useful features whenever a CSV was mostly
categorical/boolean with only one "true" numeric column.
"""
import pandas as pd


def split_numeric_categorical(df: pd.DataFrame):
    """
    Returns (numeric_df, categorical_df):
      - numeric_df: all number columns PLUS bool columns cast to int (0/1)
      - categorical_df: everything else (object/string/category), excluding
        datetimes and the bool columns already promoted to numeric_df
    """
    bool_cols = df.select_dtypes(include="bool").columns
    number_cols = df.select_dtypes(include="number").columns

    numeric_df = df[list(number_cols)].copy()
    for col in bool_cols:
        numeric_df[col] = df[col].astype(int)

    categorical_df = df.select_dtypes(
        exclude=["number", "bool", "datetime", "datetimetz"]
    )
    return numeric_df, categorical_df
