from __future__ import annotations

import re

import pandas as pd


def clean_text(value: object) -> str:
    """Clean news text for classical machine learning."""
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_title_and_text(df: pd.DataFrame) -> pd.Series:
    """Combine title and article body into one model input."""
    title = df["title"].fillna("").astype(str)
    text = df["text"].fillna("").astype(str)
    return (title + " " + text).str.strip()


def add_clean_text_column(df: pd.DataFrame, output_column: str = "clean_text") -> pd.DataFrame:
    """Add a cleaned text column and remove rows that become empty."""
    result = df.copy()
    result["combined_text"] = combine_title_and_text(result)
    result[output_column] = result["combined_text"].map(clean_text)
    result = result[result[output_column].str.len() > 0].copy()
    return result
