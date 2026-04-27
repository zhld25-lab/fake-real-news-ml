from __future__ import annotations

from pathlib import Path

import pandas as pd


LABEL_FAKE = 0
LABEL_REAL = 1
LABEL_NAMES = {
    LABEL_FAKE: "Fake News",
    LABEL_REAL: "Real News",
}


def load_news_data(fake_path: str | Path = "data/Fake.csv", true_path: str | Path = "data/True.csv") -> pd.DataFrame:
    """Load Fake.csv and True.csv, assign labels, and return one combined dataset."""
    fake_path = Path(fake_path)
    true_path = Path(true_path)

    if not fake_path.exists():
        raise FileNotFoundError(f"Fake news file not found: {fake_path}")
    if not true_path.exists():
        raise FileNotFoundError(f"True news file not found: {true_path}")

    fake_df = _read_csv(fake_path)
    true_df = _read_csv(true_path)

    fake_df = _standardize_columns(fake_df)
    true_df = _standardize_columns(true_df)

    fake_df["label"] = LABEL_FAKE
    true_df["label"] = LABEL_REAL

    fake_df["label_name"] = LABEL_NAMES[LABEL_FAKE]
    true_df["label_name"] = LABEL_NAMES[LABEL_REAL]

    fake_df["source_file"] = fake_path.name
    true_df["source_file"] = true_path.name

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.dropna(subset=["title", "text"], how="all").copy()
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["subject"] = df["subject"].fillna("")
    df["date"] = df["date"].fillna("")

    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["title", "text", "subject", "date"]
    for column in required_columns:
        if column not in df.columns:
            df[column] = ""
    return df[required_columns].copy()


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def dataset_summary(df: pd.DataFrame) -> dict[str, object]:
    """Return compact dataset statistics for README, app, or logs."""
    label_counts = df["label_name"].value_counts().to_dict()
    source_counts = df["source_file"].value_counts().to_dict() if "source_file" in df.columns else {}
    duplicate_text_rows = int(df.duplicated(subset=["title", "text"]).sum())

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "label_counts": label_counts,
        "source_counts": source_counts,
        "duplicate_title_text_rows": duplicate_text_rows,
    }
