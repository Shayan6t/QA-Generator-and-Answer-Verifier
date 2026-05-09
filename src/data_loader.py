"""
data_loader.py — RACE dataset loading, cleaning, and train/val/test splitting.

The Kaggle upload (ankitdhiman7/race-dataset) ships identical CSVs for
train, test, and val.  This module loads the data once, cleans it, and
creates a proper 70/15/15 stratified split on the 'answer' column.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_raw(filename: str = "train.csv") -> pd.DataFrame:
    """Load a single raw CSV and apply basic cleaning."""
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path)

    # Drop the unnamed index column that Kaggle added
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Drop rows with any null answer options
    df = df.dropna(subset=["A", "B", "C", "D"]).reset_index(drop=True)

    return df


def make_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
):
    """
    Stratified split on the 'answer' column.
    Returns (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["answer"],
        random_state=random_state,
    )

    # Second split: val vs test (50/50 of the remaining portion)
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["answer"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df):
    """Save cleaned splits to data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    print(f"Saved to {PROCESSED_DIR}")


def load_splits():
    """Load already-saved processed splits."""
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    return train_df, val_df, test_df


# ── CLI entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading raw dataset ...")
    df = load_raw("train.csv")
    print(f"  Total rows after cleaning: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Answer distribution:\n{df['answer'].value_counts().to_string()}")

    print("\nCreating 70/15/15 stratified split ...")
    train_df, val_df, test_df = make_splits(df)
    print(f"  Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    save_splits(train_df, val_df, test_df)
    print("Done.")
