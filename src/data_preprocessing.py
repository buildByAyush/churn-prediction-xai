"""
data_preprocessing.py

Handles loading, cleaning, and feature encoding for the Telco Customer
Churn dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw Telco churn CSV into a DataFrame.

    Args:
        csv_path: Path to the raw CSV file.

    Returns:
        Raw, unmodified DataFrame.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset: %d rows, %d columns", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset.

    - Drops the non-predictive customerID column.
    - Coerces TotalCharges to numeric, imputing missing values with the median.
    - Maps the target variable Churn from Yes/No to 1/0.

    Args:
        df: Raw DataFrame from `load_data`.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    if "customerID" in df.columns:
        df.drop(columns="customerID", inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    if n_missing:
        logger.info("Imputing %d missing TotalCharges values with median", n_missing)
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features (drop_first=True to avoid
    multicollinearity).

    Args:
        df: Cleaned DataFrame.

    Returns:
        Fully numeric, model-ready DataFrame.
    """
    encoded = pd.get_dummies(df, drop_first=True)
    logger.info("Encoded features: %d -> %d columns", df.shape[1], encoded.shape[1])
    return encoded


def preprocess_pipeline(csv_path: str | Path) -> pd.DataFrame:
    """Run the full load -> clean -> encode pipeline in one call."""
    df = load_data(csv_path)
    df = clean_data(df)
    df = encode_features(df)
    return df
