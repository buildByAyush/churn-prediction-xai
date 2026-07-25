"""Unit tests for src.data_preprocessing"""

import pandas as pd
import pytest

from src.data_preprocessing import clean_data, encode_features


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "customerID": ["001", "002"],
        "gender": ["Male", "Female"],
        "TotalCharges": ["29.85", " "],
        "Churn": ["Yes", "No"],
    })


def test_clean_data_drops_customer_id(raw_df):
    cleaned = clean_data(raw_df)
    assert "customerID" not in cleaned.columns


def test_clean_data_maps_churn_to_binary(raw_df):
    cleaned = clean_data(raw_df)
    assert set(cleaned["Churn"].unique()).issubset({0, 1})


def test_clean_data_imputes_total_charges(raw_df):
    cleaned = clean_data(raw_df)
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_encode_features_expands_columns(raw_df):
    cleaned = clean_data(raw_df)
    encoded = encode_features(cleaned)
    assert encoded.shape[1] >= cleaned.shape[1]
