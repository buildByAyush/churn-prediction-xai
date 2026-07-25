"""
train.py

Trains the baseline Logistic Regression model and the primary XGBoost
model for churn classification.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
)


def split_data(df: pd.DataFrame, target_col: str = "Churn"):
    """Stratified 80/20 train-test split."""
    X = df.drop(columns=target_col)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train the interpretable baseline model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete")
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """Train the primary high-performance model."""
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    logger.info("XGBoost training complete")
    return model
