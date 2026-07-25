"""
evaluate.py

Computes and reports standard classification metrics for a trained model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None

    def __str__(self) -> str:
        lines = [
            f"{self.model_name} Results",
            f"Accuracy:  {self.accuracy:.4f}",
            f"Precision: {self.precision:.4f}",
            f"Recall:    {self.recall:.4f}",
            f"F1-score:  {self.f1:.4f}",
        ]
        if self.roc_auc is not None:
            lines.append(f"ROC-AUC:   {self.roc_auc:.4f}")
        return "\n".join(lines)


def evaluate_model(model, X_test, y_test, model_name: str, has_proba: bool = True) -> EvalResult:
    """Evaluate a fitted classifier on held-out test data.

    Args:
        model: Fitted classifier with .predict() (and .predict_proba() if has_proba).
        X_test: Test features.
        y_test: True test labels.
        model_name: Display name for logging/reporting.
        has_proba: Whether to compute ROC-AUC via predict_proba.

    Returns:
        EvalResult with all computed metrics.
    """
    y_pred = model.predict(X_test)

    roc_auc = None
    if has_proba:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)

    result = EvalResult(
        model_name=model_name,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc,
    )

    logger.info("\n%s", result)
    return result
