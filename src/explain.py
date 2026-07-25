"""
explain.py

Generates global and local SHAP explanations for the trained XGBoost model
and saves the resulting plots to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import shap

logger = logging.getLogger(__name__)


def compute_shap_values(model, X_train, X_test):
    """Fit a SHAP TreeExplainer and compute SHAP values for the test set."""
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    logger.info("Computed SHAP values for %d test samples", len(X_test))
    return shap_values, X_test


def save_global_summary(shap_values, X_test, out_dir: str | Path) -> Path:
    """Save the SHAP global feature-importance summary plot."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shap_summary.png"

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP summary plot to %s", out_path)
    return out_path


def save_local_waterfall(shap_values, out_dir: str | Path, index: int = 0) -> Path:
    """Save a SHAP waterfall plot explaining a single prediction.

    Args:
        shap_values: SHAP values from `compute_shap_values`.
        out_dir: Directory to write the plot to.
        index: Index of the test-set sample to explain.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shap_waterfall_customer_{index}.png"

    shap.plots.waterfall(shap_values[index], show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP waterfall plot to %s", out_path)
    return out_path
