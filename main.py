"""
main.py

End-to-end entry point for the Churn Prediction XAI pipeline:
load -> clean -> encode -> split -> train (LR + XGBoost) -> evaluate -> explain.

Usage:
    python main.py --data data/telco_churn.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.data_preprocessing import clean_data, encode_features, load_data
from src.evaluate import evaluate_model
from src.explain import compute_shap_values, save_global_summary, save_local_waterfall
from src.train import split_data, train_logistic_regression, train_xgboost

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Churn Prediction XAI Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to the Telco churn CSV file")
    parser.add_argument(
        "--reports-dir", type=str, default="reports/figures", help="Directory to save SHAP plots"
    )
    parser.add_argument(
        "--skip-shap", action="store_true", help="Skip SHAP explainability step (faster runs)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("=== Step 1/4: Data Preprocessing ===")
    df = load_data(args.data)
    df = clean_data(df)
    df = encode_features(df)

    logger.info("=== Step 2/4: Train/Test Split ===")
    X_train, X_test, y_train, y_test = split_data(df)

    logger.info("=== Step 3/4: Model Training & Evaluation ===")
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, model_name="Logistic Regression")

    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")

    if not args.skip_shap:
        logger.info("=== Step 4/4: SHAP Explainability ===")
        shap_values, X_test_f = compute_shap_values(xgb_model, X_train, X_test)
        save_global_summary(shap_values, X_test_f, args.reports_dir)
        save_local_waterfall(shap_values, args.reports_dir, index=0)

    logger.info("Pipeline complete. Reports saved to: %s", Path(args.reports_dir).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
