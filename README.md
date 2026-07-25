# Explainable AI Framework for Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Published-red)](#-publication)

> A production-style ML pipeline that predicts telecom customer churn with **XGBoost** and explains every prediction — globally and per-customer — using **SHAP (SHapley Additive exPlanations)**. Published/presented at an international conference on Intelligent Computing.

---

## 📌 Overview

Telecom companies lose 15–25% of customers annually, and acquiring a new customer costs 5–7x more than retaining one. Most churn models are either accurate-but-opaque or interpretable-but-weak. This project closes that gap: it pairs a high-performance gradient-boosted classifier with a rigorous, game-theoretic explainability layer, so retention teams get **both** a ranked risk score and the _exact reasons_ behind it for every customer.

## 🎯 Key Results

| Model                          | Accuracy | Precision | Recall | F1-Score | ROC-AUC   |
| ------------------------------ | -------- | --------- | ------ | -------- | --------- |
| Logistic Regression (baseline) | 80.3%    | 0.652     | 0.556  | 0.600    | 0.792     |
| **XGBoost (final)**            | 79.1%    | 0.625     | 0.529  | 0.573    | **0.834** |

XGBoost was selected as the production model despite marginally lower raw accuracy, because **ROC-AUC is the operative business metric** here — it governs how well the model ranks at-risk customers, which directly determines the cost-effectiveness of retention campaigns.

**Top churn drivers identified via SHAP:**

1. **Tenure** — new customers (<12 months) churn disproportionately more
2. **Contract type** — month-to-month customers are highest risk; 2-year contracts strongly reduce churn
3. **Internet service (Fiber optic)** — mixed signal, indicates a dissatisfied sub-segment
4. **Monthly charges** — higher bills correlate with higher churn probability

## 🧠 Architecture

```
Raw Data (7,043 records, 21 features)
        │
        ▼
┌───────────────────┐
│ Data Preprocessing │  → type casting, median imputation, target encoding
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Feature Encoding   │  → One-Hot Encoding (20 → 30 features)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Train/Test Split   │  → 80/20 stratified split
└───────────────────┘
        │
        ▼
┌───────────────────┐         ┌───────────────────┐
│ Logistic Regression│         │      XGBoost       │
│     (baseline)      │         │  (n=200, depth=5)   │
└───────────────────┘         └───────────────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │  SHAP TreeExplainer │
                              │ (global + local exp) │
                              └───────────────────┘
                                        │
                                        ▼
                              Business-ready insights
```

## 🛠️ Tech Stack

| Category       | Tools                                                                                               |
| -------------- | --------------------------------------------------------------------------------------------------- |
| Language       | Python 3.10+                                                                                        |
| Data           | pandas, NumPy                                                                                       |
| Modeling       | scikit-learn, XGBoost                                                                               |
| Explainability | SHAP                                                                                                |
| Visualization  | matplotlib, seaborn                                                                                 |
| Dataset        | [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle) |

## 📂 Project Structure

```
churn-prediction-xai/
├── src/
│   ├── data_preprocessing.py   # loading, cleaning, encoding
│   ├── train.py                 # model training (LR + XGBoost)
│   ├── evaluate.py              # metrics computation
│   └── explain.py               # SHAP global + local explanations
├── notebooks/
│   └── XIA-model.ipynb          # original exploratory notebook
├── reports/
│   └── figures/                 # generated plots (SHAP, ROC, etc.)
├── models/                      # saved model artifacts (.pkl)
├── tests/                       # unit tests
├── main.py                      # end-to-end pipeline entry point
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

```bash
git clone https://github.com/<your-username>/churn-prediction-xai.git
cd churn-prediction-xai

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

python main.py --data data/telco_churn.csv
```

This runs the full pipeline: preprocessing → train Logistic Regression + XGBoost → evaluate → generate SHAP summary & waterfall plots into `reports/figures/`.

## 📊 Sample Output

```
XGBoost Results
Accuracy:  0.7906
Precision: 0.6246
Recall:    0.5294
F1-score:  0.5731
ROC-AUC:   0.8342
```

## 📄 Publication

This work was submitted and presented as a mini/research project at an **International Conference on Intelligent Computing**, under the title:

> _"Explainable AI Framework for Customer Churn Prediction in Telecommunications: A Comparative Study using XGBoost and SHAP"_

📎 Paper link: `<add DOI / conference proceedings link here>`

If you use this work, please cite:

```bibtex
@inproceedings{jaiswal2026churnxai,
  title     = {Explainable AI Framework for Customer Churn Prediction in Telecommunications: A Comparative Study using XGBoost and SHAP},
  author    = {Jaiswal, Ayush and Ayush},
  booktitle = {International Conference on Intelligent Computing},
  year      = {2026}
}
```

## 🔮 Future Work

- Temporal modeling of customer interaction sequences
- SMOTE / ADASYN for class-imbalance handling
- Deployment as a real-time FastAPI inference + explanation service
- Counterfactual explanations for actionable retention interventions

## 👤 Authors

- **Ayush** — [LinkedIn](https://www.linkedin.com/in/ayush-sharma-52196a2a5?utm_source=share_via&utm_content=profile&utm_medium=member_android)
- **Ayush Jiaswal**

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
