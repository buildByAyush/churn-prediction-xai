# Notebooks

The original exploratory notebook (`XIA-model.ipynb`) used to prototype this
project is available on Google Colab:

🔗 https://colab.research.google.com/drive/1s1evblgGWgeAhJX3pZU8hXkP7NQzDIhM

The production-quality, modular version of this pipeline lives in `../src/`
and is orchestrated by `../main.py`. Download the notebook from the link
above (`File > Download > .ipynb`) and place it here if you want to keep a
local copy alongside the refactored code.

## 📈 Visual Results

![SHAP Summary Plot](reports/figures/shap_summary.png)
*Global feature importance — tenure, contract type, and internet service dominate.*

![SHAP Waterfall](reports/figures/shap_waterfall_customer_0.png)
*Local explanation for a single customer prediction.*