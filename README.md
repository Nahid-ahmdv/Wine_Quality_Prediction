# **🍷 Wine Quality Prediction (Regression)**  
**OLS vs Lasso vs Random Forest, with model explainability (Permutation Importance, PDP, SHAP)**

This project predicts **white wine quality** from **11 physicochemical lab measurements** and compares classic linear baselines with a non-linear ensemble model. Beyond “which model wins?”, the main focus is **interpretability**: understanding *why* the model predicts a certain score.


## **Highlights**

- Built a clean, leakage-aware evaluation pipeline (train/holdout split + cross-validation)
- Compared:
  - Baseline (predict-the-mean)
  - **OLS (Linear Regression)**
  - **Lasso / LassoCV**
  - **Random Forest** (default + tuned)
- Interpreted models using:
  - **Permutation importance** for “impact on error” rankings
  - **PDP (Partial Dependence Plots)** for global effect shapes
  - **SHAP** for global + local explanations


## **Full write-up**
📓 Project notebook: `wine_quality_ols_lasso_rf_detailed.ipynb`  
📝 Medium article: https://medium.com/@nahid.nm57/wine-quality-prediction-3d1c6f3b961a
At the end of the Medium article, I link back to this repo for full reproducibility.


## **Dataset**

- **Dataset:** White Wine Quality (commonly attributed to the UCI Wine Quality dataset; also mirrored in mlcourse.ai resources)
- **Rows / Features:** 4,898 wines, 11 numeric features + target (`quality`)
- **Format:** `winequality-white.csv` (semicolon-separated)

The notebook loads data directly from:
- `https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/winequality-white.csv`

> Note: The dataset includes duplicate measurements (~19% in my run). I keep them (as-is) for a realistic baseline, but deduplication is a possible next step.


## **Problem Setup**

- **Task:** Regression (predict `quality`, a 0–10 score)
- **Split:** 70% train / 30% holdout (`random_state=17`)
- **Model selection metric:** **5-fold CV MSE** (also reported as CV RMSE)
- **Scaling:** Applied via `Pipeline` for linear models (scaling happens inside CV folds)


## **Models**

### **1) Baseline**
- `DummyRegressor(strategy="mean")`
- Sanity check: if a model can’t beat this, it’s not learning.

### **2) OLS (Linear Regression)**
- Transparent benchmark
- Interpretable via standardized coefficients + permutation importance

### **3) Lasso + LassoCV**
- Tests whether regularization and sparsity improve generalization
- Tuned with cross-validation (LassoCV) to select `alpha`

### **4) Random Forest**
- Captures non-linearity + interactions automatically
- Tuned via GridSearchCV over:
  - `max_depth`
  - `max_features`

## **Explainability Tools Used**

- **Permutation importance**  
  Answers: *“Which features increase error the most if I shuffle them?”*  
  (Model-agnostic, tied directly to predictive performance.)

- **PDP (Partial Dependence Plots)**  
  Answers: *“On average, what is the effect shape of this feature?”*  
  (Great for checking monotonic vs. threshold vs. curved relationships.)

- **SHAP**  
  Answers: *“For this specific wine, what pushed the prediction up/down?”*  
  (Global + local interpretability. Requires `shap` installed.)


## **Results (from my run, `random_state=17`)**

Model comparison (sorted by **CV MSE**):

| Model | CV MSE | CV RMSE | Holdout MSE | Holdout RMSE | Holdout R² |
|---|---:|---:|---:|---:|---:|
| **RF (tuned)** | 0.3979 | 0.6308 | 0.3658 | 0.6048 | 0.5304 |
| RF (default) | 0.4143 | 0.6437 | 0.3708 | 0.6089 | 0.5240 |
| LassoCV (tuned alpha) | 0.5604 | 0.7486 | 0.5833 | 0.7637 | 0.2511 |
| OLS (LinearRegression) | 0.5605 | 0.7486 | 0.5842 | 0.7644 | 0.2499 |
| Lasso (alpha=0.01) | 0.5656 | 0.7521 | 0.5737 | 0.7574 | 0.2635 |
| Baseline (mean) | 0.7866 | 0.8869 | 0.7789 | 0.8825 | -0.0000 |

**Best tuned RF parameters (my run):**
- `max_depth = 21`
- `max_features = 6`

**Top drivers (tuned RF permutation importance, ΔMSE):**
- `alcohol`
- `volatile acidity`
- `free sulfur dioxide`

> Your results should match closely if you keep the same random seed and environment versions.


## **Repository structure**

```
.
├── README.md
├── requirements.txt
├── wine_quality_ols_lasso_rf_detailed.ipynb
└── Data/
    └── winequality-white.csv   # not committed

```

## **How to Run**

### **1) Create an environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### **2) Install dependencies**

```bash
pip install -r requirements.txt
```

### **3) Launch Jupyter**

```bash
jupyter notebook
```

Then open and run: `wine_quality_ols_lasso_rf_detailed.ipynb`

## **Reproducibility Notes**

* Set `RANDOM_STATE = 17` (used throughout the notebook)
* Use `Pipeline` for scaling inside cross-validation
* Random Forest results can vary slightly across platforms/versions


## **Next Improvements**

* Deduplicate repeated measurements and compare performance
* Try boosted trees (XGBoost / LightGBM / CatBoost)
* Add uncertainty estimates for predictions
* Treat `quality` as ordinal and compare ordinal-aware approaches

---

⭐ Hope you find this project as instructive and insightful as I did while building it!
