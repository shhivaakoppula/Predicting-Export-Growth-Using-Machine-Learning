#  Predicting Export Growth with Machine Learning
### A 34-Year Longitudinal Study of Country-Level Trade Dynamics (1988–2021)

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikitlearn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![Dataset](https://img.shields.io/badge/Dataset-WITS%20World%20Bank-blue)](https://wits.worldbank.org/)

---

##  Overview

Can we predict whether a country's exports will grow next year — using only publicly available trade and tariff data?

This project answers that question by building an end-to-end machine learning pipeline on a **34-year longitudinal dataset** of country-level trade statistics from the World Integrated Trade Solution (WITS) platform. It covers **265 countries** and **8,096 country-year observations** from 1988 to 2021, capturing three full economic cycles including the 2008 Global Financial Crisis and the COVID-19 trade shock.

Export growth is framed as a **binary classification problem**: predict whether a country's exports in year *t+1* will exceed year *t*. Eleven machine learning classifiers are benchmarked under rigorous temporal validation, and the results are analyzed both statistically and through an economic policy lens.

This work was submitted as a term paper for the Machine Learning course at **SRH University**.

---

##  Key Results

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **AdaBoost** | 45.82% | 0.335 | **0.7923** ⭐ |
| Gradient Boosting | 44.60% | 0.273 | 0.7683 |
| **Logistic Regression** | **58.66%** | **0.593** | 0.7629 ⭐ |
| SVM (RBF) | 41.14% | 0.195 | 0.7551 |
| Random Forest | 49.69% | 0.422 | 0.7525 |
| XGBoost | 49.69% | 0.450 | 0.7394 |
| LightGBM | 47.86% | 0.407 | 0.7305 |
| Decision Tree | 53.97% | 0.525 | 0.5961 |

> **AdaBoost** achieves the best ROC-AUC (0.79), making it ideal for probability ranking.  
> **Logistic Regression** achieves the best accuracy + F1, making it the most deployable for binary prediction.

### 🔑 Top Predictive Features
1. `MFN_Simple_Average` — WTO-schedule tariff level
2. `Export_Lag2` — Two-year lagged export momentum
3. `Log_Export` — Scale-invariant export volume
4. `AHS_Simple_Average` — Applied tariff level (trade openness)
5. `Export_Import_Ratio` — Export orientation ratio

---

## 📂 Project Structure

```
Export_Growth_Prediction/
│
├── Trade_Export_Prediction.ipynb     # Main notebook — full pipeline
├── Data - 34_years_world_export_import_dataset.csv  # Raw dataset (WITS / World Bank)
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── paper/
    └── Machine_learning_Paper_Trade_Export_Prediction_.pdf  # Full research paper
```

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| Source | [World Integrated Trade Solution (WITS)](https://wits.worldbank.org/) |
| Coverage | 265 countries, 1988–2021 |
| Observations | 8,096 country-year rows |
| Raw Features | 33 |
| Engineered Features | 18 |
| Target Variable | Binary: export growth in next year (1) or not (0) |

### Feature Categories
- **Trade Flows** — Export & Import values (US$ Thousand)
- **AHS Tariff Variables** — Applied Harmonized System rates (12 indicators)
- **MFN Tariff Variables** — Most Favoured Nation rates (12 indicators)
- **Comparative Advantage** — Revealed Comparative Advantage (RCA) index
- **Growth Indicators** — World GDP growth, Country GDP growth

---

## ⚙️ Pipeline

```
Raw Data (8,096 × 33)
        │
        ▼
Data Cleaning
  └─ Standardize column names
  └─ Remove zero/null export rows
  └─ IQR-based outlier capping (3× IQR)
  └─ Median imputation per country group
        │
        ▼
Feature Engineering (→ 18 features)
  └─ Log_Export, Log_Import
  └─ Trade_Balance_Ratio, Export_Import_Ratio
  └─ Export_YoY_Growth (target driver)
  └─ Export_Lag1, Export_Lag2
  └─ Export_Rolling3
  └─ Surplus_Flag, Decade
  └─ Tariff features (AHS, MFN averages)
        │
        ▼
Target Construction
  └─ Binary: next-year export > current-year export
  └─ Final: 5,984 observations | 60% growth / 40% no-growth
        │
        ▼
Time-Based Train/Test Split
  └─ Train: 1988–2015 (5,493 records)
  └─ Test:  2016–2021 (491 records)
        │
        ▼
SMOTE (training set only)
  └─ Balanced: 3,313 samples per class
        │
        ▼
Benchmark: 11 Classifiers
  └─ Logistic Regression, Decision Tree, Random Forest,
     Gradient Boosting, Extra Trees, AdaBoost, XGBoost,
     LightGBM, KNN, Naive Bayes, SVM (RBF)
        │
        ▼
Hyperparameter Tuning (best model via RandomizedSearchCV)
        │
        ▼
Evaluation: Accuracy · F1-Score · ROC-AUC
```

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/shhivaakoppula/Trade-Export-Prediction.git
cd Trade-Export-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook Trade_Export_Prediction.ipynb
```

---

## 📦 Requirements

```
pandas>=2.1
numpy>=1.26
scikit-learn>=1.4
xgboost>=2.0
lightgbm>=4.6
imbalanced-learn>=0.11
matplotlib>=3.8
seaborn>=0.13
jupyter
```

---

##  Business Questions Explored

The EDA section systematically answers 25 analytical questions about global trade dynamics, including:

- Which countries dominate global exports on average?
- How did global export volumes trend from 1988 to 2021?
- Which economies had the most surplus years?
- How do tariff levels correlate with export growth probability?
- How did export growth rates distribute across decades?
- Which countries recovered fastest after trade contractions?
- How does duty-free share impact growth probability?

---

## 💡 Key Insights

**For policy analysts:**  
Tariff variables (MFN and AHS Simple Average) are the strongest predictors of export growth — consistent with decades of WTO research showing that trade liberalization drives export expansion.

**For trade finance professionals:**  
AdaBoost's ROC-AUC of 0.79 provides genuine discriminative power for prioritizing export credit guarantees across country portfolios.

**For ML practitioners:**  
Logistic Regression matches or beats complex ensemble methods on macroeconomic data with distribution shift — a reminder that lower-variance linear models are often more robust to structural breaks than high-capacity non-linear models.

---

## 🏛️ Academic Context

| | |
|---|---|
| **Author** | Shiva Goud Koppula |
| **Program** | M.Sc. Data Science (Digital Health), SRH University |
| **Subject** | Machine Learning |
| **Examiner** | Prof. Dr. Philipp Unberath |
| **Submission** | March 2026 |

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/shivagoudkoppula)
[![GitHub](https://img.shields.io/badge/GitHub-shhivaakoppula-black?logo=github)](https://github.com/shhivaakoppula)

---

*If you find this project useful, a ⭐ on GitHub is appreciated!*
