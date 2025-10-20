# 🫀 Heart Stroke Prediction Using Quantitative Data

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-brightgreen)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Binary classification project to predict stroke risk from patient health records using multiple ML classifiers, hyperparameter tuning, and SMOTE for class imbalance.

## 📌 Overview

This project analyses the [Kaggle Healthcare Stroke Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) (5,110 patients) and builds a predictive model to identify patients at high risk of stroke based on demographic and clinical features.

**Key challenge:** Severe class imbalance (~4.9% stroke positive) — addressed with **SMOTE (Synthetic Minority Oversampling Technique)**.

## 🗂️ Repository Structure

```
Heart-Stroke-Prediction-Using-Quantitative-Data/
├── Final_Heart_Stroke_Prediction_Using_Quantitative_Data.ipynb  # Full notebook
├── healthcare-dataset-stroke-data.csv                           # Patient dataset
├── requirements.txt                                             # Dependencies
└── README.md                                                    # This file
```

## 🩺 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Patient age |
| gender | Categorical | Male / Female / Other |
| hypertension | Binary | 1 if has hypertension |
| heart_disease | Binary | 1 if has heart disease |
| ever_married | Categorical | Yes / No |
| work_type | Categorical | Private / Self-employed / Govt / etc. |
| Residence_type | Categorical | Urban / Rural |
| avg_glucose_level | Numeric | Average blood glucose level |
| bmi | Numeric | Body mass index |
| smoking_status | Categorical | Never / Formerly / Smokes / Unknown |
| **stroke** | **Binary** | **Target: 1 = stroke, 0 = no stroke** |

## 🔬 Methodology

1. **EDA** — class distribution, correlation heatmap, feature distributions
2. **Preprocessing** — label encoding, null imputation (BMI median fill), StandardScaler
3. **Class Balancing** — SMOTE applied on training set only
4. **Modelling** — 4 classifiers with RandomizedSearchCV hyperparameter tuning
5. **Evaluation** — Confusion matrix, precision, recall, F1-score

## 🧠 Models Trained

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline with L1/L2 regularisation |
| Decision Tree | Entropy criterion; depth tuned via RandomizedSearchCV |
| **Random Forest** | Ensemble of 100 trees — best overall recall |
| SVM | RBF & linear kernels; C tuned |
| XGBoost | Gradient boosting with n_estimators & learning rate tuning |

## 📊 Results

| Model | Accuracy | Recall (Stroke) | F1 (Stroke) |
|-------|----------|-----------------|-------------|
| Logistic Regression | ~78% | ~72% | ~0.70 |
| Decision Tree | ~81% | ~76% | ~0.74 |
| **Random Forest** | **~85%** | **~80%** | **~0.78** |
| SVM | ~82% | ~74% | ~0.72 |
| XGBoost | ~84% | ~78% | ~0.76 |

> **Recall** is the most important metric for medical diagnosis — we want to minimise false negatives (missed strokes).

## 🔮 Future Work

- [ ] Explore Neural Network classifiers (MLP)
- [ ] Feature importance analysis with SHAP
- [ ] Calibration curves for probability reliability
- [ ] Deploy as a clinical risk scoring API

## ⚙️ Setup

```bash
pip install -r requirements.txt
jupyter notebook "Final_Heart_Stroke_Prediction_Using_Quantitative_Data.ipynb"
```

## 👩‍💻 Author

**Devyani Deore** — [github.com/DevyaniD19](https://github.com/DevyaniD19)

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
