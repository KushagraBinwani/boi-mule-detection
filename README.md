# Bank of India Mule Account Detection

## Overview

This project was developed for the Bank of India Mule Account Detection Hackathon.

The objective was to identify fraudulent mule accounts from highly imbalanced banking data using machine learning. The project combines exploratory data analysis, feature engineering, gradient boosting models, neural networks, ensembling, and error analysis to improve fraud detection performance.

Starting from a CatBoost baseline that detected only 7 of 16 fraud cases in the evaluation set, the final Neural Network model successfully detected 14 of 16 fraud cases (87.5% recall).

---

## Problem Statement

Mule accounts are bank accounts used to receive, transfer, or conceal funds obtained through fraudulent activities.

Detecting these accounts is challenging because:

* Fraud cases are extremely rare compared to legitimate accounts.
* Fraudsters often mimic normal customer behavior.
* Traditional accuracy metrics can be misleading due to severe class imbalance.

The goal of this project was to develop a machine learning pipeline capable of identifying fraudulent accounts while maintaining strong recall.

---

## Project Workflow

### 1. Exploratory Data Analysis

The project began with extensive exploratory analysis to understand:

* Feature distributions
* Fraud prevalence across feature ranges
* Missing value patterns
* Relationships between categorical variables and fraud

Several features repeatedly appeared informative:

* F115
* F2582
* F2956
* F531
* F2737

A key discovery was that missingness itself carried predictive information, leading to the creation of missing-value indicator features.

---

### 2. Feature Engineering

Domain-inspired features were developed from patterns observed during exploratory analysis.

Examples include:

* F115_high
* F2582_hot
* F2956_hot
* F531_hot
* F2737_safe
* F2582_F531
* F2956_F115
* F2582_pos_F2956_low
* fraud_cluster_1

Additional binary indicators were created for missing values in historical and family-related feature groups.

---

### 3. Model Development

Multiple model families were evaluated.

| Model          | Frauds Detected |
| -------------- | --------------- |
| CatBoost       | 7 / 16          |
| XGBoost        | 12 / 16         |
| LightGBM       | 12 / 16         |
| Ensemble       | 13 / 16         |
| Neural Network | 14 / 16         |

The Neural Network achieved the strongest fraud recall, while the Ensemble model reduced false positives and provided a stronger precision-recall tradeoff.

---

### 4. Error Analysis

The remaining missed fraud cases were investigated using:

* Nearest-neighbor analysis
* Isolation Forest anomaly detection
* Feature comparison

Results showed that the remaining frauds closely resembled legitimate accounts and were not strongly anomalous, suggesting that additional information may be required to detect them reliably.

---

## Results

### Model Progression

![Model Comparison](figures/model_comparison.png)

### Neural Network Training

![Neural Network Training Curve](figures/nn_training_curve.png)

### XGBoost Feature Importance

![XGBoost Feature Importance](figures/xgb_feature_importance.png)

---

## Repository Structure

```text
data/
├── raw/
├── processed/

figures/

notebooks/
├── 01_eda.ipynb
├── 02_catboost_baseline.ipynb
├── 03_xgboost.ipynb
├── 04_lightgbm.ipynb
├── 05_neural_networks.ipynb
├── 06_ensemble.ipynb
└── 07_error_analysis.ipynb

reports/
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* TensorFlow / Keras
* Matplotlib
* SHAP

---

## Key Takeaways

The largest performance gains did not come from changing algorithms alone.

The most important improvements resulted from:

* Careful exploratory data analysis
* Iterative feature engineering
* Investigation of model failures
* Combining multiple machine learning approaches

This project demonstrates how domain-driven feature engineering can significantly improve fraud detection performance in highly imbalanced datasets.

