# 📊 Telco Churn Prediction Pipeline

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/sklearn-latest-orange.svg)
![XGBoost](https://img.shields.io/badge/xgboost-latest-green.svg)

A production-ready Machine Learning pipeline designed to predict customer churn. This project emphasizes strict reproducibility, modular design, and the elimination of train-serving skew.

---

## 🔬 Core Engineering Highlights

* **DAG-Ready Modular Architecture:** Decoupled data preprocessing, training, and evaluation scripts to facilitate future orchestration (e.g., Apache Airflow) and CI/CD automation.
* **Leakage-Free Feature Engineering:** The statistical state (mean/std) of the `StandardScaler` is fitted exclusively on the training split and serialized to prevent data leakage during inference.
* **Estimator Benchmarking:** An automated tournament-style pipeline evaluates Logistic Regression, Random Forest, and XGBoost, auto-selecting the champion model based on the test ROC-AUC score.

---

## 🛠️ Tech Stack
* Core: Python 3.9
* Data Manipulation: Pandas, NumPy
* Machine Learning: Scikit-Learn, XGBoost
* Artifact Persistence: Joblib

## 📁 Project Structure

```text
project_churn/
│
├── data/
│   └── churn.csv                 # Dataset (not included in repo)
│
├── notebooks/
│   ├── eda.ipynb                 # Exploratory Data Analysis
│   ├── feature_engineering.ipynb # Feature engineering & scaling prototypes
│   └── modeling.ipynb            # Model experimentation & comparison
│
├── src/
│   ├── preprocessing.py          # Feature extraction, binning & dummy mapping
│   ├── train.py                  # Model training and artifact persistence
│   ├── evaluate.py               # Cold-run model performance evaluation
│   ├── scaler.pkl                # Serialized StandardScaler state
│   └── model.pkl                 # Persisted champion model (XGBoost/RF)
│
└── README.md

