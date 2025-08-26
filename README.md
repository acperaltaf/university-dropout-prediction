# University Dropout Prediction

A comprehensive machine learning project for predicting university student dropout risk using academic, administrative, and socioeconomic data.

## 📊 Project Overview

This project implements a binary classification system to predict student dropout risk using multiple machine learning algorithms. The study compares base models with their optimized versions and identifies the most effective approaches for early intervention.

## 🔍 Key Findings

- **Best Performing Models**: Random Forest (base) and Balanced Random Forest Classifier
- **Unexpected Result**: Base models outperformed Grid Search optimized versions
- **Algorithms Evaluated**: Random Forest, XGBoost, SVM, Balanced Random Forest
- **Techniques Used**: SMOTE for class balancing, SHAP for model interpretability

## 📁 Project Structure

```
university-dropout-prediction/
│
├── notebooks/                     # Jupyter notebooks (main workflow)
│   ├── 01_data_preparation.ipynb         # Target variable creation
│   ├── 02_exploratory_data_analysis.ipynb # Descriptive analysis
│   ├── 03_data_analysis.ipynb            # Data analysis
│   ├── 04_data_imputation.ipynb          # Missing data handling
│   ├── 05_feature_encoding_analysis.ipynb # Encoding analysis
│   ├── 06_feature_encoding.ipynb         # Variable encoding
│   ├── 07_model_training_smote.ipynb     # Main model training
│   ├── 08_balanced_random_forest.ipynb   # Balanced RF analysis
│   └── archive/                          # Backup notebooks
│
├── data/
│   ├── raw/                       # Original datasets
│   └── processed/                 # Processed datasets
│       └── df_objetivo/           # Target variable datasets
│
├── src/                          # Source code
│   └── shap_utils.py             # SHAP utilities for interpretability
│
├── models/                       # Trained models (to be added)
├── results/                      # Results and visualizations
├── docs/                         # Additional documentation
│
├── requirements.txt              # Project dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
```bash
git clone https://github.com/acperaltaf/university-dropout-prediction.git
cd university-dropout-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter:
```bash
jupyter notebook
```

4. Run notebooks in sequence (01 → 02 → ... → 08)

## 📋 Methodology

### 1. Data Preparation
- **Target Variable Creation**: Risk threshold defined using 80th percentile of academic delay
- **Feature Engineering**: Academic progress indicators and socioeconomic variables

### 2. Exploratory Data Analysis
- Statistical analysis of student academic patterns
- Correlation analysis between variables
- Distribution analysis of risk factors

### 3. Data Preprocessing
- **Missing Data**: Systematic imputation strategy
- **Feature Encoding**: Categorical variable transformation
- **Scaling**: Standardization of numerical features
- **Class Balancing**: SMOTE technique for imbalanced datasets

### 4. Model Development
- **Base Models**: Random Forest, XGBoost, SVM, Balanced Random Forest
- **Optimization**: Grid Search hyperparameter tuning
- **Evaluation**: ROC-AUC, Precision, Recall, F1-Score, Geometric Mean

### 5. Model Interpretability
- **SHAP Analysis**: Feature importance and impact visualization
- **Comparative Analysis**: Model performance across different metrics

## 📊 Results Summary

| Model | Type | ROC-AUC | Key Characteristics |
|-------|------|---------|-------------------|
| Random Forest | Base | **Best** | Robust, interpretable |
| Balanced Random Forest | Base | **Best** | Handles class imbalance well |
| Random Forest | Optimized | Lower | Overfitted to validation set |
| XGBoost | Base/Optimized | Good | Strong gradient boosting |
| SVM | Base/Optimized | Moderate | Less suitable for this dataset |

## 🛠️ Key Features

- **Comprehensive Pipeline**: From raw data to model deployment
- **Multiple Algorithms**: Comparative analysis of 4 different ML approaches
- **Class Imbalance Handling**: SMOTE and Balanced algorithms
- **Model Interpretability**: SHAP values for feature importance
- **Reproducible Research**: Well-documented notebook workflow

## 📈 Model Interpretability

The project includes detailed SHAP analysis to understand:
- Most important features for dropout prediction
- Feature impact on individual predictions
- Comparative feature importance across models



## 📄 License

This project is available for academic and research purposes.


