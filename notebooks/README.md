# Notebooks

This directory contains the complete analysis workflow for the university dropout prediction project. The notebooks should be executed in numerical order.

## Workflow Overview

### 📁 Main Analysis (Execute in Order)

1. **01_data_preparation.ipynb**
   - Creates the target variable `RIESGO_DESERCION`
   - Defines academic delay thresholds
   - Initial data exploration

2. **02_exploratory_data_analysis.ipynb**
   - Comprehensive statistical analysis
   - Data visualization and correlation analysis
   - Distribution analysis of key variables

3. **03_data_analysis.ipynb**
   - Advanced data analysis
   - Pattern identification
   - Feature relationship exploration

4. **04_data_imputation.ipynb**
   - Missing data analysis
   - Imputation strategy implementation
   - Data quality validation

5. **05_feature_encoding_analysis.ipynb**
   - Categorical variable analysis
   - Encoding strategy planning
   - Feature transformation analysis

6. **06_feature_encoding.ipynb**
   - Variable encoding implementation
   - Feature scaling and standardization
   - Final dataset preparation

7. **07_model_training_smote.ipynb** ⭐ **Main Results**
   - SMOTE class balancing
   - Model training (RF, XGBoost, SVM, BRF)
   - Grid Search optimization
   - Model comparison and evaluation

8. **08_balanced_random_forest.ipynb**
   - Detailed Balanced Random Forest analysis
   - Specialized evaluation for imbalanced datasets
   - Advanced performance metrics

### 📁 Archive

- **06_copy_backup.ipynb**: Backup version of analysis notebook

## Key Outputs

- **Final Dataset**: `../data/processed/df_objetivo/df_escalado.xlsx`
- **Best Models**: Random Forest (base) and Balanced Random Forest
- **Evaluation Metrics**: ROC-AUC, Precision, Recall, F1-Score, Geometric Mean

## Requirements

Make sure to install all dependencies before running:
```bash
pip install -r ../requirements.txt
```

## Notes

- Each notebook includes detailed markdown explanations
- SHAP analysis is integrated for model interpretability
- All random states are set for reproducibility (random_state=42)
