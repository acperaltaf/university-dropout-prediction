# Visualizations Directory

This directory contains  plots and charts generated during the university dropout prediction analysis.

## 📊 Image Inventory

### Exploratory Data Analysis
- **`01_target_balance_analysis.png`** - Target variable distribution before and after SMOTE balancing
- **`02_feature_distributions.png`** - Feature distributions using boxplots for key variables

### Model Performance - ROC Curves
- **`03_roc_curve_rf.png`** - ROC curve for Random Forest model
- **`04_roc_curve_xgboost.png`** - ROC curve for XGBoost model

### Model Performance - Confusion Matrices
- **`Matriz de confusión - random forest.png`** - Confusion matrix for Random Forest
- **`Matriz de confusión - xgboost.png`** - Confusion matrix for XGBoost

### Combined Performance Analysis
- **`Matrices de confusión y curvas ROC - RF y BRFC.png`** - Side-by-side comparison of Random Forest and Balanced Random Forest (confusion matrices + ROC curves)
- **`MC y ROC - xgboost.png`** - Combined confusion matrix and ROC curve for XGBoost

### Feature Importance & Interpretability
- **`importancias_RF_base.png`** - Feature importance plot from base Random Forest model
- **`SHAP - BRFC base.png`** - SHAP analysis for Balanced Random Forest Classifier
- **`SHAP - xgboost_optimizado.png`** - SHAP analysis for optimized XGBoost model

## 🎯 Key Insights

### Model Performance Summary
- Random Forest and Balanced Random Forest show excellent performance
- XGBoost demonstrates competitive results across all metrics
- Combined visualizations effectively showcase model comparisons

### Feature Analysis
- Academic performance variables (PAPA, AVANCE_CARRERA) are most predictive
- SHAP analysis reveals detailed feature impact patterns
- Feature importance plots confirm model interpretability

### Publication Quality
All visualizations are high-resolution and suitable for:
- Academic papers and conferences
- Technical presentations
- Model documentation and reports