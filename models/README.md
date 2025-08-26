# Models Directory

This directory is intended to store trained machine learning models for the university dropout prediction project.

## Planned Contents

### Model Files (to be added)
- **`random_forest_base.pkl`**: Best performing Random Forest base model
- **`balanced_random_forest_base.pkl`**: Best performing Balanced Random Forest model
- **`xgboost_base.pkl`**: XGBoost base model
- **`svm_base.pkl`**: Support Vector Machine base model
- **`random_forest_optimized.pkl`**: Grid Search optimized Random Forest
- **`model_metadata.json`**: Model parameters and performance metrics

### Model Performance Summary

| Model | Type | ROC-AUC | Precision | Recall | F1-Score | Status |
|-------|------|---------|-----------|--------|----------|---------|
| Random Forest | Base | **Best** | High | High | High | ⭐ Recommended |
| Balanced Random Forest | Base | **Best** | High | High | High | ⭐ Recommended |
| XGBoost | Base | Good | Good | Good | Good | Alternative |
| SVM | Base | Moderate | Moderate | Moderate | Moderate | Baseline |

### Usage

```python
import pickle
import pandas as pd

# Load the best model
with open('models/random_forest_base.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Model Characteristics

#### Random Forest (Base) - **Recommended**
- **Strengths**: Robust, interpretable, handles mixed data types
- **Use Case**: General dropout prediction with interpretability needs
- **Training**: SMOTE balanced dataset

#### Balanced Random Forest - **Recommended**
- **Strengths**: Built-in class balancing, excellent for imbalanced data
- **Use Case**: When class imbalance is primary concern
- **Training**: Original imbalanced dataset

### Storage Guidelines

- Models saved in pickle format for Python compatibility
- Include metadata file with training parameters
- Version control for model updates
- Compression for large models

### Future Enhancements

- Model versioning system
- Automated model deployment scripts
- Performance monitoring utilities
- A/B testing framework

## Notes

- Models are trained on anonymized academic data
- All models use `random_state=42` for reproducibility
- Performance metrics calculated on held-out test set
- Regular retraining recommended with new data
