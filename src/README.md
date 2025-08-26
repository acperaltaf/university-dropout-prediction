# Source Code

This directory contains utility functions and modules used throughout the project.

## Files

### 📄 eda_utils.py

Comprehensive utilities for exploratory data analysis (EDA) and dataset structure analysis.

#### Functions:

1. **`analizar_estructura_dataset(df, nombre_dataset, mostrar_jerarquia, target_candidates)`**
   - Complete automated analysis of dataset structure
   - Identifies variable types (numeric, categorical, binary, dates, IDs)
   - Analyzes missing values, duplicates, and data quality
   - Provides ML preprocessing recommendations
   - Returns structured results dictionary

2. **`analisis_rapido_dataset(df, nombre_dataset)`**
   - Quick summary analysis for basic dataset information
   - Lightweight version for rapid data exploration
   - Returns essential metrics only

3. **`comparar_datasets(df_antes, df_despues, nombre_antes, nombre_despues)`**
   - Compares two datasets and highlights differences
   - Useful for before/after preprocessing analysis
   - Shows changes in dimensions, missing values, and columns

#### Key Features:
- **Automated Type Detection**: Intelligently identifies variable types
- **ML-Ready Analysis**: Provides preprocessing recommendations
- **Comprehensive Reporting**: Detailed console output with emojis
- **Structured Output**: Returns organized results for further analysis
- **Memory Efficient**: Optimized for large datasets

#### Usage Example:
```python
import sys
sys.path.append('../src')
from eda_utils import analizar_estructura_dataset, comparar_datasets

# Complete analysis
results = analizar_estructura_dataset(
    df=my_dataset,
    nombre_dataset="Student Data",
    mostrar_jerarquia=True
)

# Compare datasets
comparar_datasets(df_original, df_processed, "Original", "Processed")
```

### 📄 shap_utils.py

Comprehensive utilities for SHAP (SHapley Additive exPlanations) analysis and model interpretability.

#### Functions:

1. **`crear_shap_plot(rf_model, X_train, X_test, plot_type, max_samples, clase)`**
   - Creates SHAP plots for model interpretation
   - Supports multiple plot types: "bar", "beeswarm", "violin"
   - Handles binary classification scenarios
   - Optimized for large datasets with sampling

2. **`crear_plots_comparativos(rf_model, X_train, X_test, max_samples, top_features)`**
   - Creates side-by-side SHAP plots for both classes
   - Facilitates comparison between positive and negative class predictions
   - Customizable feature display limit

3. **`mostrar_top_features(rf_model, X_train, X_test, max_samples, top_n)`**
   - Displays ranked feature importance based on SHAP values
   - Returns DataFrame with importance scores
   - Focuses on positive class (dropout risk) analysis

#### Key Features:
- **Memory Efficient**: Handles large datasets through sampling
- **Visualization Ready**: Optimized matplotlib integration
- **Error Handling**: Robust dimension checking and validation
- **Flexible**: Supports different model types and data formats

#### Usage Example:
```python
from src.shap_utils import crear_shap_plot, mostrar_top_features

# Create SHAP importance plot
shap_vals = crear_shap_plot(
    rf_model=trained_model,
    X_train=X_train,
    X_test=X_test,
    plot_type='bar',
    max_samples=100,
    clase=1  # Positive class (dropout risk)
)

# Get top features ranking
ranking = mostrar_top_features(
    rf_model=trained_model,
    X_train=X_train,
    X_test=X_test,
    top_n=10
)
```

## Integration

These utilities are specifically designed for the university dropout prediction project but can be adapted for other data science projects requiring:
- Automated exploratory data analysis
- Model interpretability analysis
- Dataset comparison and validation

## Dependencies

### eda_utils.py:
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical operations
- `collections.Counter`: Frequency analysis

### shap_utils.py:
- `shap`: SHAP values calculation
- `numpy`: Numerical operations  
- `pandas`: Data manipulation
- `matplotlib`: Visualization

## Notes

- All functions include comprehensive documentation
- Optimized for academic research workflows
- Includes built-in error handling and validation
- Memory-efficient processing for large datasets
- Consistent naming conventions and output formats
