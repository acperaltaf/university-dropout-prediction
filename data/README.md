# Data Directory

This directory contains all datasets used in the university dropout prediction project.

## Structure

### 📁 raw/
- **Purpose**: Original, unmodified datasets
- **Contents**: 
  - `2024-2.xlsx`: Original academic records dataset
  - `REPORTE HISTORIAS ACADEMICAS BLOQUEDAS_anonimizado.xlsx`: Anonymized academic history report
- **Important**: These files should never be modified

### 📁 processed/
- **Purpose**: Cleaned, transformed, and processed datasets
- **Contents**: Various stages of data processing

#### 📁 processed/df_objetivo/
Contains datasets at different processing stages:

- **`2024-2.xlsx`**: Copy of original dataset
- **`df_objetivo_riesgo_desercion.xlsx`**: Dataset with target variable created
- **`df_objetivo_riesgo_real.xlsx`**: Real risk assessment dataset
- **`df_objetivo_imputado.xlsx`**: Dataset after missing value imputation
- **`df_objetivo_imputado.csv`**: CSV version of imputed dataset
- **`df_objetivo_imputado_2.xlsx`**: Second version of imputed dataset
- **`df_escalado.xlsx`**: ⭐ **Final processed dataset** - scaled and ready for modeling

## Data Processing Pipeline

```
Raw Data → Target Creation → Imputation → Encoding → Scaling → Final Dataset
```

1. **Original Data**: `raw/2024-2.xlsx`
2. **Target Variable**: Addition of `RIESGO_DESERCION` binary variable
3. **Missing Data Handling**: Systematic imputation strategy
4. **Feature Engineering**: Categorical encoding and numerical scaling
5. **Final Dataset**: `processed/df_objetivo/df_escalado.xlsx`

## Key Variables

### Target Variable
- **`RIESGO_DESERCION`**: Binary variable (0/1) indicating dropout risk
  - Created using 80th percentile threshold of academic delay
  - 0 = Low risk, 1 = High risk

### Features Include
- Academic performance indicators
- Enrollment history
- Socioeconomic factors
- Administrative variables

## Data Privacy

- All datasets contain anonymized student information
- No personally identifiable information (PII) is included
- Academic records have been de-identified

## Usage Notes

- Use `df_escalado.xlsx` for model training
- All datasets are in Excel format for compatibility
- CSV versions available where specified
- Maintain data integrity by not modifying raw files
