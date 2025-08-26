"""
Utilidades para análisis exploratorio de datos (EDA)
Funciones para análisis automático de estructura de datasets
"""

import pandas as pd
import numpy as np
from collections import Counter


def analizar_estructura_dataset(df, nombre_dataset="Dataset", mostrar_jerarquia=True, 
                               target_candidates=['PAPA', 'PROME_ACADE', 'AVANCE_CARRERA', 
                                                'NUMERO_MATRICULAS', 'PUNTAJE_ADMISION']):
    """
    Realiza un análisis completo de la estructura de un dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset a analizar
    nombre_dataset : str
        Nombre descriptivo del dataset para mostrar en el reporte
    mostrar_jerarquia : bool
        Si mostrar el análisis de estructura jerárquica
    target_candidates : list
        Lista de variables candidatas a ser target
        
    Returns:
    --------
    dict : Diccionario con los resultados del análisis
    """
    
    print("=" * 80)
    print(f"📊 ANÁLISIS DE ESTRUCTURA DEL {nombre_dataset.upper()}")
    print("=" * 80)

    # 1. DIMENSIONALIDAD
    print("\n🔢 1. DIMENSIONALIDAD")
    print("-" * 30)
    print(f"Número de filas (registros): {df.shape[0]:,}")
    print(f"Número de columnas (variables): {df.shape[1]:,}")
    print(f"Total de celdas: {df.shape[0] * df.shape[1]:,}")

    # 2. INFORMACIÓN GENERAL
    print(f"\nTamaño en memoria: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    # 3. ANÁLISIS DE TIPOS DE VARIABLES
    print("\n📝 2. ANÁLISIS DE TIPOS DE VARIABLES")
    print("-" * 40)

    # Identificar tipos de variables automáticamente
    numeric_vars = []
    categorical_vars = []
    binary_vars = []
    date_vars = []
    id_vars = []

    for col in df.columns:
        # Verificar si es fecha
        if 'FECHA' in col.upper():
            date_vars.append(col)
        # Verificar si es ID o código
        elif any(keyword in col.upper() for keyword in ['COD_', 'CODIGO', 'DOCUMENTO']):
            id_vars.append(col)
        else:
            # Analizar el contenido de la columna
            non_null_values = df[col].dropna()
            unique_values = non_null_values.unique()
            
            # Variables binarias (solo 2 valores únicos)
            if len(unique_values) == 2:
                binary_vars.append(col)
            # Variables numéricas
            elif df[col].dtype in ['int64', 'float64']:
                numeric_vars.append(col)
            # Variables categóricas
            else:
                categorical_vars.append(col)

    print(f"🔢 Variables Numéricas ({len(numeric_vars)}):")
    for var in numeric_vars:
        print(f"   • {var}")

    print(f"\n📂 Variables Categóricas ({len(categorical_vars)}):")
    for var in categorical_vars:
        unique_count = df[var].nunique()
        print(f"   • {var} ({unique_count} categorías únicas)")

    print(f"\n⚡ Variables Binarias ({len(binary_vars)}):")
    for var in binary_vars:
        values = df[var].dropna().unique()
        print(f"   • {var} (valores: {', '.join(map(str, values))})")

    print(f"\n📅 Variables de Fecha ({len(date_vars)}):")
    for var in date_vars:
        print(f"   • {var}")

    print(f"\n🔑 Variables Identificadoras/Códigos ({len(id_vars)}):")
    for var in id_vars:
        unique_count = df[var].nunique()
        print(f"   • {var} ({unique_count} valores únicos)")

    # 4. ANÁLISIS DE CLAVES ÚNICAS E IDENTIFICADORES
    print("\n🔑 3. ANÁLISIS DE CLAVES ÚNICAS E IDENTIFICADORES")
    print("-" * 50)

    # Buscar posibles claves primarias
    primary_key_candidates = []
    for col in df.columns:
        if df[col].nunique() == len(df) and not df[col].isnull().any():
            primary_key_candidates.append(col)

    print("🎯 Candidatos a Clave Primaria:")
    if primary_key_candidates:
        for candidate in primary_key_candidates:
            print(f"   • {candidate}")
    else:
        print("   • No se encontraron claves primarias únicas")

    # Análisis de duplicados
    duplicated_rows = df.duplicated().sum()
    print(f"\n📋 Análisis de Duplicados:")
    print(f"   • Filas duplicadas completas: {duplicated_rows}")

    # Buscar posibles claves foráneas (códigos que se repiten)
    print("\n🔗 Posibles Claves Foráneas (códigos con múltiples referencias):")
    foreign_key_candidates = []
    for col in id_vars:
        unique_count = df[col].nunique()
        total_count = len(df[col].dropna())
        if unique_count < total_count and unique_count > 1:
            foreign_key_candidates.append((col, unique_count, total_count))

    for fk, unique, total in sorted(foreign_key_candidates, key=lambda x: x[1]):
        repetition_rate = (total - unique) / total * 100
        print(f"   • {fk}: {unique} valores únicos, {total} registros ({repetition_rate:.1f}% repetición)")

    # 5. ANÁLISIS DE VALORES FALTANTES
    print("\n❓ 4. ANÁLISIS DE VALORES FALTANTES")
    print("-" * 40)

    missing_analysis = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        if missing_count > 0:
            missing_analysis.append((col, missing_count, missing_pct))

    if missing_analysis:
        missing_analysis.sort(key=lambda x: x[2], reverse=True)
        print("Variables con valores faltantes:")
        for col, count, pct in missing_analysis:
            print(f"   • {col}: {count} valores faltantes ({pct:.1f}%)")
    else:
        print("✅ No se encontraron valores faltantes")

    # 6. ESTADÍSTICAS DESCRIPTIVAS BÁSICAS
    print("\n📈 5. ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES NUMÉRICAS")
    print("-" * 55)

    if numeric_vars:
        numeric_stats = df[numeric_vars].describe()
        print(numeric_stats.round(2))
    else:
        print("No hay variables numéricas para analizar")

    # 7. RESUMEN DE CARDINALIDAD
    print("\n🎯 6. RESUMEN DE CARDINALIDAD (Top variables con más categorías)")
    print("-" * 65)

    cardinality_analysis = []
    for col in df.columns:
        unique_count = df[col].nunique()
        cardinality_analysis.append((col, unique_count))

    cardinality_analysis.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 variables por número de valores únicos:")
    for i, (col, unique_count) in enumerate(cardinality_analysis[:10], 1):
        percentage = (unique_count / len(df)) * 100
        print(f"{i:2d}. {col}: {unique_count} valores únicos ({percentage:.1f}% del total)")

    # 8. ANÁLISIS ESPECÍFICO PARA MACHINE LEARNING
    print("\n🤖 7. CONSIDERACIONES PARA MACHINE LEARNING")
    print("-" * 45)

    print("📊 Variables Target Potenciales:")
    available_targets = [var for var in target_candidates if var in df.columns]

    for var in available_targets:
        if df[var].dtype in ['int64', 'float64']:
            non_null_count = df[var].count()
            min_val = df[var].min()
            max_val = df[var].max()
            print(f"   • {var}: {non_null_count} valores válidos, rango [{min_val} - {max_val}]")

    print(f"\n🔄 Variables que requieren preprocesamiento:")
    preprocessing_needed = []

    # Variables categóricas con alta cardinalidad
    high_cardinality = [(col, df[col].nunique()) for col in categorical_vars if df[col].nunique() > 20]
    if high_cardinality:
        print("   📂 Categóricas con alta cardinalidad (>20 categorías):")
        for col, count in high_cardinality:
            print(f"      • {col}: {count} categorías")
            preprocessing_needed.append(f"{col} (encoding)")

    # Variables de fecha
    if date_vars:
        print("   📅 Variables de fecha (requieren feature engineering):")
        for var in date_vars:
            print(f"      • {var}")
            preprocessing_needed.append(f"{var} (datetime features)")

    # Variables con valores faltantes significativos
    high_missing = [(col, pct) for col, count, pct in missing_analysis if pct > 10]
    if high_missing:
        print("   ❓ Variables con >10% valores faltantes:")
        for col, pct in high_missing:
            print(f"      • {col}: {pct:.1f}% faltantes")
            preprocessing_needed.append(f"{col} (imputation)")

    print(f"\n⚙️ Recomendaciones de preprocesamiento:")
    if preprocessing_needed:
        for i, rec in enumerate(preprocessing_needed, 1):
            print(f"   {i}. {rec}")
    else:
        print("   ✅ Datos parecen estar en buen estado para ML")

    # 9. ESTRUCTURA DE DATOS JERÁRQUICA (opcional)
    if mostrar_jerarquia:
        print("\n🏗️ 8. ESTRUCTURA JERÁRQUICA IDENTIFICADA")
        print("-" * 45)

        hierarchy_levels = [
            ('SEDE', 'Nivel Universidad'),
            ('COD_FACULTAD', 'FACULTAD', 'Nivel Facultad'),
            ('COD_PLAN', 'PLAN', 'Nivel Programa'),
            ('DOCUMENTO', 'Nivel Estudiante')
        ]

        print("Jerarquía de datos identificada:")
        for level in hierarchy_levels:
            if len(level) == 3:
                code_col, name_col, description = level
                if code_col in df.columns and name_col in df.columns:
                    unique_codes = df[code_col].nunique()
                    unique_names = df[name_col].nunique()
                    print(f"   • {description}: {unique_codes} códigos, {unique_names} nombres")
            else:
                col, description = level
                if col in df.columns:
                    unique_count = df[col].nunique()
                    print(f"   • {description}: {unique_count} registros únicos")

    print("\n" + "=" * 80)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 80)

    # Mostrar un resumen final
    print(f"\n📋 RESUMEN EJECUTIVO:")
    print(f"   • Dataset con {df.shape[0]:,} registros y {df.shape[1]} variables")
    print(f"   • {len(numeric_vars)} variables numéricas, {len(categorical_vars)} categóricas")
    print(f"   • {len(binary_vars)} variables binarias, {len(date_vars)} de fecha")
    print(f"   • {len(id_vars)} identificadores/códigos")
    print(f"   • {len([x for x in missing_analysis if x[2] > 0])} variables con datos faltantes")
    print(f"   • Listo para aplicar ML con preprocesamiento adecuado")
    
    # Retornar resultados estructurados
    resultados = {
        'dimensiones': df.shape,
        'tipos_variables': {
            'numericas': numeric_vars,
            'categoricas': categorical_vars,
            'binarias': binary_vars,
            'fechas': date_vars,
            'identificadores': id_vars
        },
        'claves_primarias': primary_key_candidates,
        'duplicados': duplicated_rows,
        'valores_faltantes': missing_analysis,
        'cardinalidad': cardinality_analysis,
        'targets_disponibles': available_targets,
        'preprocesamiento_requerido': preprocessing_needed
    }
    
    return resultados


def analisis_rapido_dataset(df, nombre_dataset="Dataset"):
    """
    Versión simplificada del análisis para obtener solo información básica
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset a analizar
    nombre_dataset : str
        Nombre descriptivo del dataset
        
    Returns:
    --------
    dict : Resumen básico del dataset
    """
    print(f"📊 RESUMEN RÁPIDO: {nombre_dataset}")
    print("-" * 40)
    
    # Información básica
    print(f"Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Tipos de datos
    tipos = df.dtypes.value_counts()
    print(f"Tipos de datos: {dict(tipos)}")
    
    # Valores faltantes
    missing = df.isnull().sum().sum()
    missing_pct = (missing / (df.shape[0] * df.shape[1])) * 100
    print(f"Valores faltantes: {missing} ({missing_pct:.2f}%)")
    
    # Duplicados
    duplicados = df.duplicated().sum()
    print(f"Filas duplicadas: {duplicados}")
    
    return {
        'shape': df.shape,
        'memory_kb': df.memory_usage(deep=True).sum() / 1024,
        'dtypes': dict(tipos),
        'missing_values': missing,
        'missing_percent': missing_pct,
        'duplicated_rows': duplicados
    }


def comparar_datasets(df_antes, df_despues, nombre_antes="Dataset Original", nombre_despues="Dataset Procesado"):
    """
    Compara dos datasets y muestra las diferencias principales
    
    Parameters:
    -----------
    df_antes : pandas.DataFrame
        Dataset antes del procesamiento
    df_despues : pandas.DataFrame
        Dataset después del procesamiento
    nombre_antes : str
        Nombre del primer dataset
    nombre_despues : str
        Nombre del segundo dataset
    """
    print("🔄 COMPARACIÓN DE DATASETS")
    print("=" * 50)
    
    # Comparar dimensiones
    print(f"\n📏 DIMENSIONES:")
    print(f"   {nombre_antes}: {df_antes.shape[0]:,} × {df_antes.shape[1]}")
    print(f"   {nombre_despues}: {df_despues.shape[0]:,} × {df_despues.shape[1]}")
    
    filas_diff = df_despues.shape[0] - df_antes.shape[0]
    cols_diff = df_despues.shape[1] - df_antes.shape[1]
    print(f"   Diferencia: {filas_diff:+} filas, {cols_diff:+} columnas")
    
    # Comparar valores faltantes
    missing_antes = df_antes.isnull().sum().sum()
    missing_despues = df_despues.isnull().sum().sum()
    print(f"\n❓ VALORES FALTANTES:")
    print(f"   {nombre_antes}: {missing_antes}")
    print(f"   {nombre_despues}: {missing_despues}")
    print(f"   Reducción: {missing_antes - missing_despues} valores faltantes")
    
    # Comparar columnas
    cols_antes = set(df_antes.columns)
    cols_despues = set(df_despues.columns)
    cols_nuevas = cols_despues - cols_antes
    cols_eliminadas = cols_antes - cols_despues
    
    print(f"\n📂 COLUMNAS:")
    if cols_nuevas:
        print(f"   Nuevas columnas ({len(cols_nuevas)}): {', '.join(list(cols_nuevas)[:5])}{'...' if len(cols_nuevas) > 5 else ''}")
    if cols_eliminadas:
        print(f"   Columnas eliminadas ({len(cols_eliminadas)}): {', '.join(list(cols_eliminadas)[:5])}{'...' if len(cols_eliminadas) > 5 else ''}")
    
    print(f"   Columnas comunes: {len(cols_antes & cols_despues)}")
