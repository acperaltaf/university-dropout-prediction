import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def crear_shap_plot(rf_model, X_train, X_test, plot_type="bar", max_samples=100, clase=1):
    """
    Crear UN SOLO plot de SHAP para tu caso específico
    
    Parameters:
    - rf_model: tu modelo Random Forest entrenado
    - X_train: datos de entrenamiento (DataFrame)
    - X_test: datos de prueba (DataFrame) 
    - plot_type: "bar" para importancia, "beeswarm" para distribución
    - max_samples: número máximo de muestras a analizar
    - clase: 0 para clase negativa, 1 para clase positiva (default=1)
    """
    
    print(f"=== CREANDO PLOT SHAP PARA CLASE {clase} ===")
    
    # Limitar muestras si es necesario
    if len(X_test) > max_samples:
        X_test_sample = X_test.iloc[:max_samples].copy()
        print(f"Usando {max_samples} muestras de {len(X_test)} disponibles")
    else:
        X_test_sample = X_test.copy()
        print(f"Usando todas las {len(X_test)} muestras")
    
    # Crear explainer y calcular valores SHAP
    print("Calculando valores SHAP...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_sample)
    
    print(f"Forma original de shap_values: {shap_values.shape}")
    
    # Tu caso específico: (samples, features, classes)
    # Necesitamos extraer los valores para la clase específica
    if len(shap_values.shape) == 3:
        shap_values_clase = shap_values[:, :, clase]
        print(f"Extrayendo clase {clase}, nueva forma: {shap_values_clase.shape}")
    else:
        # Por si acaso el formato cambia
        shap_values_clase = shap_values
        print(f"Formato directo, forma: {shap_values_clase.shape}")
    
    # Obtener nombres de características
    feature_names = X_test_sample.columns.tolist()
    print(f"Características: {len(feature_names)} detectadas")
    
    # Verificar dimensiones finales
    print(f"Verificación final: SHAP {shap_values_clase.shape} vs X_test {X_test_sample.shape}")
    
    if shap_values_clase.shape[1] != X_test_sample.shape[1]:
        raise ValueError(f"Error de dimensiones: SHAP tiene {shap_values_clase.shape[1]} características, X_test tiene {X_test_sample.shape[1]}")
    
    # Crear el plot
    plt.figure(figsize=(10, 8))
    
    if plot_type == "bar":
        print("Creando summary plot tipo barra (importancia promedio)...")
        shap.summary_plot(shap_values_clase, X_test_sample, 
                         feature_names=feature_names, 
                         plot_type="bar",
                         show=False)
        plt.title(f'Importancia SHAP - Clase {clase} {"(Positiva)" if clase == 1 else "(Negativa)"}')
        
    elif plot_type == "beeswarm":
        print("Creando summary plot tipo beeswarm (distribución de valores)...")
        shap.summary_plot(shap_values_clase, X_test_sample, 
                         feature_names=feature_names,
                         show=False)
        # plt.title(f'Distribución Valores SHAP - Clase {clase} {"(Positiva)" if clase == 1 else "(Negativa)"}')
        plt.title(f' ')
    
    elif plot_type == "violin":
        print("Creando summary plot tipo violin...")
        shap.summary_plot(shap_values_clase, X_test_sample, 
                         feature_names=feature_names,
                         plot_type="violin",
                         show=False)
        plt.title(f'Distribución Violin SHAP - Clase {clase} {"(Positiva)" if clase == 1 else "(Negativa)"}')
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Plot creado exitosamente para clase {clase}")
    
    return shap_values_clase

def crear_plots_comparativos(rf_model, X_train, X_test, max_samples=50, top_features=10):
    """
    Crear plots comparativos para ambas clases (lado a lado)
    """
    
    print("=== CREANDO PLOTS COMPARATIVOS ===")
    
    # Limitar muestras
    X_test_sample = X_test.iloc[:max_samples] if len(X_test) > max_samples else X_test
    
    # Calcular SHAP values
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_sample)
    
    # Extraer valores para ambas clases
    shap_clase_0 = shap_values[:, :, 0]  # Clase negativa
    shap_clase_1 = shap_values[:, :, 1]  # Clase positiva
    
    feature_names = X_test_sample.columns.tolist()
    
    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot para clase 0 (negativa)
    plt.sca(ax1)
    shap.summary_plot(shap_clase_0, X_test_sample, 
                     feature_names=feature_names, 
                     plot_type="bar",
                     max_display=top_features,
                     show=False)
    ax1.set_title('Importancia SHAP - Clase 0 (Negativa)')
    
    # Plot para clase 1 (positiva)
    plt.sca(ax2)
    shap.summary_plot(shap_clase_1, X_test_sample, 
                     feature_names=feature_names, 
                     plot_type="bar",
                     max_display=top_features,
                     show=False)
    ax2.set_title('Importancia SHAP - Clase 1 (Positiva)')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Plots comparativos creados")
    
    return shap_clase_0, shap_clase_1

def mostrar_top_features(rf_model, X_train, X_test, max_samples=100, top_n=10):
    """
    Mostrar las características más importantes según SHAP
    """
    
    print(f"=== TOP {top_n} CARACTERÍSTICAS MÁS IMPORTANTES ===")
    
    X_test_sample = X_test.iloc[:max_samples] if len(X_test) > max_samples else X_test
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_sample)
    
    # Para clase positiva (1)
    shap_clase_1 = shap_values[:, :, 1]
    
    # Calcular importancia promedio (valor absoluto)
    importancia_promedio = np.abs(shap_clase_1).mean(axis=0)
    
    # Crear DataFrame con resultados
    feature_names = X_test_sample.columns.tolist()
    df_importancia = pd.DataFrame({
        'Característica': feature_names,
        'Importancia_SHAP': importancia_promedio
    }).sort_values('Importancia_SHAP', ascending=False)
    
    print(f"\nTop {top_n} características para la clase POSITIVA:")
    print("-" * 60)
    for i, (idx, row) in enumerate(df_importancia.head(top_n).iterrows()):
        print(f"{i+1:2d}. {row['Característica']:35s} | {row['Importancia_SHAP']:.4f}")
    
    return df_importancia

# =================================================================
# CÓDIGO LISTO PARA USAR:
# =================================================================

# print("🎯 CÓDIGO OPTIMIZADO PARA TUS DATOS")
# print("=" * 60)
# print()
# print("1️⃣  CREAR PLOT PRINCIPAL (clase positiva - la que te interesa):")
# print("    shap_vals = crear_shap_plot(rf_model, X_train, X_test, plot_type='bar')")
# print()
# print("2️⃣  PARA VER DISTRIBUCIÓN DE VALORES:")
# print("    crear_shap_plot(rf_model, X_train, X_test, plot_type='beeswarm')")
# print()
# print("3️⃣  PARA COMPARAR AMBAS CLASES:")
# print("    crear_plots_comparativos(rf_model, X_train, X_test)")
# print()
# print("4️⃣  PARA VER RANKING DE CARACTERÍSTICAS:")
# print("    ranking = mostrar_top_features(rf_model, X_train, X_test)")
# print()
# print("💡 Si tienes muchos datos, usa max_samples para acelerar:")
# print("    crear_shap_plot(rf_model, X_train, X_test, max_samples=50)")