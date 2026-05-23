"""
eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import gffutils

# Configuración visual global para gráficos de producción
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Constantes de etiquetas para consistencia
ETIQUETA_CLASE = "Clase de Sitio (1: Real, 0: Señuelo)"
ETIQUETA_CROMOSOMA = "Nombre del Cromosoma"
ETIQUETA_FRECUENCIA = "Frecuencia / Cantidad"

def realizar_analisis_general(tabla_datos: pd.DataFrame, ruta_salida: Path):
    """
    Genera visualizaciones del estado general del dataset procesado.
    """
    print("\n--- Generando Reporte Visual General ---")
    
    # 1. Composición de tipos de columnas
    columnas_numericas = tabla_datos.select_dtypes(include=['number']).columns.tolist()
    columnas_objetos = tabla_datos.select_dtypes(include=['object']).columns.tolist()
    columnas_secuencia = [c for c in columnas_objetos if 'sequence' in c.lower()]
    
    conteo_tipos = pd.Series({
        'Variables Numéricas': len(columnas_numericas),
        'Variables Categóricas': len(columnas_objetos) - len(columnas_secuencia),
        'Secuencias Biológicas': len(columnas_secuencia)
    })
    
    plt.figure()
    conteo_tipos.plot(kind='barh', color='skyblue')
    plt.title("Estructura de Dimensiones del Dataset")
    plt.xlabel(ETIQUETA_FRECUENCIA)
    plt.tight_layout()
    plt.savefig(ruta_salida / "gen_01_composicion_features.png")
    plt.close()

    # 2. Mapa de calor de tipos (Análisis de tipos de datos)
    unique_dtypes = [str(t) for t in tabla_datos.dtypes.unique()]
    matriz_tipos = pd.DataFrame(0, index=tabla_datos.columns, columns=unique_dtypes)
    for columna in tabla_datos.columns:
        matriz_tipos.loc[columna, str(tabla_datos[columna].dtype)] = 1
        
    plt.figure(figsize=(10, 4))
    sns.heatmap(matriz_tipos, annot=True, cbar=False, cmap="YlGnBu")
    plt.title("Tipos de Datos por Columna")
    plt.tight_layout()
    plt.savefig(ruta_salida / "gen_02_tipos_datos_heatmap.png")
    plt.close()

    # 3. Completitud y Calidad (Mapa de Calor de Nulos)
    plt.figure()
    sns.heatmap(tabla_datos.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title("Análisis de Completitud: Ausencia de Valores Nulos")
    plt.savefig(ruta_salida / "gen_03_completitud_heatmap.png")
    plt.close()

def realizar_analisis_especifico(tabla_datos: pd.DataFrame, base_datos: gffutils.FeatureDB, ruta_salida: Path):
    """
    Genera visualizaciones biológicas y estadísticas específicas para sitios de splicing (Puntos 1-6).
    """
    print("--- Generando Reporte Visual Específico (Genómica) ---")
    
    # 1. Balance de clases (Real vs Señuelo)
    plt.figure(figsize=(7, 7))
    conteo_clases = tabla_datos['label'].value_counts()
    etiquetas_bonitas = ["Sitios Reales", "Sitios Señuelo"]
    plt.pie(conteo_clases, labels=etiquetas_bonitas, autopct='%1.1f%%', colors=['#66b3ff','#99ff99'])
    plt.title("Punto 1: Balance de Clases en el Dataset")
    plt.savefig(ruta_salida / "spec_01_balance_clases.png")
    plt.close()

    # 2. Distribución de cromosomas (Top 10)
    plt.figure()
    tabla_datos['chrom'].value_counts().head(10).plot(kind='barh', color='plum')
    plt.title("Punto 2: Top 10 Cromosomas con mayor densidad de sitios")
    plt.xlabel(ETIQUETA_FRECUENCIA)
    plt.ylabel(ETIQUETA_CROMOSOMA)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(ruta_salida / "spec_02_dist_cromosomas.png")
    plt.close()

    # 3. Composición nucleotídica global
    all_seqs = "".join(tabla_datos['sequence'].iloc[:min(10000, len(tabla_datos))].tolist())
    freqs = {nt: all_seqs.count(nt) for nt in 'ACGT'}
    plt.figure(figsize=(7, 7))
    plt.pie(freqs.values(), labels=freqs.keys(), autopct='%1.1f%%', colors=sns.color_palette('viridis', 4))
    plt.title("Punto 3: Composición Nucleotídica Global")
    plt.savefig(ruta_salida / "spec_03_composicion_global.png")
    plt.close()

    # 4. Distribución de Contenido GC (Violin Plot)
    def calcular_gc(cadena: str) -> float:
        return (cadena.count('G') + cadena.count('C')) / len(cadena) if cadena else 0
        
    tabla_datos['contenido_gc'] = tabla_datos['sequence'].apply(calcular_gc)
    plt.figure()
    df_plot = tabla_datos.copy()
    df_plot['Tipo de Sitio'] = df_plot['label'].map({1: "Real", 0: "Señuelo"})
    sns.violinplot(data=df_plot, x='Tipo de Sitio', y='contenido_gc', palette='Set2')
    plt.title("Punto 4: Comparativa de Contenido GC por Naturaleza del Sitio")
    plt.ylabel("Fracción de Guanina-Citosina")
    plt.savefig(ruta_salida / "spec_04_gc_content_violin.png")
    plt.close()

    # 5. Correlación (Heatmap Spearman)
    print("Analizando correlaciones entre variables numéricas...")
    columnas_numericas = tabla_datos.select_dtypes(include=[np.number])
    if columnas_numericas.shape[1] > 1:
        plt.figure()
        sns.heatmap(columnas_numericas.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Punto 5: Mapa de Calor de Correlación (Spearman)")
        plt.savefig(ruta_salida / "spec_05_correlacion_heatmap.png")
        plt.close()

    # 6. Longitud de exones (Referencia Genómica)
    print("Extrayendo longitudes de exones para contexto biológico...")
    longitudes_exones = [len(f) for i, f in enumerate(base_datos.features_of_type('exon')) if i < 2000]
    plt.figure()
    sns.histplot(longitudes_exones, kde=True, color='orange')
    plt.xlim(0, 1000)
    plt.title("Punto 6: Perfil de Longitud de Exones (Anotación GENCODE)")
    plt.xlabel("Longitud en Pares de Bases (pb)")
    plt.ylabel(ETIQUETA_FRECUENCIA)
    plt.tight_layout()
    plt.savefig(ruta_salida / "spec_06_longitud_exones.png")
    plt.close()
