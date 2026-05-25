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
ETIQUETA_FRECUENCIA = "Cantidad Total"

def realizar_analisis_general(tabla_datos: pd.DataFrame, base_datos: gffutils.FeatureDB, ruta_salida: Path):
    """
    Genera visualizaciones del estado general de la anotación y el dataset.
    Prioriza el uso de GFFUtils para estadísticas biológicas globales.
    """
    print("\n--- Generando Reporte Visual General (Prioridad GFFUtils) ---")
    
    # 1. Resumen de Anotación Genómica (GFFUtils)
    print("Extrayendo estadísticas globales de la anotación...")
    tipos_biologicos = list(base_datos.featuretypes())
    conteos_globales = {}
    for tipo in tipos_biologicos:
        query = f'SELECT COUNT(*) FROM features WHERE featuretype = "{tipo}"'
        conteos_globales[tipo] = base_datos.execute(query).fetchone()[0]
    
    ser_conteos = pd.Series(conteos_globales).sort_values(ascending=False)
    
    plt.figure()
    ser_conteos.head(10).plot(kind='barh', color='teal')
    plt.title("Gen 01: Top 10 Tipos de Elementos en la Anotación (GFFUtils)")
    plt.xlabel(ETIQUETA_FRECUENCIA)
    plt.ylabel("Tipo de Feature")
    plt.tight_layout()
    plt.savefig(ruta_salida / "gen_01_composicion_features.png")
    plt.close()

    # 2. Análisis de Atributos por Columna (Dataset Procesado)
    print("Analizando estructura técnica del dataset...")
    unique_dtypes = [str(t) for t in tabla_datos.dtypes.unique()]
    matriz_tipos = pd.DataFrame(0, index=tabla_datos.columns, columns=unique_dtypes)
    for col in tabla_datos.columns:
        matriz_tipos.loc[col, str(tabla_datos[col].dtype)] = 1
        
    plt.figure(figsize=(10, 4))
    sns.heatmap(matriz_tipos, annot=True, cbar=False, cmap="YlGnBu")
    plt.title("Gen 02: Tipos de Datos Técnicos en el Dataset Procesado")
    plt.tight_layout()
    plt.savefig(ruta_salida / "gen_02_tipos_datos_heatmap.png")
    plt.close()

    # 3. Mapa de Calidad y Completitud
    plt.figure()
    sns.heatmap(tabla_datos.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title(f"Gen 03: Mapa de Completitud (Total Muestras: {len(tabla_datos)})")
    plt.savefig(ruta_salida / "gen_03_completitud_heatmap.png")
    plt.close()

def realizar_analisis_especifico(tabla_datos: pd.DataFrame, base_datos: gffutils.FeatureDB, ruta_salida: Path):
    """
    Genera visualizaciones biológicas específicas para sitios de splicing (Puntos 1-6).
    """
    print("--- Generando Reporte Visual Específico (Genómica) ---")
    
    # 1. Balance de clases
    plt.figure(figsize=(7, 7))
    conteo_clases = tabla_datos['label'].value_counts()
    clases_ordenadas = ["Sitios Reales (GT)", "Sitios Señuelo (Falso GT)"]
    plt.pie(conteo_clases, labels=clases_ordenadas, autopct='%1.1f%%', colors=['#66b3ff','#99ff99'])
    plt.title(f"Spec 01: Balance de Clases (N={len(tabla_datos)})")
    plt.savefig(ruta_salida / "spec_01_balance_clases.png")
    plt.close()

    # 2. Distribución de cromosomas (GFFUtils como referencia global)
    print("Calculando distribución cromosómica global...")
    chrom_counts = {}
    for feature in base_datos.features_of_type('gene'):
        chrom_counts[feature.chrom] = chrom_counts.get(feature.chrom, 0) + 1
    
    ser_chroms = pd.Series(chrom_counts).sort_values(ascending=False).head(10)
    plt.figure()
    ser_chroms.plot(kind='barh', color='plum')
    plt.title("Spec 02: Distribución Global de Genes por Cromosoma (GFFUtils)")
    plt.xlabel("Cantidad de Genes")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(ruta_salida / "spec_02_dist_cromosomas.png")
    plt.close()

    # 3. Composición nucleotídica global
    all_seqs = "".join(tabla_datos['sequence'].iloc[:min(10000, len(tabla_datos))].tolist())
    freqs = {nt: all_seqs.count(nt) for nt in 'ACGT'}
    plt.figure(figsize=(7, 7))
    plt.pie(freqs.values(), labels=freqs.keys(), autopct='%1.1f%%', colors=sns.color_palette('viridis', 4))
    plt.title("Spec 03: Composición de Bases en Secuencias Procesadas")
    plt.savefig(ruta_salida / "spec_03_composicion_global.png")
    plt.close()

    # 4. Distribución de Contenido GC
    def calc_gc(s): return (s.count('G') + s.count('C')) / len(s) if s else 0
    tabla_datos['contenido_gc'] = tabla_datos['sequence'].apply(calc_gc)
    plt.figure()
    df_plot = tabla_datos.copy()
    df_plot['Origen'] = df_plot['label'].map({1: "Genómico Real", 0: "Señuelo Sintético"})
    sns.violinplot(data=df_plot, x='Origen', y='contenido_gc', palette='Set2')
    plt.title("Spec 04: Variabilidad de Contenido GC por Naturaleza del Sitio")
    plt.ylabel("Fracción Guanina-Citosina")
    plt.savefig(ruta_salida / "spec_04_gc_content_violin.png")
    plt.close()

    # 5. Correlación de Spearman
    df_num = tabla_datos.select_dtypes(include=[np.number]).copy()
    nombres_map = {'label': 'Naturaleza', 'pos': 'Coordenada', 'contenido_gc': 'GC %'}
    df_num = df_num.rename(columns=nombres_map)
    if df_num.shape[1] > 1:
        plt.figure()
        sns.heatmap(df_num.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Spec 05: Interacciones entre Variables Numéricas")
        plt.tight_layout()
        plt.savefig(ruta_salida / "spec_05_correlacion_heatmap.png")
        plt.close()

    # 6. Perfil de Longitud de Exones (Referencia GFFUtils)
    print("Extrayendo longitudes de exones para perfil genómico...")
    exon_lens = [len(f) for i, f in enumerate(base_datos.features_of_type('exon')) if i < 10000]
    plt.figure()
    sns.histplot(exon_lens, kde=True, color='orange', bins=100)
    plt.xlim(0, 1000)
    plt.title(f"Spec 06: Perfil de Longitudes de Exones (Muestra N=10,000)")
    plt.xlabel("Longitud (pb)")
    plt.ylabel("Densidad de Frecuencia")
    plt.tight_layout()
    plt.savefig(ruta_salida / "spec_06_longitud_exones.png")
    plt.close()
