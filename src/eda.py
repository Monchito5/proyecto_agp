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

# Constantes de etiquetas para consistencia profesional
ETIQUETA_CANTIDAD = "Cantidad de Registros"
ETIQUETA_NATURALEZA = "Naturaleza del Sitio"
ETIQUETA_TIPO = "Tipo de Sitio (Motivo)"

def realizar_analisis_general(tabla_datos: pd.DataFrame, base_datos: gffutils.FeatureDB, ruta_salida: Path):
    """
    Genera visualizaciones del estado general de la anotación y el dataset.
    """
    print("\n--- Generando Reporte Visual General ---")
    
    # 1. Resumen de Anotación Genómica (GFFUtils)
    print("Extrayendo estadísticas globales de la anotación...")
    tipos_biologicos = list(base_datos.featuretypes())
    conteos_globales = {}
    for tipo in tipos_biologicos:
        query = f'SELECT COUNT(*) FROM features WHERE featuretype = "{tipo}"'
        conteos_globales[tipo] = base_datos.execute(query).fetchone()[0]
    
    ser_conteos = pd.Series(conteos_globales).sort_values(ascending=False)
    
    plt.figure()
    ser_conteos.head(10).plot(kind='barh', color='#1f77b4')
    plt.title("Gen 01: Composición Global de la Anotación GENCODE", fontsize=12, fontweight='bold')
    plt.xlabel(ETIQUETA_CANTIDAD)
    plt.ylabel("Tipo de Elemento Genómico")
    plt.tight_layout()
    plt.savefig(ruta_salida / "gen_01_resumen_anotacion.png", dpi=300)
    plt.close()

    # 2. Análisis de Tipos de Datos (Dataset Procesado)
    print("Analizando estructura técnica del dataset...")
    unique_dtypes = [str(t) for t in tabla_datos.dtypes.unique()]
    matriz_tipos = pd.DataFrame(0, index=tabla_datos.columns, columns=unique_dtypes)
    for col in tabla_datos.columns:
        matriz_tipos.loc[col, str(tabla_datos[col].dtype)] = 1
        
    plt.figure(figsize=(10, 4))
    sns.heatmap(matriz_tipos, annot=True, cbar=False, cmap="YlGnBu")
    plt.title("Gen 02: Mapeo de Tipos de Datos del Dataset", fontsize=12, fontweight='bold')
    plt.xlabel("Tipo de Dato")
    plt.ylabel("Nombre de Columna")
    plt.tight_layout()
    plt.savefig(ruta_salida / "gen_02_analisis_tipos_datos.png", dpi=300)
    plt.close()

    # 3. Mapa de Calidad y Completitud
    plt.figure()
    sns.heatmap(tabla_datos.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title(f"Gen 03: Validación de Completitud del Dataset (N={len(tabla_datos)})", fontsize=12, fontweight='bold')
    plt.xlabel("Columnas Analizadas")
    plt.ylabel("Registros (Sin Nulos)")
    plt.tight_layout()
    plt.savefig(ruta_salida / "gen_03_completitud_dataset.png", dpi=300)
    plt.close()

def realizar_analisis_especifico(tabla_datos: pd.DataFrame, base_datos: gffutils.FeatureDB, ruta_salida: Path):
    """
    Genera visualizaciones biológicas específicas para sitios de splicing (Puntos 1-6).
    """
    print("--- Generando Reporte Visual Específico (Genómica) ---")
    
    # Mapeo de etiquetas para visualización
    df_visual = tabla_datos.copy()
    df_visual[ETIQUETA_NATURALEZA] = df_visual['label'].map({1: "Sitio Real", 0: "Sitio Señuelo"})
    df_visual[ETIQUETA_TIPO] = df_visual['tipo_sitio'].map({'donante': "Donante (GT)", 'aceptor': "Aceptor (AG)"})

    # 1. Balance de clases y tipos
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    conteo_clases = df_visual[ETIQUETA_NATURALEZA].value_counts()
    plt.pie(conteo_clases, labels=conteo_clases.index, autopct='%1.1f%%', colors=['#66b3ff','#99ff99'])
    plt.title("Spec 01a: Balance Global de Clases")

    plt.subplot(1, 2, 2)
    sns.countplot(data=df_visual, x=ETIQUETA_TIPO, hue=ETIQUETA_NATURALEZA, palette='Set2')
    plt.title("Spec 01b: Distribución por Tipo de Sitio")
    plt.ylabel(ETIQUETA_CANTIDAD)
    
    plt.tight_layout()
    plt.savefig(ruta_salida / "spec_01_balance_clases_splicing.png", dpi=300)
    plt.close()

    # 2. Distribución de cromosomas
    print("Calculando distribución cromosómica global...")
    chrom_counts = {}
    for feature in base_datos.features_of_type('gene'):
        chrom_counts[feature.chrom] = chrom_counts.get(feature.chrom, 0) + 1
    
    ser_chroms = pd.Series(chrom_counts).sort_values(ascending=False).head(10)
    plt.figure()
    ser_chroms.plot(kind='barh', color='plum')
    plt.title("Spec 02: Top 10 Cromosomas con Mayor Densidad Génica", fontsize=12, fontweight='bold')
    plt.xlabel("Cantidad de Genes Anotados")
    plt.ylabel("Identificador del Cromosoma")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(ruta_salida / "spec_02_distribucion_genes.png", dpi=300)
    plt.close()

    # 3. Composición nucleotídica global
    all_seqs = "".join(tabla_datos['sequence'].iloc[:min(10000, len(tabla_datos))].tolist())
    freqs = {nt: all_seqs.count(nt) for nt in 'ACGT'}
    plt.figure(figsize=(7, 7))
    plt.pie(freqs.values(), labels=freqs.keys(), autopct='%1.1f%%', colors=sns.color_palette('viridis', 4))
    plt.title("Spec 03: Composición de Bases Nucleotídicas Totales", fontsize=12, fontweight='bold')
    plt.savefig(ruta_salida / "spec_03_frecuencia_bases_adn.png", dpi=300)
    plt.close()

    # 4. Distribución de Contenido GC (FIGURAS SEPARADAS)
    def calc_gc(s): return (s.count('G') + s.count('C')) / len(s) if s else 0
    df_visual['Fracción GC'] = df_visual['sequence'].apply(calc_gc)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Lado Izquierdo: Donantes
    df_don = df_visual[df_visual['tipo_sitio'] == 'donante']
    sns.violinplot(data=df_don, x=ETIQUETA_NATURALEZA, y='Fracción GC', ax=ax1, palette='pastel')
    ax1.set_title("Sitios Donantes (GT)")
    ax1.set_xlabel(ETIQUETA_NATURALEZA)
    
    # Lado Derecho: Aceptores
    df_acep = df_visual[df_visual['tipo_sitio'] == 'aceptor']
    sns.violinplot(data=df_acep, x=ETIQUETA_NATURALEZA, y='Fracción GC', ax=ax2, palette='muted')
    ax2.set_title("Sitios Aceptores (AG)")
    ax2.set_xlabel(ETIQUETA_NATURALEZA)
    
    plt.suptitle("Spec 04: Comparativa de Contenido GC por Tipo y Naturaleza", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(ruta_salida / "spec_04_contenido_gc_por_tipo.png", dpi=300)
    plt.close()

    # 5. Correlación de Spearman
    print("Analizando correlaciones...")
    df_num = tabla_datos[['pos', 'label']].copy()
    df_num['Contenido GC %'] = df_visual['Fracción GC']
    df_num = df_num.rename(columns={'pos': 'Coordenada Genómica', 'label': 'Naturaleza (Real=1)'})
    
    if df_num.shape[1] > 1:
        plt.figure()
        sns.heatmap(df_num.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Spec 05: Matriz de Correlación de Spearman", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(ruta_salida / "spec_05_matriz_correlacion.png", dpi=300)
        plt.close()

    # 6. Perfil de Longitud de Exones
    print("Extrayendo longitudes de exones...")
    exon_lens = [len(f) for i, f in enumerate(base_datos.features_of_type('exon')) if i < 10000]
    plt.figure()
    sns.histplot(exon_lens, kde=True, color='orange', bins=100)
    plt.xlim(0, 1000)
    plt.title("Spec 06: Perfil de Longitudes de Exones (GENCODE)", fontsize=12, fontweight='bold')
    plt.xlabel("Longitud de la Secuencia (pb)")
    plt.ylabel("Densidad de Frecuencia")
    plt.tight_layout()
    plt.savefig(ruta_salida / "spec_06_perfil_longitud_exones.png", dpi=300)
    plt.close()
