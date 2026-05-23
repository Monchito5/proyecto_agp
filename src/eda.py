"""
eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gffutils

# Configuración visual global
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def realizar_analisis_general(tabla_datos: pd.DataFrame, ruta_salida: Path):
    """
    Genera visualizaciones del estado general del dataset.
    """
    print("\n--- Analizando Estado General del Dataset ---")
    
    # 1. Composición de tipos de columnas
    columnas_numericas = tabla_datos.select_dtypes(include=['number']).columns.tolist()
    columnas_objetos = tabla_datos.select_dtypes(include=['object']).columns.tolist()
    columnas_secuencia = [c for c in columnas_objetos if 'sequence' in c.lower()]
    
    conteo_tipos = pd.Series({
        'Numéricas': len(columnas_numericas),
        'Categóricas': len(columnas_objetos) - len(columnas_secuencia),
        'Secuencias': len(columnas_secuencia)
    })
    
    plt.figure()
    conteo_tipos.plot(kind='barh', color='skyblue')
    plt.title("Composición de Columnas por Tipo")
    plt.savefig(ruta_salida / "gen_01_composicion_features.png")
    plt.close()

    # 2. Mapa de calor de tipos
    matriz_tipos = pd.DataFrame(0, index=tabla_datos.columns, columns=tabla_datos.dtypes.unique().astype(str))
    for columna in tabla_datos.columns:
        matriz_tipos.loc[columna, str(tabla_datos[columna].dtype)] = 1
        
    plt.figure(figsize=(10, 4))
    sns.heatmap(matriz_tipos, annot=True, cbar=False, cmap="YlGnBu")
    plt.title("Mapa de Calor: Tipos de Datos por Columna")
    plt.savefig(ruta_salida / "gen_02_tipos_datos_heatmap.png")
    plt.close()

    # 3. Completitud de datos
    plt.figure()
    sns.heatmap(tabla_datos.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title("Mapa de Calor: Completitud de los Datos")
    plt.savefig(ruta_salida / "gen_03_completitud_heatmap.png")
    plt.close()

def realizar_analisis_especifico(tabla_datos: pd.DataFrame, base_datos: gffutils.FeatureDB, ruta_salida: Path):
    """
    Genera visualizaciones biológicas y estadísticas específicas.
    """
    print("--- Analizando Propiedades Genómicas ---")
    
    # 1. Balance de clases
    plt.figure(figsize=(7, 7))
    conteo_clases = tabla_datos['label'].value_counts()
    etiquetas = [f"Clase {c} (N={n})" for c, n in conteo_clases.items()]
    plt.pie(conteo_clases, labels=etiquetas, autopct='%1.1f%%', colors=['#66b3ff','#99ff99'])
    plt.title("Balance de Clases: Verdaderos (1) vs Señuelos (0)")
    plt.savefig(ruta_salida / "spec_01_balance_clases.png")
    plt.close()

    # 2. Distribución de cromosomas
    plt.figure()
    tabla_datos['chrom'].value_counts().head(10).plot(kind='barh', color='plum')
    plt.title("Top 10 Cromosomas con mayor densidad de sitios")
    plt.gca().invert_yaxis()
    plt.savefig(ruta_salida / "spec_02_dist_cromosomas.png")
    plt.close()

    # 3. Contenido GC por clase
    def calcular_contenido_gc(cadena_adn: str) -> float:
        if not cadena_adn: return 0.0
        return (cadena_adn.count('G') + cadena_adn.count('C')) / len(cadena_adn)
        
    tabla_datos['contenido_gc'] = tabla_datos['sequence'].apply(calcular_contenido_gc)
    
    plt.figure()
    sns.violinplot(data=tabla_datos, x='label', y='contenido_gc', palette='Set2')
    plt.title("Distribución de Contenido GC por Clase")
    plt.ylabel("Fracción GC")
    plt.savefig(ruta_salida / "spec_04_gc_content_violin.png")
    plt.close()

    # 4. Longitud de exones (Base de Datos)
    print("Extrayendo estadísticas de longitud desde la DB...")
    longitudes_exones = [len(f) for i, f in enumerate(base_datos.features_of_type('exon')) if i < 2000]
    
    plt.figure()
    sns.histplot(longitudes_exones, kde=True, color='orange')
    plt.xlim(0, 1000)
    plt.title("Distribución de Longitudes de Exones en el Genoma")
    plt.xlabel("Longitud (pb)")
    plt.savefig(ruta_salida / "spec_06_longitud_exones.png")
    plt.close()
