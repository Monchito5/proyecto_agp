"""
main.py
Integrador principal del Pipeline de Predicción de Sitios de Splicing.
Este programa orquesta las fases de preprocesamiento, análisis exploratorio (EDA),
optimización de hiperparámetros mediante evolución diferencial y entrenamiento final.
"""

import sys
import argparse
import gffutils
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Importaciones de módulos locales refactorizados
from data_preprocessing import (
    obtener_sitios_reales, 
    obtener_sitios_señuelo, 
    generar_señuelos_simulados, 
    extraer_y_limpiar_secuencias
)
from eda import realizar_analisis_general, realizar_analisis_especifico
from optimizer import (
    OptimizadorHiperparametros, 
    evolucion_diferencial, 
    ConfiguracionEvolutiva
)

# --- Constantes Simbólicas ---
RAIZ = Path(__file__).parent.parent
RAW_DIR = RAIZ / "data" / "raw"
EXP_DIR = RAIZ / "data" / "export"

DB_PATH = RAW_DIR / "gencode.v47.annotation.gtf.db"
FASTA_PATH = RAW_DIR / "GRCh38.primary_assembly.genome.fa"
DATASET_BAL = EXP_DIR / "dataset_consolidado_balanceado.csv"
TRAIN_PATH = EXP_DIR / "dataset_entrenamiento.csv"
TEST_PATH = EXP_DIR / "dataset_prueba.csv"

class GestorPipeline:
    """Administra el flujo de trabajo completo del proyecto."""

    def __init__(self):
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        self.conexion_db = None
        self.tabla_datos = None

    def conectar_db(self):
        """Conecta a la base de datos de GFFUtils."""
        if not DB_PATH.exists():
            print(f"Error: DB no hallada en {DB_PATH}. Ejecute download_data.py.")
            sys.exit(1)
        self.conexion_db = gffutils.FeatureDB(str(DB_PATH))

    def preprocesar(self, limite: int = 0):
        """Etapa de extracción y limpieza masiva."""
        print("\n=== ETAPA 1: PREPROCESAMIENTO ===")
        self.conectar_db()
        reales = obtener_sitios_reales(self.conexion_db, limite_genes=(None if limite == 0 else limite))
        reales_limpios = extraer_y_limpiar_secuencias(reales, FASTA_PATH)
        
        objetivo = len(reales_limpios)
        print(f"Buscando {objetivo} señuelos...")
        negativos = obtener_sitios_señuelo(FASTA_PATH, objetivo, reales_limpios)
        
        if negativos is not None:
            neg_limpios = extraer_y_limpiar_secuencias(negativos, FASTA_PATH)
            if len(neg_limpios) < objetivo:
                simulados = generar_señuelos_simulados(objetivo - len(neg_limpios))
                neg_limpios = pd.concat([neg_limpios, simulados])
        else:
            neg_limpios = generar_señuelos_simulados(objetivo)

        self.tabla_datos = pd.concat([reales_limpios, neg_limpios], ignore_index=True)
        self.tabla_datos = self.tabla_datos.sample(frac=1, random_state=42).reset_index(drop=True)
        self.tabla_datos.to_csv(DATASET_BAL, index=False)
        print(f"✓ Dataset balanceado generado: {len(self.tabla_datos)} filas.")

    def optimizar(self):
        """Búsqueda de hiperparámetros mediante Evolución Diferencial."""
        print("\n=== ETAPA: OPTIMIZACIÓN EVOLUTIVA ===")
        if not TRAIN_PATH.exists():
            self.particionar()
            
        opt = OptimizadorHiperparametros(TRAIN_PATH, TEST_PATH)
        limites = np.array([
            [-4.0, -2.0], # Log LR
            [0.1, 0.6],   # Dropout
            [32, 128],    # Filtros 1
            [64, 256],    # Filtros 2
            [128, 512],   # Filtros 3
            [2, 10],      # Kernel 1 (base)
            [1, 7],       # Kernel 2 (base)
            [1, 5]        # Kernel 3 (base)
        ])
        
        config = ConfiguracionEvolutiva(tamano_poblacion=8, total_generaciones=10)
        resultado = evolucion_diferencial(opt.funcion_aptitud, limites, config)
        
        print("\n=== MEJOR CONFIGURACIÓN ENCONTRADA ===")
        print(f"Mejor Error (Val): {resultado.mejor_aptitud:.4f}")
        print(f"Vector Óptimo: {resultado.mejor_vector}")
        
        # Guardar resultados
        np.save(EXP_DIR / "hiperparametros_optimos.npy", resultado.mejor_vector)

    def particionar(self):
        """Divide el dataset en entrenamiento y prueba."""
        print("\n=== ETAPA 3: PARTICIÓN ESTRATIFICADA ===")
        if self.tabla_datos is None:
            self.tabla_datos = pd.read_csv(DATASET_BAL)
        
        entreno, prueba = train_test_split(
            self.tabla_datos, test_size=0.2, 
            stratify=self.tabla_datos['label'], random_state=42
        )
        entreno.to_csv(TRAIN_PATH, index=False)
        prueba.to_csv(TEST_PATH, index=False)
        print(f"✓ Partición: Entreno({len(entreno)}), Prueba({len(prueba)})")

    def ejecutar_eda(self):
        """Genera reportes visuales."""
        if self.tabla_datos is None:
            self.tabla_datos = pd.read_csv(DATASET_BAL)
        self.conectar_db()
        realizar_analisis_general(self.tabla_datos, EXP_DIR)
        realizar_analisis_especifico(self.tabla_datos, self.conexion_db, EXP_DIR)

def principal():
    parser = argparse.ArgumentParser(description="Pipeline Splicing CNN")
    parser.add_argument("--paso", choices=['todo', 'pre', 'eda', 'opt', 'part'], default='todo')
    parser.add_argument("--limite", type=int, default=50) # Reducido para prueba rápida
    
    args = parser.parse_args()
    gestor = GestorPipeline()

    if args.paso in ['todo', 'pre']: gestor.preprocesar(args.limite)
    if args.paso in ['todo', 'part']: gestor.particionar()
    if args.paso in ['todo', 'eda']: gestor.ejecutar_eda()
    if args.paso == 'opt': gestor.optimizar()

if __name__ == "__main__":
    principal()
