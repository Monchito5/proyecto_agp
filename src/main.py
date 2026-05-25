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
from interpretabilidad import interpretar_modelo

# --- Constantes Simbólicas de Rutas ---
RAIZ = Path(__file__).parent.parent
RUTA_RAW = RAIZ / "data" / "raw"
RUTA_PROCESADOS = RAIZ / "data" / "processed"
RUTA_EXPORT = RAIZ / "data" / "export"
RUTA_FIGURAS = RAIZ / "data" / "figures"

ARCHIVO_BASE_DATOS = RUTA_EXPORT / "gencode.v47.annotation.gtf.db"
ARCHIVO_GENOMA_FASTA = RUTA_RAW / "GRCh38.primary_assembly.genome.fa"

ARCHIVO_DATASET_CONSOLIDADO = RUTA_PROCESADOS / "dataset_consolidado_balanceado.csv"
ARCHIVO_TRAIN = RUTA_PROCESADOS / "dataset_entrenamiento.csv"
ARCHIVO_TEST = RUTA_PROCESADOS / "dataset_prueba.csv"

# --- Configuración Estándar ---
TAMANO_TEST_ESTANDAR = 0.2
SEMILLA_ALEATORIA = 42

class GestorPipeline:
    """Administra el flujo de trabajo completo del proyecto."""

    def __init__(self):
        """Asegura la existencia de la estructura de carpetas."""
        for carpeta in [RUTA_PROCESADOS, RUTA_EXPORT, RUTA_FIGURAS]:
            carpeta.mkdir(parents=True, exist_ok=True)
        self.conexion_db = None
        self.tabla_datos = None

    def conectar_db(self):
        """Conecta a la base de datos de GFFUtils en data/export/."""
        if not ARCHIVO_BASE_DATOS.exists():
            print(f"Error crítico: DB no hallada en {ARCHIVO_BASE_DATOS}.")
            print("Asegúrese de ejecutar src/download_data.py.")
            sys.exit(1)
        self.conexion_db = gffutils.FeatureDB(str(ARCHIVO_BASE_DATOS))

    def preprocesar(self, limite: int = 0, hard_negative: bool = False):
        """Etapa de extracción y limpieza masiva.
        
        Parámetros:
            limite (int): Límite de genes a procesar (0 para todos).
            hard_negative (bool): Activa Hard-Negative Mining para señuelos.
        """
        print("\n=== ETAPA 1: PREPROCESAMIENTO ===")
        self.conectar_db()
        reales = obtener_sitios_reales(self.conexion_db, limite_genes=(None if limite == 0 else limite))
        reales_limpios = extraer_y_limpiar_secuencias(reales, ARCHIVO_GENOMA_FASTA)
        
        objetivo = len(reales_limpios)
        print(f"Buscando {objetivo} señuelos (Hard-Negative={'Sí' if hard_negative else 'No'})...")
        negativos = obtener_sitios_señuelo(ARCHIVO_GENOMA_FASTA, objetivo, reales_limpios, modo_dificil=hard_negative)
        
        if negativos is not None:
            neg_limpios = extraer_y_limpiar_secuencias(negativos, ARCHIVO_GENOMA_FASTA)
            if len(neg_limpios) < objetivo:
                simulados = generar_señuelos_simulados(objetivo - len(neg_limpios))
                neg_limpios = pd.concat([neg_limpios, simulados])
        else:
            neg_limpios = generar_señuelos_simulados(objetivo)

        self.tabla_datos = pd.concat([reales_limpios, neg_limpios], ignore_index=True)
        self.tabla_datos = self.tabla_datos.sample(frac=1, random_state=42).reset_index(drop=True)
        self.tabla_datos.to_csv(ARCHIVO_DATASET_CONSOLIDADO, index=False)
        print(f"✓ Dataset consolidado generado en: {RUTA_PROCESADOS.name}")

    def optimizar(self):
        """Búsqueda de hiperparámetros mediante Evolución Diferencial."""
        print("\n=== ETAPA: OPTIMIZACIÓN EVOLUTIVA ===")
        if not ARCHIVO_TRAIN.exists():
            self.particionar()
            
        opt = OptimizadorHiperparametros(ARCHIVO_TRAIN, ARCHIVO_TEST)
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
        
        # Guardar resultados en procesados
        np.save(RUTA_PROCESADOS / "hiperparametros_optimos.npy", resultado.mejor_vector)

    def particionar(self):
        """Divide el dataset en entrenamiento y prueba."""
        print("\n=== ETAPA 3: PARTICIÓN ESTRATIFICADA ===")
        if self.tabla_datos is None:
            self.tabla_datos = pd.read_csv(ARCHIVO_DATASET_CONSOLIDADO)
        
        entreno, prueba = train_test_split(
            self.tabla_datos, test_size=TAMANO_TEST_ESTANDAR, 
            stratify=self.tabla_datos['label'], random_state=SEMILLA_ALEATORIA
        )
        entreno.to_csv(ARCHIVO_TRAIN, index=False)
        prueba.to_csv(ARCHIVO_TEST, index=False)
        print(f"✓ Partición: Entrenamiento({len(entreno)}), Prueba({len(prueba)})")

    def ejecutar_eda(self):
        """Genera reportes visuales en data/figures/."""
        print("\n=== ETAPA 2: ANÁLISIS EXPLORATORIO DE DATOS ===")
        if self.tabla_datos is None:
            if not ARCHIVO_DATASET_CONSOLIDADO.exists():
                print("Error: Dataset no encontrado.")
                return
            self.tabla_datos = pd.read_csv(ARCHIVO_DATASET_CONSOLIDADO)
        
        self.conectar_db()
        realizar_analisis_general(self.tabla_datos, RUTA_FIGURAS)
        realizar_analisis_especifico(self.tabla_datos, self.conexion_db, RUTA_FIGURAS)
        print(f"✓ Visualizaciones generadas en: {RUTA_FIGURAS.name}")

    def interpretar(self):
        """Ejecuta el pipeline de interpretabilidad sobre el modelo entrenado."""
        import torch
        from model import RedNeuronalSplicing
        
        print("\n=== ETAPA: INTERPRETABILIDAD DEL MODELO ===")
        ruta_modelo = RUTA_EXPORT / "best_model.pth"
        if not ruta_modelo.exists():
            print("Error: Modelo no encontrado. Ejecute el entrenamiento primero.")
            return
        
        if not ARCHIVO_TEST.exists():
            print("Error: Dataset de prueba no encontrado. Ejecute '../main.py --paso todo' primero.")
            return
            
        modelo = RedNeuronalSplicing()
        modelo.load_state_dict(torch.load(ruta_modelo, map_location='cpu'))
        
        # Llamar al módulo de interpretabilidad
        interpretar_modelo(
            modelo=modelo,
            ruta_test_csv=ARCHIVO_TEST,
            ruta_salida_figuras=RUTA_FIGURAS / "interpretabilidad"
        )

def principal():
    """Función de entrada al programa principal."""
    parser = argparse.ArgumentParser(description="Pipeline Splicing CNN Production")
    parser.add_argument("--paso", choices=['todo', 'pre', 'eda', 'opt', 'part', 'int'], default='todo')
    parser.add_argument("--limite", type=int, default=0)
    parser.add_argument("--hard-negative", action="store_true", help="Activa Hard-Negative Mining para señuelos de alta dificultad.")
    
    args = parser.parse_args()
    gestor = GestorPipeline()

    if args.paso in ['todo', 'pre']: gestor.preprocesar(args.limite, args.hard_negative)
    if args.paso in ['todo', 'part']: gestor.particionar()
    if args.paso in ['todo', 'eda']: gestor.ejecutar_eda()
    if args.paso == 'opt': gestor.optimizar()
    if args.paso == 'int': gestor.interpretar()

if __name__ == "__main__":
    principal()
