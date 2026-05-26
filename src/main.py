"""
main.py
Integrador principal del Pipeline de Predicción de Sitios de Splicing.
Este programa orquesta las fases de preprocesamiento, análisis exploratorio (EDA),
optimización de hiperparámetros, interpretación de modelos y generación de reportes.
"""

import sys
import argparse
import gffutils
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuración de ruta para importaciones locales
sys.path.insert(0, str(Path(__file__).parent.absolute()))

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
from interpretabilidad import generar_reporte_interpretabilidad
from generar_reporte import GeneradorDocumento

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
        """Conecta a la base de datos en data/export/."""
        if not ARCHIVO_BASE_DATOS.exists():
            print(f"Error crítico: DB no hallada en {ARCHIVO_BASE_DATOS}.")
            sys.exit(1)
        self.conexion_db = gffutils.FeatureDB(str(ARCHIVO_BASE_DATOS))

    def preprocesar(self, limite: int = 0):
        """Etapa de extracción y limpieza masiva para Donantes y Aceptores."""
        print("\n=== ETAPA 1: PREPROCESAMIENTO (Donantes + Aceptores) ===")
        self.conectar_db()
        
        limite_genes = None
        if limite == 1000: limite_genes = 1000
        elif limite == 5000: limite_genes = 5000
        elif limite == 0: limite_genes = None
        else: limite_genes = limite

        # 1. Extracción de todos los sitios reales
        reales = obtener_sitios_reales(self.conexion_db, limite_genes=limite_genes)
        reales_limpios = extraer_y_limpiar_secuencias(reales, ARCHIVO_GENOMA_FASTA)
        
        # Separar por tipo para balanceo individual
        df_reales_don = reales_limpios[reales_limpios['tipo_sitio'] == 'donante']
        df_reales_acep = reales_limpios[reales_limpios['tipo_sitio'] == 'aceptor']
        
        # 2. Generación de Señuelos para cada tipo
        final_negativos = []
        for tipo, df_tipo in [('donante', df_reales_don), ('aceptor', df_reales_acep)]:
            objetivo = len(df_tipo)
            print(f"Buscando {objetivo} señuelos para {tipo}...")
            negativos = obtener_sitios_señuelo(ARCHIVO_GENOMA_FASTA, objetivo, df_tipo, tipo=tipo)
            
            if negativos is not None:
                neg_limpios = extraer_y_limpiar_secuencias(negativos, ARCHIVO_GENOMA_FASTA)
                if len(neg_limpios) < objetivo:
                    faltantes = objetivo - len(neg_limpios)
                    simulados = generar_señuelos_simulados(faltantes, tipo=tipo)
                    neg_limpios = pd.concat([neg_limpios, simulados])
                final_negativos.append(neg_limpios)
            else:
                final_negativos.append(generar_señuelos_simulados(objetivo, tipo=tipo))

        df_negativos_total = pd.concat(final_negativos, ignore_index=True)
        
        # 3. Consolidación Final
        self.tabla_datos = pd.concat([reales_limpios, df_negativos_total], ignore_index=True)
        self.tabla_datos = self.tabla_datos.sample(frac=1, random_state=SEMILLA_ALEATORIA).reset_index(drop=True)
        self.tabla_datos.to_csv(ARCHIVO_DATASET_CONSOLIDADO, index=False)
        
        counts = self.tabla_datos.groupby(['tipo_sitio', 'label']).size().to_dict()
        print(f"✓ Dataset consolidado: {len(self.tabla_datos)} muestras.")
        print(f"Distribución: {counts}")
        self.tabla_datos = None 

    def ejecutar_eda(self):
        """Genera reportes visuales en data/figures/."""
        print("\n=== ETAPA 2: ANÁLISIS EXPLORATORIO DE DATOS ===")
        if not ARCHIVO_DATASET_CONSOLIDADO.exists():
            print("Error: Dataset no encontrado.")
            return
        
        self.tabla_datos = pd.read_csv(ARCHIVO_DATASET_CONSOLIDADO)
        self.conectar_db()
        
        for figura in RUTA_FIGURAS.glob("*.png"):
            try: figura.unlink()
            except Exception: pass

        # Asegurar firmas correctas (3 argumentos)
        realizar_analisis_general(self.tabla_datos, self.conexion_db, RUTA_FIGURAS)
        realizar_analisis_especifico(self.tabla_datos, self.conexion_db, RUTA_FIGURAS)
        print(f"✓ Visualizaciones actualizadas.")

    def particionar(self):
        """Divide el dataset en entrenamiento y prueba."""
        print("\n=== ETAPA 3: PARTICIÓN ESTRATIFICADA ===")
        if not ARCHIVO_DATASET_CONSOLIDADO.exists():
            print("Error: Dataset no encontrado.")
            return
        self.tabla_datos = pd.read_csv(ARCHIVO_DATASET_CONSOLIDADO)
        
        entreno, prueba = train_test_split(
            self.tabla_datos, test_size=TAMANO_TEST_ESTANDAR, 
            stratify=self.tabla_datos['label'], random_state=SEMILLA_ALEATORIA
        )
        entreno.to_csv(ARCHIVO_TRAIN, index=False)
        prueba.to_csv(ARCHIVO_TEST, index=False)
        print(f"✓ Partición completada.")

    def optimizar(self):
        """Búsqueda de hiperparámetros mediante Evolución Diferencial."""
        print("\n=== ETAPA 4: OPTIMIZACIÓN EVOLUTIVA ===")
        if not ARCHIVO_TRAIN.exists():
            self.particionar()
            
        opt = OptimizadorHiperparametros(ARCHIVO_TRAIN, ARCHIVO_TEST)
        limites = np.array([
            [-4.0, -2.0], [0.1, 0.6], [32, 128], [64, 256], [128, 512], [2, 10], [1, 7], [1, 5]
        ])
        
        config = ConfiguracionEvolutiva(tamano_poblacion=8, total_generaciones=10)
        resultado = evolucion_diferencial(opt.funcion_aptitud, limites, config)
        print(f"\n✓ Optimización completada.")
        np.save(RUTA_PROCESADOS / "hiperparametros_optimos.npy", resultado.mejor_vector)

    def ejecutar_interpretacion(self):
        """Genera reportes de interpretabilidad del modelo."""
        print("\n=== ETAPA 5: ANÁLISIS DE INTERPRETABILIDAD ===")
        if not ARCHIVO_TEST.exists():
            print("Error: Dataset de prueba no hallado.")
            return
            
        from model import RedNeuronalSplicing
        ruta_hparams = RUTA_PROCESADOS / "hiperparametros_optimos.npy"
        if ruta_hparams.exists():
            v_opt = np.load(ruta_hparams)
            filtros = [int(v_opt[2]), int(v_opt[3]), int(v_opt[4])]
            kernels = [2*int(v_opt[5])+1, 2*int(v_opt[6])+1, 2*int(v_opt[7])+1]
            modelo = RedNeuronalSplicing(lista_filtros=filtros, lista_kernels=kernels)
        else:
            modelo = RedNeuronalSplicing()

        ruta_pesos = RUTA_EXPORT / "best_model.pth"
        if not ruta_pesos.exists():
            print("Error: Modelo no hallado.")
            return
            
        # Carga robusta para PyTorch 2.6+
        try:
            checkpoint = torch.load(str(ruta_pesos), map_location='cpu', weights_only=False)
        except Exception:
            checkpoint = torch.load(str(ruta_pesos), map_location='cpu')
            
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            modelo.load_state_dict(checkpoint['model_state_dict'])
        else:
            modelo.load_state_dict(checkpoint)
            
        generar_reporte_interpretabilidad(modelo, ARCHIVO_TEST, RUTA_FIGURAS / "interpretability")

    def ejecutar_entrenamiento(self):
        """Etapa de entrenamiento final con hiperparámetros óptimos."""
        print("\n=== ETAPA: ENTRENAMIENTO FINAL ===")
        if not ARCHIVO_TRAIN.exists(): self.particionar()
            
        from train import inicializar_cargadores_datos, ejecutar_entrenamiento_robusto
        from model import RedNeuronalSplicing
        
        ruta_hparams = RUTA_PROCESADOS / "hiperparametros_optimos.npy"
        if ruta_hparams.exists():
            v_opt = np.load(ruta_hparams)
            tasa_lr = 10**v_opt[0]
            abandono = v_opt[1]
            filtros = [int(v_opt[2]), int(v_opt[3]), int(v_opt[4])]
            kernels = [2*int(v_opt[5])+1, 2*int(v_opt[6])+1, 2*int(v_opt[7])+1]
            modelo = RedNeuronalSplicing(filtros, kernels, abandono, abandono)
        else:
            modelo = RedNeuronalSplicing()
            tasa_lr = 0.001
            
        c_train, c_val = inicializar_cargadores_datos(ARCHIVO_TRAIN, ARCHIVO_TEST)
        ejecutar_entrenamiento_robusto(modelo, c_train, c_val, lr=tasa_lr)

    def generar_reporte(self):
        """Crea el documento DOCX final."""
        print("\n=== ETAPA 6: GENERACIÓN DE REPORTE FINAL ===")
        try:
            GeneradorDocumento().guardar_reporte()
        except Exception as e:
            print(f"Error al generar reporte: {e}")

def mostrar_menu():
    print("\n" + "="*55 + "\n      PIPELINE DE PREDICCIÓN DE SPLICING\n" + "="*55)
    print("1. Ejecutar TODO el Pipeline")
    print("2. Preprocesamiento | 3. EDA | 4. Partición")
    print("5. Optimización | 6. Entrenamiento | 7. Interpretación")
    print("8. Reporte DOCX | 9. Salir")
    return input("Seleccione una opción: ")

def principal():
    gestor = GestorPipeline()
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--paso", choices=['todo', 'pre', 'eda', 'part', 'opt', 'ent', 'int', 'rep'])
        parser.add_argument("--limite", type=int, default=1000)
        args = parser.parse_args()
        if args.paso == 'todo':
            gestor.preprocesar(args.limite); gestor.particionar(); 
            gestor.ejecutar_eda(); gestor.optimizar(); gestor.ejecutar_entrenamiento();
            gestor.ejecutar_interpretacion(); gestor.generar_reporte()
        elif args.paso == 'pre': gestor.preprocesar(args.limite)
        elif args.paso == 'eda': gestor.ejecutar_eda()
        elif args.paso == 'part': gestor.particionar()
        elif args.paso == 'opt': gestor.optimizar()
        elif args.paso == 'ent': gestor.ejecutar_entrenamiento()
        elif args.paso == 'int': gestor.ejecutar_interpretacion()
        elif args.paso == 'rep': gestor.generar_reporte()
        return

    while True:
        opcion = mostrar_menu()
        if opcion == '1':
            lim = int(input("Límite: ") or 1000)
            gestor.preprocesar(lim); gestor.particionar(); gestor.ejecutar_eda();
            gestor.optimizar(); gestor.ejecutar_entrenamiento();
            gestor.ejecutar_interpretacion(); gestor.generar_reporte()
        elif opcion == '2': gestor.preprocesar(int(input("Límite: ") or 1000))
        elif opcion == '3': gestor.ejecutar_eda()
        elif opcion == '4': gestor.particionar()
        elif opcion == '5': gestor.optimizar()
        elif opcion == '6': gestor.ejecutar_entrenamiento()
        elif opcion == '7': gestor.ejecutar_interpretacion()
        elif opcion == '8': gestor.generar_reporte()
        elif opcion == '9': break

if __name__ == "__main__":
    principal()
