"""
train.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Dict

# Importaciones locales estandarizadas
from dataset import DatasetSplicing
from model import RedNeuronalSplicing

# --- Rutas de Producción ---
RAIZ = Path(__file__).parent.parent
DIR_PROCESADOS = RAIZ / "data" / "processed"
DIR_FIGURAS = RAIZ / "data" / "figures"
DIR_MODELOS = RAIZ / "data" / "export"

# --- Constantes del Proceso de Entrenamiento ---
TASA_APRENDIZAJE_ESTANDAR = 0.001
TAMANO_LOTE_DEFAULT = 32
TOTAL_EPOCAS_LIMITE = 30
DISPOSITIVO_PROCESO = 'cuda' if torch.cuda.is_available() else 'cpu'

def inicializar_cargadores_datos(ruta_train: Path, ruta_test: Path) -> Tuple[DataLoader, DataLoader]:
    """
    Instancia los DataLoaders de PyTorch para entrenamiento y validación.
    """
    conjunto_entreno = DatasetSplicing(ruta_train)
    conjunto_validacion = DatasetSplicing(ruta_test)

    cargador_entreno = DataLoader(conjunto_entreno, batch_size=TAMANO_LOTE_DEFAULT, shuffle=True)
    cargador_val = DataLoader(conjunto_validacion, batch_size=TAMANO_LOTE_DEFAULT, shuffle=False)

    return cargador_entreno, cargador_val

def ejecutar_entrenamiento(cargador_entreno: DataLoader, cargador_val: DataLoader):
    """
    Ciclo principal de entrenamiento y evaluación.
    """
    print(f"\nIniciando entrenamiento en dispositivo: {DISPOSITIVO_PROCESO}")
    modelo_splicing = RedNeuronalSplicing().to(DISPOSITIVO_PROCESO)
    optimizador = torch.optim.Adam(modelo_splicing.parameters(), lr=TASA_APRENDIZAJE_ESTANDAR)
    criterio_error = nn.BCELoss()

    mejores_metricas = {"error": float('inf'), "epoca": 0}
    historial_error = []

    for epoca in range(1, TOTAL_EPOCAS_LIMITE + 1):
        modelo_splicing.train()
        error_total_lote = 0.0
        
        for tensores_secuencia, etiquetas_clase in cargador_entreno:
            tensores_secuencia = tensores_secuencia.to(DISPOSITIVO_PROCESO)
            etiquetas_clase = etiquetas_clase.to(DISPOSITIVO_PROCESO)

            optimizador.zero_grad()
            predicciones = modelo_splicing(tensores_secuencia)
            error_calculado = criterio_error(predicciones, etiquetas_clase)
            
            error_calculado.backward()
            optimizador.step()
            error_total_lote += error_calculado.item()

        error_promedio = error_total_lote / len(cargador_entreno)
        historial_error.append(error_promedio)
        print(f"Epoca [{epoca}/{TOTAL_EPOCAS_LIMITE}] - Error Entrenamiento: {error_promedio:.4f}")

        if error_promedio < mejores_metricas["error"]:
            mejores_metricas["error"] = error_promedio
            torch.save(modelo_splicing.state_dict(), DIR_MODELOS / "best_model.pth")

    # Graficar curva de pérdida final en la carpeta de figuras
    plt.figure()
    plt.plot(historial_error, label='Pérdida Entrenamiento')
    plt.title("Evolución de la Función de Pérdida")
    plt.xlabel("Época")
    plt.ylabel("BCELoss")
    plt.savefig(DIR_FIGURAS / "curva_entrenamiento_loss.png")
    plt.close()

    print(f"\n✓ Entrenamiento completado. Gráfica guardada en {DIR_FIGURAS.name}.")

if __name__ == "__main__":
    RUTA_TRAIN = DIR_PROCESADOS / "dataset_entrenamiento.csv"
    RUTA_TEST = DIR_PROCESADOS / "dataset_prueba.csv"
    
    if RUTA_TRAIN.exists() and RUTA_TEST.exists():
        c_train, c_val = inicializar_cargadores_datos(RUTA_TRAIN, RUTA_TEST)
        ejecutar_entrenamiento(c_train, c_val)
    else:
        print("Error: No se hallaron los datasets en data/processed. Ejecute main.py.")
