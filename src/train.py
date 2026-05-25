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
from typing import Tuple, Optional

# Importaciones locales estandarizadas
from dataset import DatasetSplicing
from model import RedNeuronalSplicing

# --- Rutas de Producción ---
RAIZ_PROYECTO = Path(__file__).parent.parent
DIR_PROCESADOS = RAIZ_PROYECTO / "data" / "processed"
DIR_FIGURAS = RAIZ_PROYECTO / "data" / "figures"
DIR_MODELOS = RAIZ_PROYECTO / "data" / "export"

def inicializar_cargadores_datos(ruta_entrenamiento: Path, ruta_prueba: Path) -> Tuple[DataLoader, DataLoader]:
    """
    Instancia los DataLoaders de PyTorch para entrenamiento y validación.
    """
    conjunto_entreno = DatasetSplicing(ruta_entrenamiento)
    conjunto_validacion = DatasetSplicing(ruta_prueba)

    cargador_entreno = DataLoader(conjunto_entreno, batch_size=32, shuffle=True)
    cargador_val = DataLoader(conjunto_validacion, batch_size=32, shuffle=False)

    return cargador_entreno, cargador_val

def ejecutar_entrenamiento_robusto(
    modelo_red: nn.Module, 
    loader_entreno: DataLoader, 
    loader_val: DataLoader, 
    lr: float = 0.001,
    epocas: int = 10
):
    """
    Ciclo de entrenamiento completo con soporte para arquitecturas dinámicas.
    """
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo_red.to(dispositivo)
    optimizador = torch.optim.Adam(modelo_red.parameters(), lr=lr)
    criterio = nn.BCELoss()

    mejor_error = float('inf')
    historial_error = []

    print(f"Entrenando en: {dispositivo} | Épocas: {epocas}")

    for epoca in range(1, epocas + 1):
        modelo_red.train()
        error_acumulado = 0.0
        
        for tensores_x, etiquetas_y in loader_entreno:
            tensores_x, etiquetas_y = tensores_x.to(dispositivo), etiquetas_y.to(dispositivo)
            optimizador.zero_grad()
            salida = modelo_red(tensores_x)
            error = criterio(salida, etiquetas_y)
            error.backward()
            optimizador.step()
            error_acumulado += error.item()

        promedio_error = error_acumulado / len(loader_entreno)
        historial_error.append(promedio_error)
        print(f"Época {epoca}/{epocas} - Error: {promedio_error:.4f}")

        if promedio_error < mejor_error:
            mejor_error = promedio_error
            # Cambiar de directorio para evitar problemas de rutas Unicode en PyTorch/Windows
            directorio_actual = os.getcwd()
            os.chdir(str(DIR_MODELOS))
            try:
                torch.save({
                    'model_state_dict': modelo_red.state_dict(),
                    'optimizer_state_dict': optimizador.state_dict(),
                    'val_loss': mejor_error
                }, "best_model.pth")
            finally:
                os.chdir(directorio_actual)

    # Guardar curva de aprendizaje
    plt.figure()
    plt.plot(historial_error, color='teal')
    plt.title("Curva de Aprendizaje (Entrenamiento)")
    plt.savefig(DIR_FIGURAS / "entrenamiento_curva_error.png")
    plt.close()
    print("✓ Modelo guardado y curva generada.")
