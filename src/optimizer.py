"""
optimizer.py
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Callable, Tuple, Any
from pathlib import Path
from torch.utils.data import DataLoader

from model import RedNeuronalSplicing
from dataset import DatasetSplicing

@dataclass
class ConfiguracionEvolutiva:
    """Configuración para el algoritmo de Evolución Diferencial."""
    tamano_poblacion: int = 10
    total_generaciones: int = 20
    factor_mutacion: float = 0.7
    probabilidad_cruce: float = 0.9
    semilla: int = 7

@dataclass
class ResultadoOptimizacion:
    """Estructura para almacenar el mejor individuo y su desempeño."""
    mejor_vector: np.ndarray
    mejor_aptitud: float
    historial: np.ndarray

class OptimizadorHiperparametros:
    """
    Gestiona la búsqueda de parámetros óptimos mediante Evolución Diferencial.
    """

    def __init__(self, ruta_entreno: Path, ruta_val: Path):
        self.ruta_entreno = ruta_entreno
        self.ruta_val = ruta_val
        self.dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'

    def funcion_aptitud(self, vector_parametros: np.ndarray) -> float:
        """
        Evalúa el desempeño de una configuración de la CNN.
        Mapea el vector continuo a parámetros discretos y arquitectónicos.
        """
        # Mapeo de parámetros
        tasa_aprendizaje = 10**vector_parametros[0]
        tasa_abandono = vector_parametros[1]
        filtros = [int(vector_parametros[2]), int(vector_parametros[3]), int(vector_parametros[4])]
        # Kernels deben ser impares
        kernels = [
            2 * int(vector_parametros[5]) + 1,
            2 * int(vector_parametros[6]) + 1,
            2 * int(vector_parametros[7]) + 1
        ]

        print(f"\nEvaluando configuración: LR={tasa_aprendizaje:.4f}, Filtros={filtros}, Kernels={kernels}")
        
        try:
            # Entrenamiento rápido (3 épocas) para evaluación de fitness
            precision = self._entrenar_y_evaluar(tasa_aprendizaje, tasa_abandono, filtros, kernels)
            # Retornar 1 - precisión (para minimizar en ED)
            return 1.0 - precision
        except Exception as error_eval:
            print(f"Error en evaluación: {error_eval}")
            return 1.0 # Peor aptitud posible

    def _entrenar_y_evaluar(self, lr: float, dropout: float, filtros: List[int], kernels: List[int]) -> float:
        """Entrenamiento minimalista para calcular el fitness."""
        ds_entreno = DatasetSplicing(self.ruta_entreno)
        ds_val = DatasetSplicing(self.ruta_val)
        dl_entreno = DataLoader(ds_entreno, batch_size=64, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=64, shuffle=False)

        modelo = RedNeuronalSplicing(filtros, kernels, dropout, dropout).to(self.dispositivo)
        optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
        criterio = nn.BCELoss()

        # Ciclo de 3 épocas para aptitud rápida
        for _ in range(3):
            modelo.train()
            for x_batch, y_batch in dl_entreno:
                x_batch, y_batch = x_batch.to(self.dispositivo), y_batch.to(self.dispositivo)
                optimizador.zero_grad()
                pred = modelo(x_batch)
                criterio(pred, y_batch).backward()
                optimizador.step()

        # Evaluación
        modelo.eval()
        correctos = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in dl_val:
                x_batch, y_batch = x_batch.to(self.dispositivo), y_batch.to(self.dispositivo)
                pred = (modelo(x_batch) > 0.5).float()
                correctos += (pred == y_batch).sum().item()
                total += y_batch.size(0)

        return correctos / total if total > 0 else 0.0

def evolucion_diferencial(
    func_obj: Callable[[np.ndarray], float],
    limites: np.ndarray,
    config: ConfiguracionEvolutiva
) -> ResultadoOptimizacion:
    """
    Implementación del Algoritmo Evolutivo Diferencial (DE/rand/1/bin).
    """
    generador = np.random.default_rng(config.semilla)
    inf, sup = limites[:, 0], limites[:, 1]
    dim = limites.shape[0]

    # Inicialización
    poblacion = generador.uniform(inf, sup, size=(config.tamano_poblacion, dim))
    aptitud = np.array([func_obj(ind) for ind in poblacion])

    mejor_indice = np.argmin(aptitud)
    mejor_vector = poblacion[mejor_indice].copy()
    mejor_aptitud = aptitud[mejor_indice]

    historial = [mejor_aptitud]

    for gen in range(1, config.total_generaciones + 1):
        print(f"\n--- Generación Evolutiva {gen}/{config.total_generaciones} ---")
        nueva_poblacion = poblacion.copy()
        nueva_aptitud = aptitud.copy()

        for i in range(config.tamano_poblacion):
            candidatos = [idx for idx in range(config.tamano_poblacion) if idx != i]
            r1, r2, r3 = generador.choice(candidatos, size=3, replace=False)

            # Mutación
            mutante = poblacion[r1] + config.factor_mutacion * (poblacion[r2] - poblacion[r3])
            
            # Cruce Binomial
            j_azar = generador.integers(dim)
            mascara = generador.random(dim) < config.probabilidad_cruce
            mascara[j_azar] = True
            hijo = np.where(mascara, mutante, poblacion[i])
            hijo = np.clip(hijo, inf, sup)

            # Selección Codiciosa
            aptitud_hijo = func_obj(hijo)
            if aptitud_hijo < aptitud[i]:
                nueva_poblacion[i] = hijo
                nueva_aptitud[i] = aptitud_hijo

        poblacion, aptitud = nueva_poblacion, nueva_aptitud
        if np.min(aptitud) < mejor_aptitud:
            mejor_indice = np.argmin(aptitud)
            mejor_aptitud = aptitud[mejor_indice]
            mejor_vector = poblacion[mejor_indice].copy()

        historial.append(mejor_aptitud)
        print(f"Mejor aptitud actual (Error): {mejor_aptitud:.4f}")

    return ResultadoOptimizacion(mejor_vector, mejor_aptitud, np.array(historial))
