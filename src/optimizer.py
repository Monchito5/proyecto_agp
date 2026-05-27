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
    semilla_azar: int = 7

@dataclass
class ResultadoOptimizacion:
    """Estructura para almacenar el mejor individuo y su desempeño."""
    mejor_vector: np.ndarray
    mejor_aptitud: float
    historial_error: np.ndarray

class OptimizadorHiperparametros:
    """
    Gestiona la búsqueda de parámetros óptimos mediante Evolución Diferencial.
    """

    def __init__(self, ruta_entrenamiento: Path, ruta_validacion: Path):
        self.ruta_entrenamiento = ruta_entrenamiento
        self.ruta_validacion = ruta_validacion
        self.dispositivo_calculo = 'cuda' if torch.cuda.is_available() else 'cpu'

    def funcion_aptitud(self, vector_parametros: np.ndarray) -> float:
        """
        Evalúa el desempeño de una configuración de la CNN.
        """
        tasa_aprendizaje = 10**vector_parametros[0]
        tasa_abandono_capas = vector_parametros[1]
        lista_filtros = [int(vector_parametros[2]), int(vector_parametros[3]), int(vector_parametros[4])]
        lista_kernels = [
            2 * int(vector_parametros[5]) + 1,
            2 * int(vector_parametros[6]) + 1,
            2 * int(vector_parametros[7]) + 1
        ]

        print(f"\nEvaluando configuración: LR={tasa_aprendizaje:.4f}, Filtros={lista_filtros}")
        
        try:
            precision_obtenida = self._entrenar_y_evaluar(
                tasa_aprendizaje, tasa_abandono_capas, lista_filtros, lista_kernels
            )
            return 1.0 - precision_obtenida
        except Exception as error_ejecucion:
            print(f"Error en evaluación: {error_ejecucion}")
            return 1.0 

    def _entrenar_y_evaluar(self, tasa_lr: float, abandono: float, filtros: List[int], kernels: List[int]) -> float:
        """Entrenamiento minimalista para calcular el fitness."""
        dataset_entreno = DatasetSplicing(self.ruta_entrenamiento)
        dataset_val = DatasetSplicing(self.ruta_validacion)
        loader_entreno = DataLoader(dataset_entreno, batch_size=64, shuffle=True)
        loader_val = DataLoader(dataset_val, batch_size=64, shuffle=False)

        modelo_red = RedNeuronalSplicing(filtros, kernels, abandono, abandono).to(self.dispositivo_calculo)
        optimizador_red = torch.optim.Adam(modelo_red.parameters(), lr=tasa_lr)
        criterio_error = nn.BCELoss()

        for _ in range(3):
            modelo_red.train()
            for tensores_x, etiquetas_y, tensores_tipo in loader_entreno:
                tensores_x = tensores_x.to(self.dispositivo_calculo)
                etiquetas_y = etiquetas_y.to(self.dispositivo_calculo)
                tensores_tipo = tensores_tipo.to(self.dispositivo_calculo)
                
                optimizador_red.zero_grad()
                prediccion = modelo_red(tensores_x, tensores_tipo)
                criterio_error(prediccion, etiquetas_y).backward()
                optimizador_red.step()

        modelo_red.eval()
        conteo_correctos = 0
        total_muestras = 0
        with torch.no_grad():
            for tensores_x, etiquetas_y, tensores_tipo in loader_val:
                tensores_x = tensores_x.to(self.dispositivo_calculo)
                etiquetas_y = etiquetas_y.to(self.dispositivo_calculo)
                tensores_tipo = tensores_tipo.to(self.dispositivo_calculo)
                
                prediccion_binaria = (modelo_red(tensores_x, tensores_tipo) > 0.5).float()
                conteo_correctos += (prediccion_binaria == etiquetas_y).sum().item()
                total_muestras += etiquetas_y.size(0)

        return conteo_correctos / total_muestras if total_muestras > 0 else 0.0

def evolucion_diferencial(
    funcion_objetivo: Callable[[np.ndarray], float],
    limites_parametros: np.ndarray,
    config_evolutiva: ConfiguracionEvolutiva
) -> ResultadoOptimizacion:
    """
    Implementación del Algoritmo Evolutivo Diferencial.
    """
    generador_azar = np.random.default_rng(config_evolutiva.semilla_azar)
    inf, sup = limites_parametros[:, 0], limites_parametros[:, 1]
    dimension_problema = limites_parametros.shape[0]

    poblacion_actual = generador_azar.uniform(inf, sup, size=(config_evolutiva.tamano_poblacion, dimension_problema))
    aptitud_poblacion = np.array([funcion_objetivo(ind) for ind in poblacion_actual])

    indice_mejor = np.argmin(aptitud_poblacion)
    vector_mejor = poblacion_actual[indice_mejor].copy()
    aptitud_mejor = aptitud_poblacion[indice_mejor]

    historial_aptitud = [aptitud_mejor]

    for epoca_gen in range(1, config_evolutiva.total_generaciones + 1):
        print(f"\n--- Generación Evolutiva {epoca_gen}/{config_evolutiva.total_generaciones} ---")
        nueva_poblacion = poblacion_actual.copy()
        nueva_aptitud = aptitud_poblacion.copy()

        for i in range(config_evolutiva.tamano_poblacion):
            indices_candidatos = [idx for idx in range(config_evolutiva.tamano_poblacion) if idx != i]
            r1, r2, r3 = generador_azar.choice(indices_candidatos, size=3, replace=False)

            vector_mutante = poblacion_actual[r1] + config_evolutiva.factor_mutacion * (poblacion_actual[r2] - poblacion_actual[r3])
            
            j_azar = generador_azar.integers(dimension_problema)
            mascara_cruce = generador_azar.random(dimension_problema) < config_evolutiva.probabilidad_cruce
            mascara_cruce[j_azar] = True
            vector_hijo = np.where(mascara_cruce, vector_mutante, poblacion_actual[i])
            vector_hijo = np.clip(vector_hijo, inf, sup)

            aptitud_hijo = funcion_objetivo(vector_hijo)
            if aptitud_hijo < aptitud_poblacion[i]:
                nueva_poblacion[i] = vector_hijo
                nueva_aptitud[i] = aptitud_hijo

        poblacion_actual, aptitud_poblacion = nueva_poblacion, nueva_aptitud
        if np.min(aptitud_poblacion) < aptitud_mejor:
            indice_mejor = np.argmin(aptitud_poblacion)
            aptitud_mejor = aptitud_poblacion[indice_mejor]
            vector_mejor = poblacion_actual[indice_mejor].copy()

        historial_aptitud.append(aptitud_mejor)
        print(f"Mejor aptitud (Error de Val): {aptitud_mejor:.4f}")

    return ResultadoOptimizacion(vector_mejor, aptitud_mejor, np.array(historial_aptitud))
