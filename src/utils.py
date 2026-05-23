"""
utils.py
"""

import numpy as np
from typing import Dict, List

# Mapa de codificación para consistencia en el proyecto
MAPA_ONE_HOT: Dict[str, List[int]] = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1]
}

LONGITUD_SECUENCIA_ESTANDAR = 200

def codificar_secuencia(secuencia_adn: str, longitud_objetivo: int = 200) -> np.ndarray:
    """
    Codifica una secuencia de ADN en formato One-Hot.
    
    Parámetros:
        secuencia_adn (str): Cadena de nucleótidos (A, C, G, T).
        longitud_objetivo (int): Longitud final esperada de la matriz.
        
    Retorna:
        np.ndarray: Matriz de (longitud_objetivo, 4) con la codificación binaria.
    """
    secuencia_limpia = secuencia_adn.upper()
    if len(secuencia_limpia) != longitud_objetivo:
        secuencia_limpia = secuencia_limpia[:longitud_objetivo].ljust(longitud_objetivo, 'N')
        
    matriz_codificada = np.zeros((longitud_objetivo, 4), dtype=np.float32)
    for i, nucleotido in enumerate(secuencia_limpia):
        if nucleotido in MAPA_ONE_HOT:
            matriz_codificada[i] = MAPA_ONE_HOT[nucleotido]
            
    return matriz_codificada

def normalizar_posicion_genomica(posicion_actual: int, longitud_total: int) -> float:
    """
    Normaliza la posición dentro de un cromosoma al rango [0, 1].
    
    Parámetros:
        posicion_actual (int): Coordenada genómica.
        longitud_total (int): Tamaño total del cromosoma o región.
        
    Retorna:
        float: Valor normalizado.
    """
    if longitud_total <= 0:
        return 0.0
    return posicion_actual / longitud_total

def escalar_valor_min_max(valor: float, minimo: float, maximo: float) -> float:
    """
    Aplica escalamiento Min-Max a un valor numérico.
    
    Parámetros:
        valor (float): Cantidad a escalar.
        minimo (float): Valor mínimo del rango.
        maximo (float): Valor máximo del rango.
        
    Retorna:
        float: Valor escalado entre 0 y 1.
    """
    if maximo == minimo:
        return 0.0
    return (valor - minimo) / (maximo - minimo)
