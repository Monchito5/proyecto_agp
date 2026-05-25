"""
interpretabilidad.py
Módulo para la extracción de motivos biológicos y visualización de interpretabilidad
aplicando Saliency Maps, Integrated Gradients y extracción de PWMs.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker
from pathlib import Path
from typing import Optional, List, Tuple
from torch.utils.data import DataLoader

# Importaciones locales refactorizadas
from dataset import DatasetSplicing

# Mapa para decodificar índices One-Hot a caracteres de ADN
DICCIONARIO_INVERSO_ADN = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

def calcular_mapas_saliencia(modelo_entrenado: nn.Module, lote_entrada: torch.Tensor) -> torch.Tensor:
    """
    Calcula los Saliency Maps para identificar nucleótidos críticos.
    
    Parámetros:
        modelo_entrenado (nn.Module): Red neuronal cargada.
        lote_entrada (torch.Tensor): Tensores de secuencia (batch, L, 4).
        
    Retorna:
        torch.Tensor: Gradientes absolutos normalizados.
    """
    modelo_entrenado.eval()
    lote_entrada.requires_grad_(True)
    modelo_entrenado.zero_grad()
    
    vector_salida = modelo_entrenado(lote_entrada)
    mascara_gradiente = torch.ones_like(vector_salida)
    
    tensores_gradiente = torch.autograd.grad(
        outputs=vector_salida, 
        inputs=lote_entrada, 
        grad_outputs=mascara_gradiente,
        create_graph=False, 
        retain_graph=False
    )[0]
    
    return tensores_gradiente.abs()

def calcular_gradientes_integrados(
    modelo_red: nn.Module, 
    secuencia_objetivo: torch.Tensor, 
    secuencia_base: Optional[torch.Tensor] = None, 
    numero_pasos: int = 50
) -> torch.Tensor:
    """
    Implementa Integrated Gradients para atribución de importancia robusta.
    """
    modelo_red.eval()
    if secuencia_base is None:
        secuencia_base = torch.zeros_like(secuencia_objetivo)
    
    lista_escalas = torch.linspace(0, 1, steps=numero_pasos + 1).view(-1, 1, 1).to(secuencia_objetivo.device)
    lote_interpolado = secuencia_base.unsqueeze(0) + lista_escalas * (secuencia_objetivo.unsqueeze(0) - secuencia_base.unsqueeze(0))
    lote_interpolated = lote_interpolado.clone().detach().requires_grad_(True)
    
    salidas_red = modelo_red(lote_interpolated)
    mascara_grad = torch.ones_like(salidas_red)
    
    gradientes_totales = torch.autograd.grad(
        outputs=salidas_red,
        inputs=lote_interpolated,
        grad_outputs=mascara_grad,
        create_graph=False,
        retain_graph=False
    )[0]
    
    atribuciones_finales = gradientes_totales.mean(dim=0) * (secuencia_objetivo - secuencia_base)
    return atribuciones_finales.detach()

def extraer_matrices_pwm(modelo_cnn: nn.Module, nombre_capa: str = 'capa_conv1') -> np.ndarray:
    """
    Transforma los filtros de la primera capa en Position Weight Matrices (PWMs).
    """
    objeto_capa = getattr(modelo_cnn, nombre_capa)
    tensores_peso = objeto_capa.weight.detach().cpu().numpy() # (out, 4, kernel)
    
    # Transponer a formato (out, kernel, 4)
    coleccion_pwms = tensores_peso.transpose(0, 2, 1)
    
    # Normalización para visualización (solo pesos positivos relevantes)
    coleccion_pwms = np.clip(coleccion_pwms, a_min=0, a_max=None)
    sumatorias = coleccion_pwms.sum(axis=-1, keepdims=True)
    sumatorias[sumatorias == 0] = 1.0
    
    return coleccion_pwms / sumatorias

def generar_reporte_interpretabilidad(
    modelo_final: nn.Module, 
    ruta_datos_prueba: Path, 
    directorio_figuras: Path,
    dispositivo_uso: str = 'cpu'
):
    """
    Orquesta la generación de todas las visualizaciones de importancia.
    """
    directorio_figuras.mkdir(parents=True, exist_ok=True)
    modelo_final.to(dispositivo_uso).eval()
    
    dataset_eval = DatasetSplicing(ruta_datos_prueba)
    cargador_eval = DataLoader(dataset_eval, batch_size=32, shuffle=False)
    
    print("Iniciando análisis de interpretabilidad...")
    
    # Tomar un lote para análisis
    lote_adn, _ = next(iter(cargador_eval))
    lote_adn = lote_adn.to(dispositivo_uso)
    
    # 1. Saliency Maps
    mapas_saliencia = calcular_mapas_saliencia(modelo_final, lote_adn)
    perfil_importancia = mapas_saliencia.sum(dim=-1).mean(dim=0).cpu().numpy()
    
    plt.figure(figsize=(12, 4))
    plt.plot(perfil_importancia, color='purple')
    plt.title("Perfil de Importancia Global (Saliency)")
    plt.savefig(directorio_figuras / "interpretabilidad_saliencia.png")
    plt.close()
    
    # 2. PWMs
    pwms_extraidas = extraer_matrices_pwm(modelo_final)
    for i in range(min(5, pwms_extraidas.shape[0])):
        df_pwm = pd.DataFrame(pwms_extraidas[i], columns=['A', 'C', 'G', 'T'])
        plt.figure(figsize=(8, 3))
        logomaker.Logo(df_pwm)
        plt.title(f"Filtro Convolucional {i+1}")
        plt.savefig(directorio_figuras / f"motivo_filtro_{i+1}.png")
        plt.close()

    print(f"✓ Análisis de interpretabilidad completado en: {directorio_figuras.name}")
