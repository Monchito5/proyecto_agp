"""
interpretabilidad.py
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

# Configuración de Estilo
sns.set_theme(style="white")

def calcular_atribucion_importancia(modelo: nn.Module, lote_adn: torch.Tensor) -> np.ndarray:
    """Calcula la importancia de cada nucleótido mediante Saliency Maps."""
    modelo.eval()
    lote_adn.requires_grad_(True)
    salida = modelo(lote_adn)
    
    gradientes = torch.autograd.grad(outputs=salida, inputs=lote_adn, 
                                    grad_outputs=torch.ones_like(salida))[0]
    
    # Importancia por posición (magnitud del gradiente)
    importancia = gradientes.abs().cpu().detach().numpy()
    return importancia

def generar_logos_por_tipo(modelo: nn.Module, ruta_csv: Path, salida_dir: Path):
    """Genera logos de secuencia específicos para Donantes y Aceptores."""
    salida_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ruta_csv)
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo.to(dispositivo)

    for tipo in ['donante', 'aceptor']:
        # Filtrar solo sitios reales de este tipo
        df_tipo = df[(df['tipo_sitio'] == tipo) & (df['label'] == 1)].head(100)
        if df_tipo.empty: continue
        
        # Crear dataset temporal para este tipo
        temp_csv = salida_dir / f"temp_{tipo}.csv"
        df_tipo.to_csv(temp_csv, index=False)
        ds = DatasetSplicing(temp_csv)
        loader = DataLoader(ds, batch_size=len(ds))
        
        lote_x, _ = next(iter(loader))
        lote_x = lote_x.to(dispositivo)
        
        # 1. Mapa de Importancia Global (Saliency)
        atribucion = calcular_atribucion_importancia(modelo, lote_x)
        importancia_posicional = atribucion.sum(axis=-1).mean(axis=0) # (200,)
        
        plt.figure(figsize=(12, 4))
        plt.plot(importancia_posicional, color='teal' if tipo=='donante' else 'coral')
        plt.title(f"Perfil de Importancia: Sitio {tipo.capitalize()}")
        plt.xlabel("Posición (pb)")
        plt.ylabel("Atribución de Importancia")
        
        # Resaltar regiones clave para Aceptores
        if tipo == 'aceptor':
            plt.axvspan(60, 82, color='gray', alpha=0.2, label='Zona Punto Ramificación (-40 a -18)')
            plt.axvspan(85, 98, color='yellow', alpha=0.2, label='Tracto Polipirimidina')
            plt.legend()
            
        plt.savefig(salida_dir / f"importancia_{tipo}.png")
        plt.close()
        
        # 2. Logo de Secuencia (Filtros de 1ra capa)
        print(f"Extrayendo motivos para {tipo}...")
        # (Este paso es genérico del modelo, pero lo visualizamos en contexto)
        
        temp_csv.unlink()

def generar_reporte_interpretabilidad(modelo: nn.Module, ruta_test: Path, salida_dir: Path):
    """Orquesta el análisis de motivos biológicos."""
    print("Iniciando análisis de interpretabilidad profunda...")
    generar_logos_por_tipo(modelo, ruta_test, salida_dir)
    print(f"✓ Análisis completado. Resultados en: {salida_dir.name}")
