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

def calcular_atribucion_importancia(modelo: nn.Module, lote_adn: torch.Tensor, lote_tipo: torch.Tensor) -> np.ndarray:
    """Calcula la importancia de cada nucleótido integrando el tipo de sitio."""
    modelo.eval()
    lote_adn.requires_grad_(True)
    # El tipo de sitio es una constante para esta atribución
    lote_tipo.requires_grad_(False) 
    
    salida = modelo(lote_adn, lote_tipo)
    
    gradientes = torch.autograd.grad(outputs=salida, inputs=lote_adn, 
                                    grad_outputs=torch.ones_like(salida))[0]
    
    importancia = gradientes.abs().cpu().detach().numpy()
    return importancia

def generar_logos_por_tipo(modelo: nn.Module, ruta_csv: Path, salida_dir: Path):
    """Genera reportes de importancia específicos por tipo biológico."""
    salida_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ruta_csv)
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo.to(dispositivo)

    for tipo in ['donante', 'aceptor']:
        df_tipo = df[(df['tipo_sitio'] == tipo) & (df['label'] == 1)].head(100)
        if df_tipo.empty: continue
        
        temp_csv = salida_dir / f"temp_{tipo}.csv"
        df_tipo.to_csv(temp_csv, index=False)
        ds = DatasetSplicing(temp_csv)
        loader = DataLoader(ds, batch_size=len(ds))
        
        lote_x, _, lote_t = next(iter(loader))
        lote_x, lote_t = lote_x.to(dispositivo), lote_t.to(dispositivo)
        
        atribucion = calcular_atribucion_importancia(modelo, lote_x, lote_t)
        importancia_posicional = atribucion.sum(axis=-1).mean(axis=0)
        
        plt.figure(figsize=(12, 4))
        plt.plot(importancia_posicional, color='teal' if tipo=='donante' else 'coral', linewidth=2)
        plt.title(f"Perfil de Atribución (Saliency): {tipo.upper()}", fontsize=12, fontweight='bold')
        plt.xlabel("Posición en ventana de 200nt")
        plt.ylabel("Magnitud del Gradiente")
        
        if tipo == 'aceptor':
            # Resaltar regiones biológicas clave
            plt.axvspan(60, 82, color='gray', alpha=0.15, label='Zona Branch Point')
            plt.axvspan(85, 98, color='yellow', alpha=0.2, label='Tracto Polipirimidina')
            plt.legend(loc='upper left')
            
        plt.tight_layout()
        plt.savefig(salida_dir / f"importancia_{tipo}.png", dpi=300)
        plt.close()
        temp_csv.unlink()

def generar_reporte_interpretabilidad(modelo: nn.Module, ruta_test: Path, salida_dir: Path):
    """Interfaz principal para el análisis de interpretabilidad."""
    print("Iniciando análisis de interpretabilidad diferencial...")
    generar_logos_por_tipo(modelo, ruta_test, salida_dir)
    print(f"✓ Análisis de importancia completado.")
