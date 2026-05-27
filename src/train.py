"""
train.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Dict, List, Any
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score

# Importaciones locales estandarizadas
from dataset import DatasetSplicing
from model import RedNeuronalSplicing

# --- Rutas de Producción ---
RAIZ_PROYECTO = Path(__file__).parent.parent
DIR_PROCESADOS = RAIZ_PROYECTO / "data" / "processed"
DIR_FIGURAS = RAIZ_PROYECTO / "data" / "figures"
DIR_MODELOS = RAIZ_PROYECTO / "data" / "export"

# Configuración visual global
sns.set_theme(style="whitegrid")

def inicializar_cargadores_datos(ruta_entrenamiento: Path, ruta_prueba: Path) -> Tuple[DataLoader, DataLoader]:
    """
    Instancia los DataLoaders de PyTorch.
    """
    conjunto_entreno = DatasetSplicing(ruta_entrenamiento)
    conjunto_validacion = DatasetSplicing(ruta_prueba)

    cargador_entreno = DataLoader(conjunto_entreno, batch_size=32, shuffle=True)
    cargador_val = DataLoader(conjunto_validacion, batch_size=32, shuffle=False)

    return cargador_entreno, cargador_val

def evaluar_modelo(modelo: nn.Module, loader: DataLoader, dispositivo: str) -> Dict[str, Any]:
    """
    Calcula métricas integrando el tipo de sitio.
    """
    modelo.eval()
    todas_etiquetas = []
    todas_predicciones = []
    todas_probabilidades = []
    error_acumulado = 0.0
    criterio = nn.BCELoss()

    with torch.no_grad():
        for tensores_x, etiquetas_y, tensores_tipo in loader:
            tensores_x = tensores_x.to(dispositivo)
            etiquetas_y = etiquetas_y.to(dispositivo)
            tensores_tipo = tensores_tipo.to(dispositivo)
            
            salida_prob = modelo(tensores_x, tensores_tipo)
            error = criterio(salida_prob, etiquetas_y)
            error_acumulado += error.item()
            
            prediccion = (salida_prob > 0.5).float()
            
            todas_etiquetas.extend(etiquetas_y.cpu().numpy())
            todas_predicciones.extend(prediccion.cpu().numpy())
            todas_probabilidades.extend(salida_prob.cpu().numpy())

    etiquetas_np = np.array(todas_etiquetas).flatten()
    pred_np = np.array(todas_predicciones).flatten()
    prob_np = np.array(todas_probabilidades).flatten()

    fpr, tpr, _ = roc_curve(etiquetas_np, prob_np)
    roc_auc = auc(fpr, tpr)

    return {
        "loss": error_acumulado / len(loader),
        "accuracy": accuracy_score(etiquetas_np, pred_np),
        "f1": f1_score(etiquetas_np, pred_np),
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr
    }

def graficar_metrias_entrenamiento(historial: Dict[str, List[float]], ruta_salida: Path):
    """Genera gráficas de desempeño."""
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(historial['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(historial['val_loss'], label='Val Loss', color='#d62728', linestyle='--', linewidth=2)
    plt.title("Curvas de Pérdida")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(historial['val_acc'], label='Val Acc', color='#2ca02c', linewidth=2)
    plt.plot(historial['val_f1'], label='Val F1', color='#ff7f0e', linewidth=2)
    plt.title("Métricas de Clasificación")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(ruta_salida / "entrenamiento_curvas_desempeño.png"), dpi=300)
    plt.close()

def graficar_curva_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, ruta_salida: Path):
    """Genera curva ROC final."""
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.fill_between(fpr, tpr, alpha=0.1, color='orange')
    plt.title('Curva ROC: Splicing (Donantes + Aceptores)')
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.6)
    plt.savefig(str(ruta_salida / "evaluacion_final_roc.png"), dpi=300)
    plt.close()

def ejecutar_entrenamiento_robusto(
    modelo_red: nn.Module, 
    loader_entreno: DataLoader, 
    loader_val: DataLoader, 
    lr: float = 0.001,
    epocas: int = 10
):
    """
    Entrenamiento integrando metadatos de tipo de sitio.
    """
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo_red.to(dispositivo)
    optimizador = torch.optim.Adam(modelo_red.parameters(), lr=lr)
    criterio = nn.BCELoss()

    mejor_error_val = float('inf')
    historial = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    print(f"\nIniciando Entrenamiento (Donantes + Aceptores) | Disp: {dispositivo.upper()}")

    for epoca in range(1, epocas + 1):
        modelo_red.train()
        error_entreno_acum = 0.0
        
        for tensores_x, etiquetas_y, tensores_tipo in loader_entreno:
            tensores_x = tensores_x.to(dispositivo)
            etiquetas_y = etiquetas_y.to(dispositivo)
            tensores_tipo = tensores_tipo.to(dispositivo)
            
            optimizador.zero_grad()
            salida = modelo_red(tensores_x, tensores_tipo)
            error = criterio(salida, etiquetas_y)
            error.backward()
            optimizador.step()
            error_entreno_acum += error.item()

        metricas = evaluar_modelo(modelo_red, loader_val, dispositivo)
        historial['train_loss'].append(error_entreno_acum / len(loader_entreno))
        historial['val_loss'].append(metricas['loss'])
        historial['val_acc'].append(metricas['accuracy'])
        historial['val_f1'].append(metricas['f1'])

        print(f"Época {epoca:02d}/{epocas} | Train: {historial['train_loss'][-1]:.4f} | Val: {metricas['loss']:.4f} | F1: {metricas['f1']:.4f}")

        if metricas['loss'] < mejor_error_val:
            mejor_error_val = metricas['loss']
            try:
                torch.save({
                    'model_state_dict': modelo_red.state_dict(),
                    'val_loss': float(mejor_error_val)
                }, str(DIR_MODELOS / "best_model.pth"))
            except Exception: pass

    graficar_metrias_entrenamiento(historial, DIR_FIGURAS)
    met = evaluar_modelo(modelo_red, loader_val, dispositivo)
    graficar_curva_roc(met['fpr'], met['tpr'], met['roc_auc'], DIR_FIGURAS)
    print("✓ Entrenamiento completado.")
