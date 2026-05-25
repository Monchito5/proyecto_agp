"""
interpretabilidad.py
Módulo para la extracción de motivos biológicos y visualización de interpretabilidad
aplicando Saliency Maps, Integrated Gradients, y extracción de PWMs.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from torch.utils.data import DataLoader
import warnings

# Importaciones locales
from dataset import DatasetSplicing

# Mapa inverso para decodificación
MAP_INVERSO = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


def secuencia_de_tensor(tensor: torch.Tensor) -> str:
    """Convierte un tensor one-hot a una secuencia de ADN."""
    indices = torch.argmax(tensor, dim=-1).cpu().numpy()
    return ''.join([MAP_INVERSO.get(i, 'N') for i in indices])


def onehot_to_values(tensor: torch.Tensor) -> np.ndarray:
    """
    Convierte un tensor one-hot a una matriz de pesos por nucleótido.
    Para la visualización de Saliency Maps con logomaker, se usa la
    importancia derivada del gradiente, no el one-hot directo.
    """
    return tensor.cpu().numpy()


def calcular_saliency_maps(modelo: nn.Module, tensor_entrada: torch.Tensor) -> torch.Tensor:
    """
    Calcula los Saliency Maps básicos para un batch de secuencias.
    
    Parámetros:
        modelo (nn.Module): Modelo entrenado.
        tensor_entrada (torch.Tensor): Batch de secuencias one-hot (batch, L, 4).
        
    Retorna:
        torch.Tensor: Gradiente absoluto de la salida respecto a la entrada (batch, L, 4).
    """
    modelo.eval()
    tensor_entrada.requires_grad_(True)
    modelo.zero_grad()
    
    salida = modelo(tensor_entrada)
    # Asumiendo que salida es un vector (batch, 1), tomamos la clase positiva (índice 0)
    grad_outputs = torch.ones_like(salida)
    
    gradientes = torch.autograd.grad(
        outputs=salida, 
        inputs=tensor_entrada, 
        grad_outputs=grad_outputs,
        create_graph=False, 
        retain_graph=False
    )[0]
    
    return gradientes.abs()


def calcular_integrated_gradients(
    modelo: nn.Module, 
    secuencia_base: torch.Tensor, 
    referencia: Optional[torch.Tensor] = None, 
    pasos: int = 50
) -> torch.Tensor:
    """
    Implementación de Integrated Gradients (IG) para secuencias de ADN.
    
    Parámetros:
        modelo (nn.Module): Modelo entrenado.
        secuencia_base (torch.Tensor): Secuencia one-hot (L, 4).
        referencia (torch.Tensor, opcional): Secuencia de referencia (ej. zeros).
        pasos (int): Número de pasos para la interpolación.
        
    Retorna:
        torch.Tensor: Atribuciones Integrated Gradients (L, 4).
    """
    modelo.eval()
    if referencia is None:
        referencia = torch.zeros_like(secuencia_base)
    
    # Interpolación entre referencia y secuencia_base
    escalas = torch.linspace(0, 1, steps=pasos + 1).view(-1, 1, 1).to(secuencia_base.device)
    secuencia_interpolada = referencia.unsqueeze(0) + escalas * (secuencia_base.unsqueeze(0) - referencia.unsqueeze(0))
    secuencia_interpolada.requires_grad_(True)
    
    salidas = modelo(secuencia_interpolada)
    grad_outputs = torch.ones_like(salidas)
    
    gradientes = torch.autograd.grad(
        outputs=salidas,
        inputs=secuencia_interpolada,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Promediar gradientes y multiplicar por (x - x')
    atribuciones = gradientes.mean(dim=0) * (secuencia_base - referencia)
    return atribuciones.detach()


def extraer_pwms(modelo: nn.Module, nombre_capa: str = 'capa_conv1') -> np.ndarray:
    """
    Extrae los pesos de una capa convolucional y los transforma en Position Weight Matrices (PWMs).
    
    Parámetros:
        modelo (nn.Module): Modelo entrenado.
        nombre_capa (str): Nombre exacto del atributo de la capa (ej. 'capa_conv1').
        
    Retorna:
        np.ndarray: Array de shape (num_filtros, kernel_size, 4).
    """
    capa = getattr(modelo, nombre_capa)
    pesos = capa.weight.detach().cpu().numpy()  # (out_channels, in_channels, kernel_size)
    
    # Los pesos en PyTorch son (out, in, kernel). Necesitamos (out, kernel, 4)
    # Asumiendo in_channels=4 (A, C, G, T)
    if pesos.shape[1] != 4:
        raise ValueError(f"Se esperaban 4 canales de entrada, se encontró {pesos.shape[1]}")
    
    # Transponer para tener (out, kernel, 4)
    pwms = pesos.transpose(0, 2, 1)
    
    # Normalizar para que sumen 1 en la dimensión de nucleótidos (simular probabilidades)
    pwms = np.clip(pwms, a_min=0, a_max=None) # Solo pesos positivos para la visualización tipo logo
    sumas = pwms.sum(axis=-1, keepdims=True)
    sumas[sumas == 0] = 1 # Evitar división por cero
    pwms = pwms / sumas
    
    return pwms


def visualizar_pwm_logomaker(pwms: np.ndarray, ruta_salida_dir: Path, num_a_mostrar: int = 5):
    """
    Genera logos de secuencia para los primeros N filtros usando Logomaker.
    
    Parámetros:
        pwms (np.ndarray): Array de PWMs shape (num_filtros, kernel_size, 4).
        ruta_salida_dir (Path): Directorio para guardar las figuras.
        num_a_mostrar (int): Cantidad de filtros a visualizar.
    """
    ruta_salida_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(min(num_a_mostrar, pwms.shape[0])):
        pwm = pwms[i]
        df_pwm = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])
        
        plt.figure(figsize=(8, 3))
        logo = logomaker.Logo(df_pwm, color_scheme={'A': 'red', 'C': 'blue', 'G': 'orange', 'T': 'green'})
        logo.style_xticks(anchor=0)
        plt.title(f"Logo del Filtro Convolucional {i+1}")
        plt.tight_layout()
        plt.savefig(ruta_salida_dir / f"logo_filtro_{i+1}.png", dpi=300)
        plt.close()


def agregar_importancia_por_posicion(saliency_batch: torch.Tensor) -> np.ndarray:
    """
    Agrega la importancia por posición para generar un perfil de importancia global.
    
    Parámetros:
        saliency_batch (torch.Tensor): Tensor de saliency de shape (batch, L, 4).
        
    Retorna:
        np.ndarray: Vector de importancia promedio por posición (L,).
    """
    # Sumar sobre los 4 nucleótidos para obtener importancia posicional
    importancia_posicional = saliency_batch.sum(dim=-1)
    # Promediar sobre el batch
    perfil = importancia_posicional.mean(dim=0).cpu().numpy()
    return perfil


def guardar_perfiles_importancia(perfil: np.ndarray, ruta_salida: Path):
    """
    Guarda un plot del perfil de importancia agregada.
    
    Parámetros:
        perfil (np.ndarray): Vector de importancia por posición (L,).
        ruta_salida (Path): Ruta para guardar la figura.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(perfil, color='purple')
    plt.fill_between(range(len(perfil)), perfil, alpha=0.3, color='purple')
    plt.title("Perfil de Importancia Agregada (Saliency Map)")
    plt.xlabel("Posición en la Secuencia (nt)")
    plt.ylabel("Importancia del Gradient")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def generar_logo_por_atribucion(
    secuencias_onehot: np.ndarray, 
    atribuciones: np.ndarray, 
    ruta_salida: Path
):
    """
    Genera un logo de secuencia donde la altura de cada nucleótido
    representa su importancia (weighted sequence logo).
    
    Parámetros:
        secuencias_onehot (np.ndarray): Array de one-hot (batch, L, 4).
        atribuciones (np.ndarray): Importancia por nucleótido (batch, L, 4).
        ruta_salida (Path): Ruta para guardar el logo.
    """
    # Agregar atribuciones sólamente donde el nucleótido está presente (one-hot es 1)
    pesos = secuencias_onehot * atribuciones
    # Promedio sobre el batch
    promedio = pesos.mean(axis=0) # (L, 4)
    
    df = pd.DataFrame(promedio, columns=['A', 'C', 'G', 'T'])
    
    plt.figure(figsize=(12, 4))
    logo = logomaker.Logo(df, color_scheme={'A': 'red', 'C': 'blue', 'G': 'orange', 'T': 'green'})
    logo.style_xticks(anchor=0)
    plt.title("Logo de Secuencia basado en Atribución de Importancia")
    plt.ylabel("Importancia Agregada")
    plt.xlabel("Posición (nt)")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def interpretar_modelo(
    modelo: nn.Module, 
    ruta_test_csv: Path, 
    ruta_salida_figuras: Path, 
    dispositivo: str = 'cpu',
    num_muestras_ig: int = 10
):
    """
    Función orquestadora que corre todo el pipeline de interpretabilidad.
    
    Parámetros:
        modelo (nn.Module): Modelo entrenado.
        ruta_test_csv (Path): Ruta al CSV del conjunto de prueba.
        ruta_salida_figuras (Path): Directorio para guardar visualizaciones.
        dispositivo (str): 'cpu' o 'cuda'.
        num_muestras_ig (int): Número de muestras para calcular Integrated Gradients.
    """
    ruta_salida_figuras.mkdir(parents=True, exist_ok=True)
    modelo = modelo.to(dispositivo)
    modelo.eval()

    dataset = DatasetSplicing(ruta_test_csv)
    # Cargar un batch para Saliency Maps (aumentar batch_size para mejor estadística)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    saliency_maps_list = []
    secuencias_list = []
    atribuciones_list = []
    
    print("Calculando Saliency Maps e Integrated Gradients...")
    with torch.no_grad():
        for i, (secuencias, etiquetas) in enumerate(loader):
            secuencias = secuencias.to(dispositivo)
            
            # 1. Calcular Saliency Maps
            saliency = calcular_saliency_maps(modelo, secuencias)
            saliency_maps_list.append(saliency.cpu())
            secuencias_list.append(secuencias.cpu())
            
            # 2. Calcular Integrated Gradients para unas pocas muestras
            if i == 0: # Solo para el primer batch para no saturar tiempo/computo
                for j in range(min(num_muestras_ig, secuencias.shape[0])):
                    ig = calcular_integrated_gradients(modelo, secuencias[j])
                    atribuciones_list.append(ig)
                
            if i >= 10: # Limitar a los primeros 10 batches para agregación rápida
                break
            
    # Concatenar resultados
    saliencias_completas = torch.cat(saliency_maps_list, dim=0)
    secuencias_completas = torch.cat(secuencias_list, dim=0)
    
    # --- Paso A: Perfil de importancia posicional agregado ---
    perfil = agregar_importancia_por_posicion(saliencias_completas)
    guardar_perfiles_importancia(perfil, ruta_salida_figuras / "perfil_importancia_saliency.png")
    print(f"✓ Perfil de importancia generado en {ruta_salida_figuras}")

    # --- Paso B: Generar Logo por Atribución ---
    if atribuciones_list:
        atribuciones_tensor = torch.stack(atribuciones_list)
        generar_logo_por_atribucion(
            secuencias_completas[:num_muestras_ig].numpy(), 
            atribuciones_tensor.cpu().numpy(), 
            ruta_salida_figuras / "logo_atribucion_integrated.png"
        )
        print(f"✓ Logo por atribución generado.")

    # --- Paso C: Extracción y visualización de PWMs ---
    print("Extrayendo pesos de la primera capa convolucional...")
    pwms = extraer_pwms(modelo)
    visualizar_pwm_logomaker(pwms, ruta_salida_figuras / "pwms")
    print(f"✓ PWMs y logos de filtros generados en {ruta_salida_figuras / 'pwms'}")


if __name__ == "__main__":
    # Ejemplo de uso standalone (si el modelo ya está entrenado)
    from model import RedNeuronalSplicing
    from pathlib import Path
    
    model = RedNeuronalSplicing()
    # Cargar pesos si existen
    # model.load_state_dict(torch.load('data/export/best_model.pth'))
    
    # ruta_test = Path("data/processed/dataset_prueba.csv")
    # interpretar_modelo(model, ruta_test, Path("data/figures/interpretability"))
    print("Ejecutar interpretar_modelo() desde main.py o un script externo.")
