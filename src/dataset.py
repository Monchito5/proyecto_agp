"""
dataset.py
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple

# Importación local de utilidades refactorizadas
from utils import codificar_secuencia, LONGITUD_SECUENCIA_ESTANDAR

class DatasetSplicing(Dataset):
    """
    Clase para el manejo de datos genómicos en PyTorch.
    Realiza la codificación One-Hot al vuelo para optimizar memoria.
    """

    def __init__(self, ruta_csv: Path):
        """
        Inicializa el cargador de datos.
        
        Parámetros:
            ruta_csv (Path): Ubicación del archivo con secuencias y etiquetas.
        """
        if not ruta_csv.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos en: {ruta_csv}")
            
        self.tabla_datos = pd.read_csv(ruta_csv)
        
        if 'sequence' not in self.tabla_datos.columns or 'label' not in self.tabla_datos.columns:
            raise ValueError("El CSV debe contener las columnas 'sequence' y 'label'.")
            
        self.lista_secuencias = self.tabla_datos['sequence'].tolist()
        self.lista_etiquetas = self.tabla_datos['label'].tolist()
        
        print(f"[Dataset] Cargadas {len(self.lista_secuencias)} muestras.")

    def __len__(self) -> int:
        """Retorna el total de elementos en el dataset."""
        return len(self.lista_secuencias)

    def __getitem__(self, indice: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtiene una muestra codificada.
        
        Retorna:
            Tuple: (Tensor de secuencia One-Hot, Tensor de etiqueta).
        """
        cadena_adn = self.lista_secuencias[indice]
        valor_etiqueta = self.lista_etiquetas[indice]
        
        # Codificación One-Hot (Punto clave de procesamiento)
        matriz_codificada = codificar_secuencia(cadena_adn, LONGITUD_SECUENCIA_ESTANDAR)
        
        # Conversión a tensores de PyTorch
        tensor_secuencia = torch.from_numpy(matriz_codificada)
        tensor_etiqueta = torch.tensor([valor_etiqueta], dtype=torch.float32)
        
        return tensor_secuencia, tensor_etiqueta
