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
    """

    def __init__(self, ruta_csv: Path):
        """
        Inicializa el cargador de datos.
        """
        if not ruta_csv.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos en: {ruta_csv}")
            
        self.tabla_datos = pd.read_csv(ruta_csv)
        
        requeridos = ['sequence', 'label', 'tipo_sitio']
        if not all(col in self.tabla_datos.columns for col in requeridos):
            # Fallback para compatibilidad con datasets viejos
            if 'tipo_sitio' not in self.tabla_datos.columns:
                self.tabla_datos['tipo_sitio'] = 'donante'
            
        self.lista_secuencias = self.tabla_datos['sequence'].tolist()
        self.lista_etiquetas = self.tabla_datos['label'].tolist()
        # Mapeo: donante -> 0, aceptor -> 1
        self.lista_tipos = self.tabla_datos['tipo_sitio'].map({'donante': 0, 'aceptor': 1}).tolist()
        
        print(f"[Dataset] Cargadas {len(self.lista_secuencias)} muestras.")

    def __len__(self) -> int:
        return len(self.lista_secuencias)

    def __getitem__(self, indice: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retorna: (secuencia, etiqueta, tipo)
        """
        cadena_adn = self.lista_secuencias[indice]
        valor_etiqueta = self.lista_etiquetas[indice]
        valor_tipo = self.lista_tipos[indice]
        
        matriz_codificada = codificar_secuencia(cadena_adn, LONGITUD_SECUENCIA_ESTANDAR)
        
        tensor_secuencia = torch.from_numpy(matriz_codificada)
        tensor_etiqueta = torch.tensor([valor_etiqueta], dtype=torch.float32)
        tensor_tipo = torch.tensor([valor_tipo], dtype=torch.float32)
        
        return tensor_secuencia, tensor_etiqueta, tensor_tipo
