"""
model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class RedNeuronalSplicing(nn.Module):
    """
    Arquitectura de Red Neuronal Convolucional 1D para predicción de sitios de splicing.
    Integra información del tipo de sitio (Donante/Aceptor) en la etapa densa.
    """
    
    def __init__(
        self,
        lista_filtros: List[int] = [64, 128, 256],
        lista_kernels: List[int] = [15, 11, 9],
        tasa_abandono_conv: float = 0.4,
        tasa_abandono_fc: float = 0.5
    ):
        """
        Inicializa las capas de la red.
        """
        super(RedNeuronalSplicing, self).__init__()

        # BLOQUE CONVOLUCIONAL 1
        self.capa_conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=lista_filtros[0],
            kernel_size=lista_kernels[0],
            padding=lista_kernels[0] // 2
        )
        self.normalizacion1 = nn.BatchNorm1d(lista_filtros[0])
        self.abandono1 = nn.Dropout(tasa_abandono_conv)
        self.agrupacion1 = nn.MaxPool1d(kernel_size=4, stride=4)

        # BLOQUE CONVOLUCIONAL 2
        self.capa_conv2 = nn.Conv1d(
            in_channels=lista_filtros[0],
            out_channels=lista_filtros[1],
            kernel_size=lista_kernels[1],
            padding=lista_kernels[1] // 2
        )
        self.normalizacion2 = nn.BatchNorm1d(lista_filtros[1])
        self.abandono2 = nn.Dropout(tasa_abandono_conv)
        self.agrupacion2 = nn.MaxPool1d(kernel_size=4, stride=4)

        # BLOQUE CONVOLUCIONAL 3
        self.capa_conv3 = nn.Conv1d(
            in_channels=lista_filtros[1],
            out_channels=lista_filtros[2],
            kernel_size=lista_kernels[2],
            padding=lista_kernels[2] // 2
        )
        self.normalizacion3 = nn.BatchNorm1d(lista_filtros[2])
        self.abandono3 = nn.Dropout(tasa_abandono_conv - 0.1)
        self.agrupacion3 = nn.MaxPool1d(kernel_size=3, stride=3)

        # CAPAS DENSAS
        # Entrada: Características de convolución + 1 neurona para el Tipo de Sitio
        self.dimension_entrada_fc = (lista_filtros[2] * 4) + 1
        self.capa_fc1 = nn.Linear(self.dimension_entrada_fc, 256)
        self.normalizacion_fc = nn.BatchNorm1d(256)
        self.abandono_fc = nn.Dropout(tasa_abandono_fc)
        
        self.capa_salida = nn.Linear(256, 1)
        self.activacion_final = nn.Sigmoid()

    def forward(self, tensor_x: torch.Tensor, tensor_tipo: torch.Tensor) -> torch.Tensor:
        """
        Flujo de datos integrando el tipo de sitio.
        """
        # Reordenar: (batch, 4, 200)
        tensor_x = tensor_x.permute(0, 2, 1)

        # Bloques Convolucionales
        tensor_x = F.relu(self.normalizacion1(self.capa_conv1(tensor_x)))
        tensor_x = self.agrupacion1(self.abandono1(tensor_x))

        tensor_x = F.relu(self.normalizacion2(self.capa_conv2(tensor_x)))
        tensor_x = self.agrupacion2(self.abandono2(tensor_x))

        tensor_x = F.relu(self.normalizacion3(self.capa_conv3(tensor_x)))
        tensor_x = self.agrupacion3(self.abandono3(tensor_x))

        # Aplanar y Concatenar metadatos (Tipo de Sitio)
        tensor_x = tensor_x.view(tensor_x.size(0), -1)
        # Concatenar el tipo de sitio (0 o 1) al vector de características
        tensor_combinado = torch.cat((tensor_x, tensor_tipo), dim=1)

        # Clasificador
        tensor_combinado = F.relu(self.normalizacion_fc(self.capa_fc1(tensor_combinado)))
        tensor_combinado = self.activacion_final(self.capa_salida(self.abandono_fc(tensor_combinado)))
        
        return tensor_combinado
