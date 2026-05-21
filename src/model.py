import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SplicingCNN1D(nn.Module):
    
    
    def __init__(
        self,
        input_length: int = 200,
        input_channels: int = 4,
        dropout_rate_conv: float = 0.4,
        dropout_rate_fc: float = 0.5
    ):
        
        super(SplicingCNN1D, self).__init__()

        # BLOQUE CONVOLUCIONAL 1: Detectores de motivos largos
        # Kernel=15: Captura contextos extendidos (~7bp a cada lado del centro)
        # Padding=7: Mantiene misma longitud de secuencia (200 -> 200)
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,   # 4 (one-hot)
            out_channels=64,               # 64 filtros/detectores
            kernel_size=15,                # Ventana larga para motivos complejos
            padding=7                      # Mantener longitud
        )
        
        self.bn1 = nn.BatchNorm1d(64)      # Normalización para estabilidad
        self.dropout1 = nn.Dropout(dropout_rate_conv)
        
        # MaxPool: Reduce dimensionalidad espacial 4x (200 -> 50)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        

        # BLOQUE CONVOLUCIONAL 2: Detectores de motivos medios
        # Kernel=11: Motivos intermedios (~5bp contexto)
        # Input: (batch, 64, 50) después del pooling
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=11,
            padding=5
        )
        
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate_conv)
        
        # MaxPool: Reduce 4x (50 -> 12)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        

        # BLOQUE CONVOLUCIONAL 3: Detectores de motivos cortos
        # Kernel=9: Motivos locales (~4bp contexto)
        # Input: (batch, 128, 12) después del segundo pooling
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=9,
            padding=4
        )
        
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate_conv - 0.1)  # 0.3
        
        # MaxPool final: Reduce 3x (12 -> 4)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)
        

        # CAPAS FULLY CONNECTED: Clasificador
        # Cálculo de features antes de flatten:
        # Después de pool3: (batch, 256, 4)
        # Flatten: batch * 256 * 4 = batch * 1024
        
        features_size = 256 * 4  # 1024
        
        self.fc1 = nn.Linear(features_size, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(dropout_rate_fc)
        
        self.fc2 = nn.Linear(256, 1)  # Salida binaria
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
 
        # TRANSFORMACIÓN DE ENTRADA
        # Input: (batch_size, 200, 4)
        # Conv1D espera: (batch_size, channels, length)
        # Permutar: (batch, 200, 4) -> (batch, 4, 200)
        x = x.permute(0, 2, 1)
        # Ahora: (batch_size, 4, 200)
        

        # BLOQUE 1: Motivos largos (kernel=15)
        # =================================================================
        # Conv1D: (batch, 4, 200) -> (batch, 64, 200)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # MaxPool1D: (batch, 64, 200) -> (batch, 64, 50)
        x = self.pool1(x)
        

        # BLOQUE 2: Motivos medios (kernel=11)
        # =================================================================
        # Conv1D: (batch, 64, 50) -> (batch, 128, 50)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # MaxPool1D: (batch, 128, 50) -> (batch, 128, 12)
        x = self.pool2(x)
        

        # BLOQUE 3: Motivos cortos (kernel=9)
        # =================================================================
        # Conv1D: (batch, 128, 12) -> (batch, 256, 12)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # MaxPool1D: (batch, 256, 12) -> (batch, 256, 4)
        x = self.pool3(x)
        

        # FLATTEN Y CAPAS DENSE
        # =================================================================
        # Flatten: (batch, 256, 4) -> (batch, 1024)
        x = x.view(x.size(0), -1)
        
        # Dense 1: (batch, 1024) -> (batch, 256)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        
        # Dense 2 (salida): (batch, 256) -> (batch, 1)
        x = self.fc2(x)
        
        # Sigmoid: (batch, 1) -> probabilidad [0, 1]
        x = self.sigmoid(x)
        
        return x
    
    def get_layer_shapes(self, input_shape: Tuple[int, int, int] = (1, 200, 4)) -> None:
        
        print("\n" + "=" * 70)
        print("DIMENSIONES POR CAPA")
        print("=" * 70)
        
        # Simular forward pass con batch de tamaño 1
        dummy_input = torch.zeros(1, *input_shape[1:])
        x = dummy_input.permute(0, 2, 1)  # (1, 4, 200)
        
        print(f"Entrada: {tuple(dummy_input.shape)} -> Permute: {tuple(x.shape)}")
        
        # Bloque 1
        x = self.conv1(x)
        print(f"After Conv1 (64 filtros, kernel=15): {tuple(x.shape)}")
        x = self.pool1(x)
        print(f"After Pool1 (stride=4): {tuple(x.shape)}")
        
        # Bloque 2
        x = self.conv2(x)
        print(f"After Conv2 (128 filtros, kernel=11): {tuple(x.shape)}")
        x = self.pool2(x)
        print(f"After Pool2 (stride=4): {tuple(x.shape)}")
        
        # Bloque 3
        x = self.conv3(x)
        print(f"After Conv3 (256 filtros, kernel=9): {tuple(x.shape)}")
        x = self.pool3(x)
        print(f"After Pool3 (stride=3): {tuple(x.shape)}")
        
        # FC
        x = x.view(x.size(0), -1)
        print(f"After Flatten: {tuple(x.shape)}")
        x = self.fc1(x)
        print(f"After FC1 (256 neuronas): {tuple(x.shape)}")
        x = self.fc2(x)
        print(f"After FC2 (salida): {tuple(x.shape)}")
        
        print("=" * 70)


class SplicingCNN1DLite(nn.Module):
    
    
    def __init__(self, input_length: int = 200, input_channels: int = 4):
        super(SplicingCNN1DLite, self).__init__()
        
        # Bloque 1
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=15, padding=7)
        self.pool1 = nn.MaxPool1d(4, stride=4)
        
        # Bloque 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, padding=5)
        self.pool2 = nn.MaxPool1d(4, stride=4)
        
        # FC
        self.fc1 = nn.Linear(64 * 3, 32)  # 64 filtros * 3 posiciones
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (batch, 200, 4) -> (batch, 4, 200)
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        
        return x


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable


def main():
    
    print("=" * 70)
    print("MÓDULO 4: ARQUITECTURA CNN 1D - VALIDACIÓN")
    print("=" * 70)
    
    # Instanciar modelo
    print("\n[1] INSTANCIANDO MODELO CNN 1D...")
    model = SplicingCNN1D(
        input_length=200,
        input_channels=4,
        dropout_rate_conv=0.4,
        dropout_rate_fc=0.5
    )
    
    # Mostrar arquitectura
    print("\n[2] ARQUITECTURA DEL MODELO:")
    print(model)
    
    # Mostrar dimensiones por capa
    print("\n[3] FLUJO DE DIMENSIONES:")
    model.get_layer_shapes()
    
    # Contar parámetros
    total_params, trainable_params = count_parameters(model)
    print(f"\n[4] PARÁMETROS:")
    print(f"    Total: {total_params:,}")
    print(f"    Entrenables: {trainable_params:,}")
    
    # Prueba de forward pass
    print("\n[5] PRUEBA DE FORWARD PASS:")
    dummy_batch = torch.randn(32, 200, 4)  # batch_size=32
    output = model(dummy_batch)
    print(f"    Input shape: {dummy_batch.shape}")
    print(f"    Output shape: {output.shape}")
    print(f"    Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"    (Valores entre 0 y 1 por sigmoid)")
    
    print("\n" + "=" * 70)
    print("[OK] Arquitectura CNN 1D validada exitosamente.")
    print("=" * 70)
    print("\nSiguiente paso: Ejecutar Módulo 5 para entrenar el modelo.")


if __name__ == "__main__":
    main()