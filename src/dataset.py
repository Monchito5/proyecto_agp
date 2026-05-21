import os
import sys
from pathlib import Path
from typing import Tuple, Dict, List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Configuración de rutas
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_EXPORT_DIR = PROJECT_ROOT / "data" / "export"

# Archivo de entrada
CONSOLIDATED_FILE = DATA_EXPORT_DIR / "dataset_consolidado_balanceado.csv"

# Mapeo One-Hot
ONE_HOT_MAP: Dict[str, List[int]] = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1]
}

NUCLEOTIDE_ORDER = ['A', 'C', 'G', 'T']
SEQUENCE_LENGTH = 200
NUM_NUCLEOTIDES = 4


class SplicingDataset(Dataset):
    
    
    def __init__(self, csv_path: Path):
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {csv_path}\n"
                f"Asegúrate de ejecutar primero el Módulo 2."
            )
        
        # Cargar datos
        self.df = pd.read_csv(csv_path)
        
        if self.df.empty:
            raise ValueError("El archivo CSV está vacío")
        
        required_cols = {'sequence', 'label'}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(
                f"El archivo debe contener las columnas: {required_cols}. "
                f"Columnas encontradas: {list(self.df.columns)}"
            )
        
        self.sequences = self.df['sequence'].tolist()
        self.labels = self.df['label'].tolist()
        
        print(f"[SplicingDataset] Cargadas {len(self.sequences)} secuencias")
        print(f"[SplicingDataset] Clase positiva (1): {sum(self.labels)}")
        print(f"[SplicingDataset] Clase negativa (0): {len(self.labels) - sum(self.labels)}")
    
    def __len__(self) -> int:
        """Retorna el número total de muestras en el dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Codificar secuencia en one-hot
        one_hot_encoded = self._encode_sequence(sequence)
        
        # Convertir a tensor PyTorch
        tensor = torch.FloatTensor(one_hot_encoded)
        
        return tensor, label
    
    def _encode_sequence(self, sequence: str) -> np.ndarray:
        
        # Validar longitud
        if len(sequence) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Secuencia tiene longitud {len(sequence)}, "
                f"esperada {SEQUENCE_LENGTH}"
            )
        
        # Codificar cada nucleótido
        encoded = np.zeros((SEQUENCE_LENGTH, NUM_NUCLEOTIDES), dtype=np.float32)
        
        for i, nucleotide in enumerate(sequence):
            if nucleotide in ONE_HOT_MAP:
                encoded[i] = ONE_HOT_MAP[nucleotide]
            else:
                raise ValueError(
                    f"Nucleótido inválido '{nucleotide}' en posición {i}. "
                    f"Solo se permiten: {NUCLEOTIDE_ORDER}"
                )
        
        return encoded
    
    def get_stats(self) -> Dict:
        
        return {
            'total_samples': len(self),
            'positive_class': sum(self.labels),
            'negative_class': len(self.labels) - sum(self.labels),
            'sequence_length': SEQUENCE_LENGTH,
            'encoding_dim': NUM_NUCLEOTIDES
        }


def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    
    sequences, labels = zip(*batch)
    return torch.stack(sequences), torch.tensor(labels)


def validate_dataset(dataset: SplicingDataset, n_samples: int = 5) -> None:
    
    print("\n" + "=" * 70)
    print("VALIDACIÓN DE CODIFICACIÓN ONE-HOT")
    print("=" * 70)
    
    for i in range(n_samples):
        tensor, label = dataset[i]
        sequence = dataset.sequences[i]
        
        print(f"\n[ muestra {i} ]")
        print(f"  Secuencia (texto): {sequence[:50]}...")
        print(f"  Label: {label}")
        print(f"  Tensor shape: {tensor.shape}")
        print(f"  Tensor dtype: {tensor.dtype}")
        
        # Verificar que es binario
        unique_values = torch.unique(tensor)
        print(f"  Valores únicos: {unique_values.tolist()}")
        
        # Mostrar primera posición como ejemplo
        print(f"  Posición 0 (primer nucleótido '{sequence[0]}'): {tensor[0].tolist()}")
    
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS DEL DATASET")
    print("=" * 70)
    stats = dataset.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 70)


def test_dataloader(dataset: SplicingDataset, batch_size: int = 32) -> None:
    
    print("\n" + "=" * 70)
    print("PRUEBA DE DATALOADER")
    print("=" * 70)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Obtener un batch
    batch_sequences, batch_labels = next(iter(dataloader))
    
    print(f"\n  Batch size: {batch_sequences.shape[0]}")
    print(f"  Tensor shape: {batch_sequences.shape}")
    print(f"  Labels shape: {batch_labels.shape}")
    print(f"  Labels distribution: {torch.bincount(batch_labels)}")
    
    print(f"[OK] DataLoader funcionando correctamente")
    print("=" * 70)


def main():
    
    print("=" * 70)
    print("MÓDULO 3: DATASET PYTORCH CON ONE-HOT ENCODING")
    print("=" * 70)
    print(f"Directorio de trabajo: {PROJECT_ROOT}")
    print(f"Archivo de entrada: {CONSOLIDATED_FILE}")
    print("=" * 70)
    
    try:
        # Paso 1: Instanciar dataset
        print("\n[1] INSTANCIANDO DATASET...")
        dataset = SplicingDataset(CONSOLIDATED_FILE)
        
        # Paso 2: Validar codificación
        print("\n[2] VALIDANDO CODIFICACIÓN ONE-HOT...")
        validate_dataset(dataset, n_samples=5)
        
        # Paso 3: Probar DataLoader
        print("\n[3] PROBANDO DATALOADER...")
        test_dataloader(dataset, batch_size=32)
        
        print(f"[OK] Módulo 3 validado exitosamente.")
        print("\nSiguiente paso: Ejecutar Módulo 4 para definir la arquitectura CNN.")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()