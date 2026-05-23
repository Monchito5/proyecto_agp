import torch
import numpy as np
from typing import Dict, List

# Mapeo One-Hot global para consistencia
ONE_HOT_MAP: Dict[str, List[int]] = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1]
}

def encode_sequence(sequence: str, length: int = 200) -> np.ndarray:
    """
    Codifica una secuencia de ADN en formato One-Hot.
    Limpia caracteres inválidos o 'N' asumiendo que ya fueron filtrados.
    """
    if len(sequence) != length:
        # Si es menor, hacer padding con ceros (o manejar error)
        # Si es mayor, truncar. En este proyecto deben ser exactas.
        sequence = sequence[:length].ljust(length, 'N')
        
    encoded = np.zeros((length, 4), dtype=np.float32)
    for i, nt in enumerate(sequence.upper()):
        if nt in ONE_HOT_MAP:
            encoded[i] = ONE_HOT_MAP[nt]
        else:
            # 'N' o nucleótido inválido se queda como [0, 0, 0, 0]
            continue
    return encoded

def normalize_genomic_pos(pos: int, chrom_len: int) -> float:
    """Normaliza la posición genómica entre 0 y 1."""
    return pos / chrom_len if chrom_len > 0 else 0.0

def scale_feature(val: float, min_val: float, max_val: float) -> float:
    """Escalado Min-Max simple."""
    if max_val == min_val: return 0.0
    return (val - min_val) / (max_val - min_val)
