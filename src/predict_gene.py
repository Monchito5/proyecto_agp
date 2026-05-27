"""
predict_gene.py
"""

import sys
import argparse
import torch
import torch.nn as nn
import gffutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyfaidx import Fasta
from typing import Tuple

# Importaciones locales refactorizadas
from model import RedNeuronalSplicing
from utils import codificar_secuencia

# --- Constantes de Rutas ---
RAIZ_PROYECTO = Path(__file__).parent.parent
RUTA_EXPORT = RAIZ_PROYECTO / "data" / "export"
RUTA_FIGURAS = RAIZ_PROYECTO / "data" / "figures"
ARCHIVO_BD = RUTA_EXPORT / "gencode.v47.annotation.gtf.db"
ARCHIVO_FASTA = RAIZ_PROYECTO / "data" / "raw" / "GRCh38.primary_assembly.genome.fa"

def extraer_secuencia_gen(nombre_gen: str, db: gffutils.FeatureDB, fasta: Fasta) -> Tuple[str, int, str]:
    """Recupera la secuencia de un gen por su nombre."""
    lista = list(db.features_of_type("gene", filter_dict={"gene_name": nombre_gen}))
    if not lista: raise ValueError(f"Gen {nombre_gen} no hallado.")
    g = lista[0]
    return fasta[g.chrom][g.start : g.end].seq.upper(), g.start, g.chrom

def realizar_prediccion_deslizante(modelo: nn.Module, secuencia: str, dispositivo: str = 'cpu') -> pd.DataFrame:
    """Aplica el modelo sobre el gen buscando candidatos GT (Donantes) y AG (Aceptores)."""
    modelo.eval()
    resultados = []
    
    print("Escaneando secuencia del gen...")
    for i in range(0, len(secuencia) - 200, 5): # Paso de 5nt para mayor detalle
        ventana = secuencia[i : i + 200]
        dinuc_central = ventana[100:102]
        
        # Determinar tipo basado en el dinucleótido central
        tipo_val = None
        if dinuc_central == 'GT': tipo_val = 0.0 # Donante
        elif dinuc_central == 'AG': tipo_val = 1.0 # Aceptor
        
        if tipo_val is not None:
            matriz = codificar_secuencia(ventana, 200)
            tensor_x = torch.from_numpy(matriz).unsqueeze(0).to(dispositivo)
            tensor_t = torch.tensor([[tipo_val]], dtype=torch.float32).to(dispositivo)
            
            with torch.no_grad():
                prob = modelo(tensor_x, tensor_t).item()
                
            resultados.append({
                'pos_relativa': i + 100,
                'puntuacion': prob,
                'tipo': 'Donante' if tipo_val == 0.0 else 'Aceptor'
            })
            
    return pd.DataFrame(resultados)

def graficar_resultados(df: pd.DataFrame, nombre: str, ruta: Path):
    """Visualiza las puntuaciones por tipo de sitio."""
    plt.figure(figsize=(15, 6))
    for tipo, color in [('Donante', 'teal'), ('Aceptor', 'coral')]:
        sub = df[df['tipo'] == tipo]
        plt.scatter(sub['pos_relativa'], sub['puntuacion'], label=tipo, color=color, s=15, alpha=0.6)
        
    plt.axhline(0.5, color='black', linestyle='--', linewidth=1, label='Umbral')
    plt.title(f"Predicción de Sitios de Splicing: {nombre}", fontsize=13, fontweight='bold')
    plt.xlabel("Posición en el Gen (pb)")
    plt.ylabel("Probabilidad del Modelo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta, dpi=300)
    plt.close()

def principal():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=str, required=True)
    args = parser.parse_args()
    
    ruta_pesos = RUTA_EXPORT / "best_model.pth"
    if not ruta_pesos.exists(): return print("Error: No hay modelo entrenado.")

    db = gffutils.FeatureDB(str(ARCHIVO_BD))
    fasta = Fasta(str(ARCHIVO_FASTA))
    seq, inicio, chrom = extraer_secuencia_gen(args.gen, db, fasta)
    
    modelo = RedNeuronalSplicing() # Cargar arquitectura dinámica si es necesario
    checkpoint = torch.load(str(ruta_pesos), map_location='cpu')
    modelo.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    
    df_res = realizar_prediccion_deslizante(modelo, seq)
    out_dir = RUTA_FIGURAS / "predicciones_genes"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    graficar_resultados(df_res, args.gen, out_dir / f"{args.gen}_perfil.png")
    print(f"✓ Análisis de {args.gen} completado.")

if __name__ == "__main__":
    principal()
