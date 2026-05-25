"""
predict_gene.py
"""

import sys
import argparse
import torch
import gffutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path
from pyfaidx import Fasta
from typing import Tuple

# Importaciones locales refactorizadas
from model import RedNeuronalSplicing
from utils import codificar_secuencia
from interpretabilidad import calcular_mapas_saliencia

# --- Constantes de Rutas de Producción ---
RAIZ_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS_RAW = RAIZ_PROYECTO / "data" / "raw"
RUTA_DATOS_EXPORT = RAIZ_PROYECTO / "data" / "export"
RUTA_DATOS_FIGURAS = RAIZ_PROYECTO / "data" / "figures"
ARCHIVO_BD_GENOMICA = RUTA_DATOS_EXPORT / "gencode.v47.annotation.gtf.db"
ARCHIVO_FASTA_GENOMA = RUTA_DATOS_RAW / "GRCh38.primary_assembly.genome.fa"
RUTA_MODELO_PESOS = RUTA_DATOS_EXPORT / "best_model.pth"

def extraer_secuencia_gen(nombre_gen: str, db_genomica: gffutils.FeatureDB, fasta_adn: Fasta) -> Tuple[str, int, str]:
    """
    Recupera la secuencia de nucleótidos de un gen específico.
    """
    lista_genes = list(db_genomica.features_of_type("gene", filter_dict={"gene_name": nombre_gen}))
    
    if not lista_genes:
        raise ValueError(f"No se encontró el gen '{nombre_gen}' en la base de datos.")
    
    objeto_gen = lista_genes[0]
    cadena_adn = fasta_adn[objeto_gen.chrom][objeto_gen.start : objeto_gen.end].seq.upper()
    
    print(f"✓ Secuencia de {nombre_gen} obtenida ({len(cadena_adn)} pb).")
    return cadena_adn, objeto_gen.start, objeto_gen.chrom

def realizar_prediccion_deslizante(modelo_red: nn.Module, secuencia_larga: str, dispositivo: str = 'cpu') -> pd.DataFrame:
    """
    Aplica el modelo sobre una secuencia larga usando ventana deslizante de 200pb.
    """
    modelo_red.eval()
    lista_resultados = []
    paso_desplazamiento = 10
    
    print("Iniciando ventana deslizante sobre el gen...")
    for i in range(0, len(secuencia_larga) - 200, paso_desplazamiento):
        fragmento = secuencia_larga[i : i + 200]
        
        # Filtro biológico rápido: buscar GT central (posición 100)
        if fragmento[100:102] != 'GT':
            continue
            
        matriz_adn = codificar_secuencia(fragmento, 200)
        tensor_adn = torch.from_numpy(matriz_adn).unsqueeze(0).to(dispositivo)
        
        with torch.no_grad():
            probabilidad = modelo_red(tensor_adn).item()
            
        lista_resultados.append({
            'posicion_relativa': i,
            'puntuacion': probabilidad,
            'secuencia_fragmento': fragmento
        })
        
    return pd.DataFrame(lista_resultados)

def graficar_paisaje_splicing(tabla_predicciones: pd.DataFrame, nombre_gen: str, ruta_grafica: Path):
    """Genera un perfil de puntuaciones a lo largo del gen."""
    plt.figure(figsize=(15, 5))
    plt.plot(tabla_predicciones['posicion_relativa'], tabla_predicciones['puntuacion'], color='blue', alpha=0.6)
    plt.axhline(0.5, color='red', linestyle='--', label='Umbral (0.5)')
    plt.title(f"Perfil de Predicción de Splicing: {nombre_gen}")
    plt.xlabel("Posición Relativa (pb)")
    plt.ylabel("Probabilidad de Sitio Donante")
    plt.savefig(ruta_grafica)
    plt.close()

def principal():
    """Punto de entrada para predicción de genes específicos."""
    analizador = argparse.ArgumentParser()
    analizador.add_argument("--gen", type=str, required=True, help="Símbolo del gen (ej. BRCA1).")
    argumentos = analizador.parse_args()
    
    if not RUTA_MODELO_PESOS.exists():
        print("Error: No se encontró el modelo entrenado.")
        return

    db_gen = gffutils.FeatureDB(str(ARCHIVO_BD_GENOMICA))
    fasta_gen = Fasta(str(ARCHIVO_FASTA_GENOMA))
    
    secuencia, inicio, chrom = extraer_secuencia_gen(argumentos.gen, db_gen, fasta_gen)
    
    modelo = RedNeuronalSplicing() # Asume arquitectura por defecto
    modelo.load_state_dict(torch.load(RUTA_MODELO_PESOS, map_location='cpu'))
    
    tabla_resultados = realizar_prediccion_deslizante(modelo, secuencia)
    
    directorio_salida = RUTA_DATOS_FIGURAS / "predicciones_genes"
    directorio_salida.mkdir(parents=True, exist_ok=True)
    
    graficar_paisaje_splicing(tabla_resultados, argumentos.gen, directorio_salida / f"{argumentos.gen}_perfil.png")
    print(f"✓ Análisis del gen {argumentos.gen} completado.")

if __name__ == "__main__":
    principal()
