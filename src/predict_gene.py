"""
predict_gene.py
Script para predecir sitios de splicing candidatos en genes de interés.
Permite ejecutar el modelo entrenado sobre un gen específico desde la línea de 
comandos y visualizar las regiones más relevantes.
"""

import sys
import argparse
import torch
import gffutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyfaidx import Fasta

# Importaciones locales
from model import RedNeuronalSplicing
from utils import codificar_secuencia, LONGITUD_SECUENCIA_ESTANDAR
from interpretabilidad import calcular_saliency_maps

# --- Constantes de Rutas ---
RAIZ = Path(__file__).parent.parent
RUTA_RAW = RAIZ / "data" / "raw"
RUTA_EXPORT = RAIZ / "data" / "export"
RUTA_FIGURAS = RAIZ / "data" / "figures"
ARCHIVO_BASE_DATOS = RUTA_EXPORT / "gencode.v47.annotation.gtf.db"
ARCHIVO_GENOMA_FASTA = RUTA_RAW / "GRCh38.primary_assembly.genome.fa"
RUTA_MODELO = RAIZ / "data" / "export" / "best_model.pth"


def obtener_secuencia_gen(gen_name: str, base_datos: gffutils.FeatureDB, genoma: Fasta):
    """
    Obtiene la secuencia genómica y coordenadas de un gen por su nombre (gene_name)
    o por su gene_id.
    
    Parámetros:
        gen_name (str): Gene_symbol (ej. 'BRCA1') o ENSG ID.
        base_datos (FeatureDB): Conexión a la base de datos GFFUtils.
        genoma (Fasta): Objeto Fasta del genoma.
        
    Retorna:
        Tuple[str, int, str, str, str]: (secuencia_completa, pos_start, chrom, strand, gene_id)
    """
    genes_encontrados = list(base_datos.features_of_type("gene", filter_dict={"gene_name": gen_name}))
    
    if not genes_encontrados and gen_name.startswith("ENSG"):
        # Fallback por gene_id
        try:
            genes_encontrados = [base_datos[gen_name]]
        except KeyError:
            pass
    
    if not genes_encontrados:
        raise ValueError(f"Gen '{gen_name}' no encontrado en la base de datos.")
    
    gen_obj = genes_encontrados[0]
    chrom = gen_obj.chrom
    start = gen_obj.start
    end = gen_obj.end
    
    # Extraer secuencia de Fasta (considerando limites del cromosoma)
    total_crom_len = len(genoma[chrom])
    start_idx = max(1, start)
    end_idx = min(total_crom_len, end)
    
    secuencia = genoma[chrom][start_idx:end_idx].seq.upper()
    
    print(f"Gen '{gen_name}' ({chrom}:{start}-{end}, longitud: {len(secuencia)} pb)")
    return secuencia, start, chrom, gen_obj.strand, genObj.attributes.get('gene_id', ['unknown'])[0]


def generar_ventana_deslizante(secuencia: str, ventana: int = 200, paso: int = 1):
    """
    Genera todas las ventanas deslizadas de una secuencia larga.
    
    Parámetros:
        secuencia (str): Secuencia de ADN.
        ventana (int): Tamaño de cada ventana (200 pb por defecto).
        paso (int): Desplazamiento entre ventanas.
        
    Yields:
        Tuple[int, str]: (posicion_inicial_0_indexed, subsecuencia)
    """
    longitud = len(secuencia)
    for i in range(0, longitud - ventana + 1, paso):
        yield i, secuencia[i:i+ventana]


def predecir_gen(modelo: RedNeuronalSplicing, secuencia: str, dispositivo: str = 'cpu', ventana: int = 200, paso: int = 10):
    """
    Aplica el modelo en ventanas deslizantes y devuelve las puntuaciones.
    
    Parámetros:
        modelo (RedNeuronalSplicing): Modelo entrenado.
        secuencia (str): Secuencia genómica completa del gen.
        dispositivo (str): 'cpu' o 'cuda'.
        ventana (int): Tamaño de ventana.
        paso (int): Paso de desplazamiento.
        
    Retorna:
        pd.DataFrame: DataFrame con 'posicion', 'probabilidad', y 'secuencia'.
    """
    modelo.eval()
    modelo.to(dispositivo)
    
    resultados = []
    for pos, subseq in generar_ventana_deslizante(secuencia, ventana, paso):
        # Verificar la presencia del dinucleótido GT en la posición 100 (centro)
        if 'GT' not in subseq[98:102]: # Revisar entorno del centro
            resultados.append({'posicion': pos, 'probabilidad': 0.0, 'secuencia': subseq})
            continue
            
        # Codificar y predecir
        one_hot = codificar_secuencia(subseq, VENTANA)
        tensor = torch.from_numpy(one_hot).float().unsqueeze(0).to(dispositivo) # (1, L, 4)
        
        with torch.no_grad():
            pred = modelo(tensor).item()
        
        resultados.append({'posicion': pos, 'probabilidad': pred, 'secuencia': subseq})
    
    return pd.DataFrame(resultados)


def visualizar_predicciones(df_pred: pd.DataFrame, gen_name: str, ruta_salida: Path):
    """
    Visualiza el paisaje de predicciones sobre un gen.
    
    Parámetros:
        df_pred (pd.DataFrame): DataFrame de resultados.
        gen_name (str): Nombre del gen.
        ruta_salida (Path): Directorio para guardar figuras.
    """
    ruta_salida.mkdir(parents=True, exist_ok=True)
    ruta_output = ruta_salida.
    
    # Plot de predicciones locales
    plt.figure(figsize=(15, 5))
    plt.plot(df_pred['posicion'], df_pred['probabilidad'], linewidth=0.8, alpha=0.7)
    plt.axhline(0.5, color='r', linestyle='--', label='Umbral de Decisión (0.5)')
    plt.title(f"Puntuaciones de Predicción de Splicing: Gen {gen_name}")
    plt.xlabel("Posición Genómica (relativa al inicio)")
    plt.ylabel("Probabilidad de Sitio Donante")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_salida / "predicciones_gen.png", dpi=300)
    plt.close()


def analizar_saliency_regiones(modelo, df_pred_altos, ruta_salida):
    """
    Calcula y visualiza Saliency Maps de las regiones con mayor puntuación.
    """
    if df_pred_altos.empty:
        print("No se encontraron regiones de alta probabilidad.")
        return
    
    fig, axes = plt.subplots(nrows=min(3, len(df_pred_altos)), ncols=1, figsize=(16, 4 * min(3, len(df_pred_altos))))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for i, (idx, fila) in enumerate(df_pred_altos.head(3).iterrows()):
        sec = fila['secuencia']
        one_hot = codificar_secuencia(sec, longitud_objetivo=200)
        tensor = torch.from_numpy(one_hot).float().unsqueeze(0)
        
        saliency = calcular_saliency_maps(modelo, tensor)[0] # (L, 4)
        perfil = saliency.sum(dim=-1).numpy() # (L,)
        
        ax = axes[i]
        ax.plot(perfil, color='purple')
        ax.fill_between(range(200), perfil, color='purple', alpha=0.3)
        ax.set_title(f"Saliency Map - Posición {fila['posicion']}")
        ax.set_xlabel("Posición en la Ventana (nt)")
        ax.set_ylabel("Importancia del Gradiente")
        ax.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Sitio GT central')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(ruta_salida / "saliency_map_regiones.png", dpi=300)
    plt.close()
    print(f"✓ Visualización de Saliency Maps guardada en {ruta_salida}")


# Entry point del script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predecir sitios de splicing en genes de interés.")
    parser.add_argument("gene", type=str, help="Nombre del gen de interés (ej. BRCA1 o TP53).")
    parser.add_argument("--umbral", type=float, default=0.8, help="Umbral de probabilidad para filtrar candidatos.")
    parser.add_argument("--paso", type=int, default=10, help="Paso de la ventana deslizante (default: 10).")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo: 'cpu' o 'cuda'.")
    
    args = parser.parse_args()
    
    # Verificaciones de rutas
    if not ARCHIVO_BASE_DATOS.exists():
        print(f"Error: Base de datos GENCODE no encontrada en {ARCHIVO_BASE_DATOS}")
        sys.exit(1)
    if not RUTA_MODELO.exists():
        print(f"Error: Modelo no encontrado en {RUTA_MODELO}. Entrene el modelo primero.")
        sys.exit(1)
    
    base_db = gffutils.FeatureDB(str(ARCHIVO_BASE_DATOS))
    genoma = Fasta(str(ARCHIVO_GENOMA_FASTA))
    
    print(f"Buscando secuencia para el gen: {args.gene}...")
    secuencia_completa, start_global, chrom, strand, gene_id = obtener_secuencia_gen(args.gene, base_db, genoma)
    
    # Cargar modelo
    modelo = RedNeuronalSplicing()
    modelo.load_state_dict(torch.load(RUTA_MODELO, map_location=args.device))
    modelo.to(args.device)
    
    print(f"Corriendo predicciones con ventana de 200pb y paso de {args.paso}...")
    df_pred = predecir_gen(modelo, secuencia_completa, dispositivo=args.device, paso=args.paso)
    
    # Filtrar candidatos de alta probabilidad
    candidatos = df_pred[df_pred['probabilidad'] > args.umbral]
    print(f"Se encontraron {len(candidatos)} candidatos con probabilidad > {args.umbral}")
    
    # Guardar resultados
    salida_dir = RUTA_FIGURAS / "genes_interes"
    salida_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV con resultados
    candidatos.to_csv(salida_dir / f"candidatos_{args.gene}.csv", index=False)
    df_pred.to_csv(salida_dir / f"predicciones_completas_{args.gene}.csv", index=False)
    
    # Visualizaciones
    visualizar_predicciones(df_pred, args.gene, salida_dir)
    analizar_saliency_regiones(modelo, candidatos, salida_dir)
    
    print(f"\n✓ Análisis del gen {args.gene} completado.")
    print(f"  - Candidatos: {salida_dir / f'candidatos_{args.gene}.csv'}")
    print(f"  - Visualizaciones: {salida_dir}")
