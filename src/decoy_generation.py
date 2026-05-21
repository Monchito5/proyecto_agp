import os
import sys
import random
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

# Intentar importar Biopython para manejo de FASTA
try:
    from Bio import SeqIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("[ADVERTENCIA] Biopython no está instalado. Usando modo simulación.")


# Configuración de rutas
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_EXPORT_DIR = PROJECT_ROOT / "data" / "export"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Archivos
VERDADEROS_FILE = DATA_EXPORT_DIR / "dataset_verdaderos_unicos.csv"
DECOY_OUTPUT_FILE = DATA_EXPORT_DIR / "dataset_consolidado_balanceado.csv"

# Genoma de referencia (ajustar según disponibilidad)
FASTA_FILE = DATA_RAW_DIR / "GRCh38.primary_assembly.genome.fa"

# Parámetros
SEQUENCE_LENGTH = 200  # Longitud de ventana en nucleótidos
MIN_DISTANCE_FROM_ANNOTATED = 1000  # Distancia mínima (bp) de sitios anotados


def cargar_secuencias_verdaderas(ruta_archivo: Path) -> pd.DataFrame:
   
    if not ruta_archivo.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {ruta_archivo}\n"
            f"Asegúrate de ejecutar primero el Módulo 1."
        )
    
    df = pd.read_csv(ruta_archivo)
    
    if df.empty:
        raise ValueError("El archivo de secuencias verdaderas está vacío")
    
    print(f"[VERDADEROS] Cargadas {len(df)} secuencias únicas")
    
    return df


def extraer_ventana_secuencia(secuencia_completa: str, pos_centro: int, 
                               longitud: int = 200) -> str:
    
    mitad = longitud // 2
    inicio = max(0, pos_centro - mitad)
    fin = inicio + longitud
    
    # Ajustar si estamos cerca del inicio
    if fin > len(secuencia_completa):
        fin = len(secuencia_completa)
        inicio = max(0, fin - longitud)
    
    ventana = secuencia_completa[inicio:fin]
    
    # Padding si la ventana es más corta que la longitud deseada
    if len(ventana) < longitud:
        ventana = ventana + 'N' * (longitud - len(ventana))
    
    return ventana


def buscar_dinucleotido_gt(secuencia: str) -> List[int]:
    
    posiciones = []
    pos = 0
    while True:
        pos = secuencia.find('GT', pos)
        if pos == -1:
            break
        posiciones.append(pos)
        pos += 1  # Buscar siguiente posición
    
    return posiciones


def generar_decoys_del_genoma(
    fasta_path: Path,
    n_decoys: int,
    sequence_length: int = 200,
    verdaderas_sequences: Optional[List[str]] = None
) -> List[str]:
    
    if not fasta_path.exists():
        raise FileNotFoundError(
            f"Archivo FASTA no encontrado: {fasta_path}\n"
            f"Descarga el genoma hg38 y colócalo en: {fasta_path}\n"
            f"Ejemplo: wget ftp://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/GRCh38.primary_assembly.dna.toplevel.fa.gz"
        )
    
    decoys_encontrados = []
    random.seed(42)
    
    print(f"[DECOYS] Cargando genoma desde: {fasta_path}")
    
    if BIOPYTHON_AVAILABLE:
        # Modo real: usar Biopython para leer FASTA
        print("[DECOYS] Usando Biopython para extraer secuencias...")
        
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            cromosoma = record.id
            secuencia = str(record.seq).upper()
            
            print(f"  Procesando {cromosoma} ({len(secuencia)} bp)...")
            
            # Buscar todos los 'GT' en este cromosoma
            posiciones_gt = buscar_dinucleotido_gt(secuencia)
            print(f"    Encontrados {len(posiciones_gt)} sitios 'GT'")
            
            # Extraer ventanas alrededor de cada 'GT'
            for pos in posiciones_gt:
                ventana = extraer_ventana_secuencia(secuencia, pos, sequence_length)
                
                # Filtrar: sin 'N' y no duplicada
                if 'N' not in ventana:
                    if verdaderas_sequences is None or ventana not in verdaderas_sequences:
                        decoys_encontrados.append(ventana)
                        
                        # Limitar para no consumir memoria
                        if len(decoys_encontrados) >= n_decoys * 2:
                            break
            
            if len(decoys_encontrados) >= n_decoys * 2:
                break
        
        # Muestrear aleatoriamente para obtener exactamente n_decoys
        if len(decoys_encontrados) > n_decoys:
            decoys_encontrados = random.sample(decoys_encontrados, n_decoys)
        
    else:
        # Modo simulación: generar secuencias decoy sintéticas realistas
        print("[DECOYS] Modo simulación: generando secuencias decoy sintéticas...")
        print("[DECOYS] Para modo real, instala Biopython: pip install biopython")
        
        nucleotides = ['A', 'C', 'G', 'T']
        target_gt = n_decoys
        
        while len(decoys_encontrados) < target_gt:
            # Generar secuencia aleatoria
            secuencia = [random.choice(nucleotides) for _ in range(sequence_length)]
            
            # Insertar 'GT' en posición aleatoria (evitando bordes extremos)
            pos_gt = random.randint(50, sequence_length - 52)
            secuencia[pos_gt] = 'G'
            secuencia[pos_gt + 1] = 'T'
            
            ventana = ''.join(secuencia)
            
            # Evitar duplicados y secuencias verdaderas
            if ventana not in decoys_encontrados:
                if verdaderas_sequences is None or ventana not in verdaderas_sequences:
                    decoys_encontrados.append(ventana)
    
    print(f"[DECOYS] Generadas {len(decoys_encontrados)} secuencias decoy únicas")
    
    return decoys_encontrados


def generar_decoys_simulados(n_decoys: int, sequence_length: int = 200,
                              verdaderas_sequences: Optional[List[str]] = None) -> List[str]:
    
    print("[DECOYS] Generando secuencias decoy simuladas...")
    print("[DECOYS] Para usar genoma real, coloca FASTA en: data/raw/GRCh38.primary_assembly.genome.fa")
    
    decoys = []
    nucleotides = ['A', 'C', 'G', 'T']
    target_gt = n_decoys
    
    # Intentar generar suficientes secuencias únicas
    intentos = 0
    max_intentos = n_decoys * 10
    
    while len(decoys) < target_gt and intentos < max_intentos:
        intentos += 1
        
        # Generar secuencia con composición ~50% GC (realista para regiones codificantes)
        secuencia = []
        for _ in range(sequence_length):
            if random.random() < 0.5:
                secuencia.append(random.choice(['G', 'C']))
            else:
                secuencia.append(random.choice(['A', 'T']))
        
        # Insertar 'GT' en posición aleatoria (evitando bordes)
        pos_gt = random.randint(50, sequence_length - 52)
        secuencia[pos_gt] = 'G'
        secuencia[pos_gt + 1] = 'T'
        
        ventana = ''.join(secuencia)
        
        # Verificar unicidad
        if ventana not in decoys:
            if verdaderas_sequences is None or ventana not in verdaderas_sequences:
                decoys.append(ventana)
    
    print(f"[DECOYS] Generadas {len(decoys)} secuencias decoy únicas")
    
    return decoys


def consolidar_dataset(
    df_verdaderos: pd.DataFrame,
    decoy_sequences: List[str]
) -> pd.DataFrame:
    
    n_verdaderos = len(df_verdaderos)
    n_decoys = len(decoy_sequences)
    
    # Asegurar balance perfecto
    if n_decoys != n_verdaderos:
        print(f"[BALANCE] Ajustando número de decoys...")
        if n_decoys > n_verdaderos:
            # Muestrear aleatoriamente
            decoy_sequences = random.sample(decoy_sequences, n_verdaderos)
        else:
            # Generar más decoys
            adicionales = n_verdaderos - n_decoys
            print(f"[BALANCE] Generando {adicionales} decoys adicionales...")
            adicionales_decoys = generar_decoys_simulados(adicionales)
            decoy_sequences.extend(adicionales_decoys)
    
    # Crear DataFrames
    df_verdaderos_labeled = df_verdaderos.copy()
    df_verdaderos_labeled['label'] = 1
    
    df_decoys = pd.DataFrame({
        'sequence': decoy_sequences,
        'label': 0
    })
    
    # Combinar
    df_consolidado = pd.concat([df_verdaderos_labeled, df_decoys], ignore_index=True)
    
    # Mezclar aleatoriamente
    df_consolidado = df_consolidado.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"[CONSOLIDADO] Verdaderos (label=1): {len(df_verdaderos_labeled)}")
    print(f"[CONSOLIDADO] Decoys (label=0): {len(df_decoys)}")
    print(f"[CONSOLIDADO] Total: {len(df_consolidado)}")
    
    return df_consolidado


def guardar_dataset_consolidado(df: pd.DataFrame, ruta_salida: Path) -> None:
    
    try:
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta_salida, index=False)
        print(f"[EXPORTACIÓN] Dataset guardado en: {ruta_salida}")
        print(f"[EXPORTACIÓN] Formato: {df.shape[0]} filas x {df.shape[1]} columnas")
        print(f"[EXPORTACIÓN] Columnas: {list(df.columns)}")
    except OSError as e:
        raise OSError(f"Error al guardar: {e}")


def main():
    
    print("=" * 70)
    print("MÓDULO 2: GENERACIÓN DE DECOYS Y CONSOLIDACIÓN DE DATASET")
    print("=" * 70)
    print(f"Directorio de trabajo: {PROJECT_ROOT}")
    print(f"Archivo verdaderos: {VERDADEROS_FILE}")
    print(f"Archivo FASTA (opcional): {FASTA_FILE}")
    print(f"Archivo salida: {DECOY_OUTPUT_FILE}")
    print("=" * 70)
    
    try:
        # Paso 1: Cargar secuencias verdaderas
        print("\n[1] CARGANDO SECUENCIAS VERDADERAS...")
        df_verdaderos = cargar_secuencias_verdaderas(VERDADEROS_FILE)
        n_verdaderos = len(df_verdaderos)
        print(f"    ✓ {n_verdaderos} secuencias verdaderas únicas")
        
        # Paso 2: Generar secuencias decoy
        print("\n[2] GENERANDO SECUENCIAS DECOY...")
        
        # Obtener lista de secuencias verdaderas para evitar duplicados
        verdaderas_list = df_verdaderos['sequence'].tolist()
        
        # Intentar usar genoma real si está disponible
        if FASTA_FILE.exists() and BIOPYTHON_AVAILABLE:
            decoys = generar_decoys_del_genoma(
                FASTA_FILE,
                n_verdaderos,
                SEQUENCE_LENGTH,
                verdaderas_list
            )
        else:
            # Usar modo simulación
            decoys = generar_decoys_simulados(
                n_verdaderos,
                SEQUENCE_LENGTH,
                verdaderas_list
            )
        
        # Paso 3: Consolidar dataset
        print("\n[3] CONSOLIDANDO DATASET BALANCEADO...")
        df_consolidado = consolidar_dataset(df_verdaderos, decoys)
        
        # Paso 4: Guardar resultado
        print("\n[4] EXPORTANDO DATASET FINAL...")
        guardar_dataset_consolidado(df_consolidado, DECOY_OUTPUT_FILE)
        
        # Estadísticas finales
        print("\n" + "=" * 70)
        print("RESUMEN FINAL")
        print("=" * 70)
        print(f"✓ Secuencias verdaderas (label=1): {len(df_consolidado[df_consolidado['label'] == 1])}")
        print(f"✓ Secuencias decoy (label=0): {len(df_consolidado[df_consolidado['label'] == 0])}")
        print(f"✓ Balance: 50/50")
        print(f"✓ Longitud secuencia: {SEQUENCE_LENGTH} nt")
        print(f"✓ Archivo: {DECOY_OUTPUT_FILE}")
        print("=" * 70)
        print("\n[OK] Pipeline completado exitosamente.")
        print("\nSiguiente paso: Ejecutar Módulo 3 para preparar datos para la CNN.")
        
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