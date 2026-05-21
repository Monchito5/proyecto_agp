import os
import sys
import pandas as pd
from pathlib import Path
from typing import Tuple


# Configuración de rutas usando pathlib (multi-plataforma)
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_EXPORT_DIR = PROJECT_ROOT / "data" / "export"

# Archivos de entrada y salida
INPUT_FILE = DATA_EXPORT_DIR / "dataset_splicing_200nt.csv"
OUTPUT_FILE = DATA_EXPORT_DIR / "dataset_verdaderos_unicos.csv"


def cargar_dataset(ruta_archivo: Path) -> pd.DataFrame:

    if not ruta_archivo.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {ruta_archivo}\n"
            f"Verifica que el archivo 'dataset_splicing_200nt.csv' exista en {DATA_EXPORT_DIR}"
        )
    
    df = pd.read_csv(ruta_archivo)
    
    if df.empty:
        raise ValueError("El archivo CSV está vacío")
    
    if 'sequence' not in df.columns:
        raise ValueError(
            f"El archivo debe contener la columna 'sequence'. "
            f"Columnas encontradas: {list(df.columns)}"
        )
    
    return df


def filtrar_secuencias_validas(df: pd.DataFrame) -> pd.DataFrame:
    
    inicial = len(df)
    
    # Filtrar: mantener solo secuencias SIN 'N'
    df_limpio = df[~df['sequence'].str.contains('N', na=False)].reset_index(drop=True)
    
    eliminadas = inicial - len(df_limpio)
    
    print(f"[FILTRADO] Secuencias iniciales: {inicial}")
    print(f"[FILTRADO] Secuencias eliminadas (contienen 'N'): {eliminadas}")
    print(f"[FILTRADO] Secuencias válidas: {len(df_limpio)}")
    
    return df_limpio


def deduplicar_secuencias(df: pd.DataFrame) -> pd.DataFrame:
    
    before = len(df)
    
    # Deduplicación estricta por secuencia
    df_unico = df.drop_duplicates(subset=['sequence'], keep='first').reset_index(drop=True)
    
    eliminados = before - len(df_unico)
    
    print(f"[DEDUPLICACIÓN] Secuencias antes: {before}")
    print(f"[DEDUPLICACIÓN] Duplicados eliminados: {eliminados}")
    print(f"[DEDUPLICACIÓN] Secuencias únicas: {len(df_unico)}")
    
    return df_unico


def extraer_solo_secuencias(df: pd.DataFrame) -> pd.DataFrame:
    
    columnas_iniciales = list(df.columns)
    
    df_final = df[['sequence']].reset_index(drop=True)
    
    print(f"[METADATOS] Columnas descartadas: {columnas_iniciales}")
    print(f"[METADATOS] Columnas finales: {list(df_final.columns)}")
    
    return df_final


def guardar_dataset(df: pd.DataFrame, ruta_salida: Path) -> None:
    
    try:
        # Asegurar que el directorio existe
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(ruta_salida, index=False)
        
        print(f"[EXPORTACIÓN] Archivo guardado: {ruta_salida}")
        print(f"[EXPORTACIÓN] Tamaño: {df.shape[0]} filas x {df.shape[1]} columnas")
        
    except OSError as e:
        raise OSError(f"Error al guardar el archivo: {e}")


def main():
    
    print("=" * 70)
    print("MÓDULO 1: LIMPIEZA Y DEDUPLICACIÓN DE SECUENCIAS DE SPlicing")
    print("=" * 70)
    print(f"Directorio de trabajo: {PROJECT_ROOT}")
    print(f"Archivo de entrada: {INPUT_FILE}")
    print(f"Archivo de salida: {OUTPUT_FILE}")
    print("=" * 70)
    
    try:
        # Paso 1: Cargar dataset real
        print("\n[1] CARGANDO DATASET REAL...")
        df_raw = cargar_dataset(INPUT_FILE)
        print(f"    ✓ Dataset cargado: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas")
        print(f"    Columnas: {list(df_raw.columns)}")
        
        # Paso 2: Filtrar secuencias con 'N'
        print("\n[2] FILTRANDO SECUENCIAS INVÁLIDAS...")
        df_filtrado = filtrar_secuencias_validas(df_raw)
        
        # Paso 3: Deduplicar secuencias
        print("\n[3] DEDUPLICANDO SECUENCIAS...")
        df_dedup = deduplicar_secuencias(df_filtrado)
        
        # Paso 4: Extraer solo columnas de secuencia
        print("\n[4] DESCARTANDO METADATOS...")
        df_final = extraer_solo_secuencias(df_dedup)
        
        # Paso 5: Guardar resultado
        print("\n[5] EXPORTANDO DATASET LIMPIO...")
        guardar_dataset(df_final, OUTPUT_FILE)
        
        # Resumen final
        print("\n" + "=" * 70)
        print("RESUMEN FINAL")
        print("=" * 70)
        print(f"✓ Secuencias únicas limpias: {len(df_final)}")
        print(f"✓ Longitud de secuencia: {len(df_final['sequence'].iloc[0])} nt")
        print(f"✓ Archivo de salida: {OUTPUT_FILE}")
        print("=" * 70)
        print("\n[OK] Pipeline completado exitosamente.")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except OSError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()