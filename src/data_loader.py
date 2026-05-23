import sys
from pathlib import Path
from data_preprocessing import obtener_sitios_splicing, obtener_sitios_senuelo, generar_decoys_simulados, extraer_secuencias_y_limpiar
import gffutils
import pandas as pd

DB_FILE = Path("data/raw/gencode.v47.annotation.gtf.db")
FASTA_FILE = Path("data/raw/GRCh38.primary_assembly.genome.fa")
EXPORT_DIR = Path("data/export")

# Nombres de archivos compatibles con el resto del proyecto
RAW_OUTPUT = EXPORT_DIR / "dataset_splicing_200nt.csv"
FINAL_OUTPUT = EXPORT_DIR / "dataset_consolidado_balanceado.csv"

def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DB_FILE.exists():
        print(f"Error: No existe {DB_FILE}")
        return
    db = gffutils.FeatureDB(str(DB_FILE))
    print("✓ Conexión a DB exitosa.")

    # 1. Obtener sitios reales
    print("\n--- Paso 1: Extracción de Sitios Reales ---")
    df_reales = obtener_sitios_splicing(db, limit=500) # Límite para rapidez en pruebas
    print(f"Sitios reales (coordenadas únicas) encontrados: {len(df_reales)}")

    # 2. Extraer secuencias reales
    print("\n--- Paso 2: Extracción de Secuencias Reales ---")
    df_reales_seq = extraer_secuencias_y_limpiar(df_reales, FASTA_FILE)
    df_reales_seq.to_csv(RAW_OUTPUT, index=False)
    
    # 3. Obtener sitios señuelo (Balanceo)
    print("\n--- Paso 3: Generación de Sitios Señuelo ---")
    n_objetivo = len(df_reales_seq)
    df_negativos = obtener_sitios_senuelo(FASTA_FILE, n_objetivo, df_reales_seq)
    
    if df_negativos is not None:
        # Extraer secuencias para los negativos encontrados en genoma
        df_negativos = extraer_secuencias_y_limpiar(df_negativos, FASTA_FILE)
        
        # Si no alcanzamos el objetivo con el genoma, completar con simulación
        if len(df_negativos) < n_objetivo:
            print(f"Completando con {n_objetivo - len(df_negativos)} secuencias simuladas...")
            df_sim = generar_decoys_simulados(n_objetivo - len(df_negativos))
            df_negativos = pd.concat([df_negativos, df_sim], ignore_index=True)
    else:
        # Modo simulación total para negativos
        df_negativos = generar_decoys_simulados(n_objetivo)

    # 4. Consolidar dataset balanceado
    print("\n--- Paso 4: Consolidación Final ---")
    df_final = pd.concat([df_reales_seq, df_negativos], ignore_index=True)
    # Mezclar aleatoriamente
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_final.to_csv(FINAL_OUTPUT, index=False)
    print(f"\n✓ Dataset final guardado en {FINAL_OUTPUT}")
    print(f"Composición final: {df_final['label'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
