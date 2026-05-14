import sys
from pathlib import Path
from data_preprocessing import obtener_sitios_splicing, extraer_secuencias_ventana
import gffutils


DB_FILE = Path("data/raw/gencode.v47.annotation.gtf.db")
FASTA_FILE = Path("data/raw/GRCh38.primary_assembly.genome.fa")
EXPORT_DIR = Path("data/export")

def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    

    if not DB_FILE.exists():
        print(f"Error: No existe {DB_FILE}")
        return
    db = gffutils.FeatureDB(str(DB_FILE))
    print("✓ Conexión a DB exitosa.")

    # Obtener coordenadas
    print("Extrayendo sitios de splicing...")
    df_sitios = obtener_sitios_splicing(db, limit=1000) # Muestra pequeña para probar
    print(f"Sitios únicos encontrados: {len(df_sitios)}")

    # Integrar con Genoma
    if FASTA_FILE.exists():
        print("Integrando con genoma FASTA para extraer secuencias de 200nt...")
        df_final = extraer_secuencias_ventana(df_sitios, FASTA_FILE)
        
        # Guardar resultado para el modelo
        df_final.to_csv(EXPORT_DIR / "dataset_splicing_200nt.csv", index=False)
        print(f"Dataset listo en {EXPORT_DIR}/dataset_splicing_200nt.csv")
    else:
        print("Archivo FASTA no encontrado. Solo se extrajeron coordenadas.")

if __name__ == "__main__":
    main()