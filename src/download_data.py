"""
download_data.py
"""

import gzip
import urllib.request
from pathlib import Path
import gffutils

# --- Constantes Simbólicas ---
URL_GTF_GENCODE = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.annotation.gtf.gz"
URL_FASTA_GENCODE = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/GRCh38.primary_assembly.genome.fa.gz"

def descargar_recursos_gencode(directorio_salida: str = "data/raw"):
    """
    Descarga y descomprime la anotación GTF y el genoma de referencia de GENCODE.
    También crea la base de datos SQLite para consultas rápidas.
    
    Parámetros:
        directorio_salida (str): Carpeta donde se guardarán los archivos.
    """
    ruta_base = Path(directorio_salida)
    ruta_base.mkdir(parents=True, exist_ok=True)

    ruta_gtf_comprimido = ruta_base / "gencode.v47.annotation.gtf.gz"
    ruta_fasta_comprimido = ruta_base / "GRCh38.primary_assembly.genome.fa.gz"
    
    ruta_gtf_descomprimido = str(ruta_gtf_comprimido).replace('.gz', '')
    ruta_fasta_descomprimido = str(ruta_fasta_comprimido).replace('.gz', '')
    ruta_base_datos = ruta_gtf_descomprimido + ".db"
    
    print("Descargando anotación GTF...")
    urllib.request.urlretrieve(URL_GTF_GENCODE, ruta_gtf_comprimido)
    
    print("Descargando genoma de referencia...")
    urllib.request.urlretrieve(URL_FASTA_GENCODE, ruta_fasta_comprimido)

    print("Descomprimiendo archivos...")
    with gzip.open(ruta_gtf_comprimido, 'rb') as archivo_entrada:
        with open(ruta_gtf_descomprimido, 'wb') as archivo_salida:
            archivo_salida.write(archivo_entrada.read())
            
    with gzip.open(ruta_fasta_comprimido, 'rb') as archivo_entrada:
        with open(ruta_fasta_descomprimido, 'wb') as archivo_salida:
            archivo_salida.write(archivo_entrada.read())
    
    print("Construyendo base de datos GFFUtils (este paso requiere tiempo)...")
    try:
        gffutils.create_db(
            ruta_gtf_descomprimido, 
            dbfn=ruta_base_datos, 
            force=True, 
            keep_order=True, 
            merge_strategy='merge', 
            sort_attribute_values=True,
            disable_infer_genes=True,
            disable_infer_transcripts=True
        )
        print(f"✓ Base de datos generada exitosamente en: {ruta_base_datos}")
    except Exception as error_db:
        print(f"Error crítico al crear la base de datos: {error_db}")

if __name__ == "__main__":
    descargar_recursos_gencode()
