"""
download_data.py
"""

import os
import gzip
import urllib.request
from pathlib import Path
import gffutils

# --- Constantes Simbólicas de Rutas y URLs ---
RAIZ = Path(__file__).parent.parent
DIR_RAW = RAIZ / "data" / "raw"
DIR_EXPORT = RAIZ / "data" / "export"

URL_GTF_GENCODE = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.annotation.gtf.gz"
URL_FASTA_GENCODE = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/GRCh38.primary_assembly.genome.fa.gz"

def descargar_recursos_gencode():
    """
    Descarga y descomprime la anotación GTF y el genoma de referencia.
    Organiza los archivos crudos en data/raw y la base de datos en data/export.
    """
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    DIR_EXPORT.mkdir(parents=True, exist_ok=True)

    ruta_gtf_gz = DIR_RAW / "gencode.v47.annotation.gtf.gz"
    ruta_fasta_gz = DIR_RAW / "GRCh38.primary_assembly.genome.fa.gz"
    
    gtf_descomp = str(ruta_gtf_gz).replace('.gz', '')
    fasta_descomp = str(ruta_fasta_gz).replace('.gz', '')
    
    # La base de datos se guarda en export como recurso externo procesado
    ruta_db = DIR_EXPORT / "gencode.v47.annotation.gtf.db"
    
    print("Descargando recursos de GENCODE (GTF y FASTA)...")
    if not ruta_gtf_gz.exists():
        urllib.request.urlretrieve(URL_GTF_GENCODE, ruta_gtf_gz)
    if not ruta_fasta_gz.exists():
        urllib.request.urlretrieve(URL_FASTA_GENCODE, ruta_fasta_gz)

    print("Descomprimiendo archivos en data/raw/...")
    for gz_path, out_path in [(ruta_gtf_gz, gtf_descomp), (ruta_fasta_gz, fasta_descomp)]:
        if not Path(out_path).exists():
            with gzip.open(gz_path, 'rb') as f_in:
                with open(out_path, 'wb') as f_out:
                    f_out.write(f_in.read())
    
    print("Construyendo Base de Datos GFFUtils en data/export/...")
    if not ruta_db.exists():
        try:
            gffutils.create_db(
                gtf_descomp, 
                dbfn=str(ruta_db), 
                force=True, 
                keep_order=True, 
                merge_strategy='merge', 
                sort_attribute_values=True,
                disable_infer_genes=True,
                disable_infer_transcripts=True
            )
            print(f"✓ Base de Datos generada: {ruta_db.name}")
        except Exception as e:
            print(f"Error al crear base de datos: {e}")
    else:
        print("✓ La base de datos ya existe.")

if __name__ == "__main__":
    descargar_recursos_gencode()
