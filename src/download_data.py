import os
import gzip
import urllib.request
from pathlib import Path
import gffutils 

def download_gencode(version="47", release="47", output_dir="data/raw"):
    """Descarga anotación GTF y secuencias cDNA/ADN cromosómico de GENCODE"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Archivos más importantes:
    # - Anotación completa (GTF)
    gtf_url = f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{release}/gencode.v{release}.annotation.gtf.gz"
    # - Secuencias de todos los cromosomas (ADN)
    dna_url = f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{release}/GRCh38.primary_assembly.genome.fa.gz"

    gtf_out = os.path.join(output_dir, f"gencode.v{release}.annotation.gtf.gz")
    dna_out = os.path.join(output_dir, "GRCh38.primary_assembly.genome.fa.gz")
    db_out = gtf_out.replace('.gz', '.db')
    
    print("Descargando anotación GTF...")
    urllib.request.urlretrieve(gtf_url, gtf_out)
    print("Descargando genoma...")
    urllib.request.urlretrieve(dna_url, dna_out)

    # Descomprimir 
    with gzip.open(gtf_out, 'rb') as f_in:
        with open(gtf_out.replace('.gz', ''), 'wb') as f_out:
            f_out.write(f_in.read())
    with gzip.open(dna_out, 'rb') as f_in:
        with open(dna_out.replace('.gz', ''), 'wb') as f_out:
            f_out.write(f_in.read())
    

    print("Creando base de datos gffutils (esto puede tardar unos minutos)...")
    try:
        db = gffutils.create_db(
            gtf_out, 
            dbfn=db_out, 
            force=True, 
            keep_order=True, 
            merge_strategy='merge', 
            sort_attribute_values=True,
            disable_infer_genes=True, # Recomendado para GENCODE ya que trae los genes definidos
            disable_infer_transcripts=True
        )
        print(f"Base de datos creada exitosamente en: {db_out}")
    except Exception as e:
        print(f"Error al crear la base de datos: {e}")

    print(f"Archivos guardados en {output_dir}")

if __name__ == "__main__":
    download_gencode()