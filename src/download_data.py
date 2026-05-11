import os
import gzip
import urllib.request
from pathlib import Path

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

    print(f"Archivos guardados en {output_dir}")

if __name__ == "__main__":
    download_gencode()