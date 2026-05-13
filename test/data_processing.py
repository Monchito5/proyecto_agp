import gffutils
from pyfaidx import Fasta
import pandas as pd
from pathlib import Path

def obtener_sitios_splicing(db, limit=None):
    """
    Extrae coordenadas de sitios donantes (GT) de la base de datos.
    """
    datos = []
    genes = db.features_of_type('gene')
    
    for i, gene in enumerate(genes):
        if limit and i >= limit: break
        
        # Obtenemos exones de los transcritos del gen
        for transcript in db.children(gene, featuretype='transcript'):
            exones = list(db.children(transcript, featuretype='exon', order_by='start'))
            
            # Un sitio donante existe al final de cada exón (excepto el último)
            for j in range(len(exones) - 1):
                exon_actual = exones[j]
                datos.append({
                    'chrom': exon_actual.chrom,
                    'donor_pos': exon_actual.end if exon_actual.strand == '+' else exon_actual.start,
                    'strand': exon_actual.strand,
                    'transcript_id': transcript.id
                })
    
    return pd.DataFrame(datos).drop_duplicates()

def extraer_secuencias_ventana(df, fasta_path, ventana=100):
    """
    Usa pyfaidx para obtener 100nt exón / 100nt intrón (Total 200nt).
    """
    genome = Fasta(str(fasta_path))
    secuencias = []
    
    for _, row in df.iterrows():
        chrom, pos = row['chrom'], row['donor_pos']
        seq = genome[chrom][pos - ventana : pos + ventana].seq
        secuencias.append(seq.upper())
        
    df['sequence'] = secuencias
    return df