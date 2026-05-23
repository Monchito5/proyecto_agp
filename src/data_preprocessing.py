import gffutils
from pyfaidx import Fasta
import pandas as pd
from pathlib import Path
import random

def obtener_sitios_splicing(db, limit=None):
    """
    Extrae coordenadas de sitios donantes (GT) reales de la base de datos.
    Deduplica por coordenadas para evitar sesgos por múltiples transcritos.
    """
    datos = []
    genes = db.features_of_type('gene')
    
    print("Extrayendo sitios reales...")
    for i, gene in enumerate(genes):
        if limit and i >= limit: break
        
        for transcript in db.children(gene, featuretype='transcript'):
            exones = list(db.children(transcript, featuretype='exon', order_by='start'))
            
            for j in range(len(exones) - 1):
                exon_actual = exones[j]
                # Sitio donante (5' splice site)
                if exon_actual.strand == '+':
                    pos = exon_actual.end
                else:
                    pos = exon_actual.start
                
                datos.append({
                    'chrom': exon_actual.chrom,
                    'pos': pos,
                    'strand': exon_actual.strand,
                    'label': 1
                })
    
    df = pd.DataFrame(datos).drop_duplicates(subset=['chrom', 'pos', 'strand'])
    return df

def generar_decoys_simulados(n_decoys, sequence_length=200):
    """
    Genera secuencias decoy sintéticas con composición realista.
    """
    print(f"Modo simulación: Generando {n_decoys} secuencias decoy sintéticas...")
    decoys = []
    nucleotides = ['A', 'C', 'G', 'T']
    
    while len(decoys) < n_decoys:
        # Generar secuencia con composición ~50% GC
        secuencia = []
        for _ in range(sequence_length):
            if random.random() < 0.5:
                secuencia.append(random.choice(['G', 'C']))
            else:
                secuencia.append(random.choice(['A', 'T']))
        
        # Insertar 'GT' en posición aleatoria (simulando sitio donante)
        pos_gt = random.randint(sequence_length // 4, 3 * sequence_length // 4)
        secuencia[pos_gt] = 'G'
        secuencia[pos_gt + 1] = 'T'
        
        ventana = ''.join(secuencia)
        if ventana not in decoys:
            decoys.append(ventana)
            
    return pd.DataFrame({
        'chrom': 'simulated',
        'pos': 0,
        'strand': '+',
        'label': 0,
        'sequence': decoys
    })

def obtener_sitios_senuelo(fasta_path, n_objetivo, df_reales, ventana=100):
    """
    Genera sitios 'señuelo' (negativos) buscando dinucleótidos GT aleatorios 
    que NO estén en el conjunto de sitios reales.
    """
    if not fasta_path.exists():
        print("Archivo FASTA no encontrado. Usando modo simulación para decoys.")
        return None

    try:
        genome = Fasta(str(fasta_path))
    except Exception as e:
        print(f"Error al leer FASTA: {e}. Usando modo simulación.")
        return None

    reales_set = set(zip(df_reales['chrom'], df_reales['pos'], df_reales['strand']))
    
    datos_neg = []
    chroms = [c for c in genome.keys() if '_' not in c and len(c) < 6] # Solo cromosomas principales
    
    print(f"Buscando {n_objetivo} sitios señuelo en el genoma...")
    intentos = 0
    while len(datos_neg) < n_objetivo and intentos < n_objetivo * 50:
        intentos += 1
        chrom = random.choice(chroms)
        chrom_len = len(genome[chrom])
        if chrom_len <= ventana * 2: continue
        
        pos = random.randint(ventana + 1, chrom_len - ventana - 1)
        
        # Verificar GT
        seq_check = genome[chrom][pos:pos+2].seq.upper()
        strand = random.choice(['+', '-'])
        
        if seq_check == 'GT':
            if (chrom, pos, strand) not in reales_set:
                datos_neg.append({
                    'chrom': chrom,
                    'pos': pos,
                    'strand': strand,
                    'label': 0
                })
    
    if len(datos_neg) < n_objetivo * 0.1:
        print("No se encontraron suficientes sitios en el genoma. Usando modo simulación.")
        return None
        
    return pd.DataFrame(datos_neg)

def extraer_secuencias_y_limpiar(df, fasta_path, ventana=100):
    """
    Extrae secuencias del genoma, elimina aquellas con 'N' y asegura unicidad.
    """
    if 'sequence' in df.columns: # Si ya tiene secuencias (ej. simuladas)
        return df.drop_duplicates(subset=['sequence'])

    try:
        genome = Fasta(str(fasta_path))
    except Exception:
        print("Error: FASTA requerido para extraer secuencias reales.")
        return pd.DataFrame()

    secuencias = []
    indices_validos = []
    
    print(f"Extrayendo secuencias para {len(df)} sitios...")
    for idx, row in df.iterrows():
        chrom, pos = row['chrom'], row['pos']
        try:
            # Ventana simétrica 200nt
            seq = genome[chrom][pos - ventana : pos + ventana].seq.upper()
            if 'N' not in seq and len(seq) == (ventana * 2):
                secuencias.append(seq)
                indices_validos.append(idx)
        except Exception:
            continue
            
    df_result = df.loc[indices_validos].copy()
    df_result['sequence'] = secuencias
    
    antes = len(df_result)
    df_result = df_result.drop_duplicates(subset=['sequence'])
    print(f"Limpieza: {antes} -> {len(df_result)} secuencias únicas.")
    
    return df_result
