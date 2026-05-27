"""
data_preprocessing.py
"""

import gffutils
from pyfaidx import Fasta
import pandas as pd
from pathlib import Path
import random
from typing import Optional

def obtener_sitios_reales(base_datos: gffutils.FeatureDB, limite_genes: Optional[int] = None) -> pd.DataFrame:
    """
    Extrae coordenadas de sitios donantes (GT) y aceptores (AG) reales.
    """
    lista_sitios = []
    coleccion_genes = base_datos.features_of_type('gene')
    
    print("Extrayendo sitios reales (Donantes y Aceptores)...")
    for i, objeto_gene in enumerate(coleccion_genes):
        if limite_genes and i >= limite_genes:
            break
        
        for objeto_transcrito in base_datos.children(objeto_gene, featuretype='transcript'):
            lista_exones = list(base_datos.children(objeto_transcrito, featuretype='exon', order_by='start'))
            
            for j in range(len(lista_exones)):
                objeto_exon = lista_exones[j]
                
                # 1. Sitio Donante (al final del exón, si no es el último)
                if j < len(lista_exones) - 1:
                    pos_don = objeto_exon.end if objeto_exon.strand == '+' else objeto_exon.start
                    lista_sitios.append({
                        'chrom': objeto_exon.chrom, 'pos': pos_don, 'strand': objeto_exon.strand,
                        'label': 1, 'tipo_sitio': 'donante'
                    })
                
                # 2. Sitio Aceptor (al inicio del exón, si no es el primero)
                if j > 0:
                    pos_acep = objeto_exon.start if objeto_exon.strand == '+' else objeto_exon.end
                    lista_sitios.append({
                        'chrom': objeto_exon.chrom, 'pos': pos_acep, 'strand': objeto_exon.strand,
                        'label': 1, 'tipo_sitio': 'aceptor'
                    })
    
    tabla_datos = pd.DataFrame(lista_sitios)
    return tabla_datos.drop_duplicates(subset=['chrom', 'pos', 'strand', 'tipo_sitio'])

def generar_señuelos_simulados(cantidad_objetivo: int, tipo: str = 'donante') -> pd.DataFrame:
    """Genera secuencias sintéticas imitando Donantes o Aceptores."""
    print(f"Modo simulación: Generando {cantidad_objetivo} señuelos para {tipo}...")
    lista_decoys = []
    motivo = 'GT' if tipo == 'donante' else 'AG'
    
    while len(lista_decoys) < cantidad_objetivo:
        sec_lista = [random.choice(['G', 'C']) if random.random() < 0.5 else random.choice(['A', 'T']) for _ in range(200)]
        pos_motivo = 100
        sec_lista[pos_motivo] = motivo[0]
        sec_lista[pos_motivo + 1] = motivo[1]
        
        cadena = ''.join(sec_lista)
        if cadena not in lista_decoys: lista_decoys.append(cadena)
            
    return pd.DataFrame({
        'chrom': 'simulado', 'pos': 0, 'strand': '+', 'label': 0, 'tipo_sitio': tipo, 'sequence': lista_decoys
    })

def obtener_sitios_señuelo(ruta_fasta: Path, cantidad: int, tabla_reales: pd.DataFrame, tipo: str = 'donante') -> Optional[pd.DataFrame]:
    """Busca dinucleótidos (GT/AG) en el genoma que no son sitios funcionales."""
    if not ruta_fasta.exists(): return None
    try: objeto_genoma = Fasta(str(ruta_fasta))
    except Exception: return None

    set_reales = set(zip(tabla_reales['chrom'], tabla_reales['pos'], tabla_reales['strand']))
    lista_negativos = []
    motivo_buscado = 'GT' if tipo == 'donante' else 'AG'
    nombres_chroms = [c for c in objeto_genoma.keys() if '_' not in c and len(c) < 6]
    
    intentos = 0
    while len(lista_negativos) < cantidad and intentos < cantidad * 50:
        intentos += 1
        chrom = random.choice(nombres_chroms)
        long_chrom = len(objeto_genoma[chrom])
        if long_chrom <= 200: continue
        
        pos = random.randint(101, long_chrom - 101)
        dinuc = objeto_genoma[chrom][pos : pos + 2].seq.upper()
        strand = random.choice(['+', '-'])
        
        if dinuc == motivo_buscado:
            if (chrom, pos, strand) not in set_reales:
                lista_negativos.append({
                    'chrom': chrom, 'pos': pos, 'strand': strand, 'label': 0, 'tipo_sitio': tipo
                })
    
    return pd.DataFrame(lista_negativos) if lista_negativos else None

def extraer_y_limpiar_secuencias(tabla_entrada: pd.DataFrame, ruta_fasta: Path) -> pd.DataFrame:
    """Extrae secuencias de 200pb, filtra 'N' y asegura unicidad."""
    if 'sequence' in tabla_entrada.columns:
        return tabla_entrada.drop_duplicates(subset=['sequence'])

    try: objeto_genoma = Fasta(str(ruta_fasta))
    except Exception: return pd.DataFrame()

    datos_finales = []
    print(f"Extrayendo secuencias para {len(tabla_entrada)} candidatos...")
    
    for _, fila in tabla_entrada.iterrows():
        try:
            # Ventana: -100 a +100 del sitio de unión
            secuencia = objeto_genoma[fila['chrom']][fila['pos'] - 100 : fila['pos'] + 100].seq.upper()
            if 'N' not in secuencia and len(secuencia) == 200:
                nueva_fila = fila.to_dict()
                nueva_fila['sequence'] = secuencia
                datos_finales.append(nueva_fila)
        except Exception: continue
            
    tabla_res = pd.DataFrame(datos_finales)
    if not tabla_res.empty:
        tabla_res = tabla_res.drop_duplicates(subset=['sequence'])
    return tabla_res
