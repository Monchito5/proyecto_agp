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
    Extrae coordenadas de sitios donantes (GT) reales de la base de datos genómica.
    
    Parámetros:
        base_datos (FeatureDB): Conexión a la base de datos de GFFUtils.
        limite_genes (int): Cantidad máxima de genes a procesar.
        
    Retorna:
        pd.DataFrame: Tabla con columnas [chrom, pos, strand, label].
    """
    lista_sitios = []
    coleccion_genes = base_datos.features_of_type('gene')
    
    print("Extrayendo sitios reales del genoma...")
    for i, objeto_gene in enumerate(coleccion_genes):
        if limite_genes and i >= limite_genes:
            break
        
        for objeto_transcrito in base_datos.children(objeto_gene, featuretype='transcript'):
            lista_exones = list(base_datos.children(objeto_transcrito, featuretype='exon', order_by='start'))
            
            for j in range(len(lista_exones) - 1):
                objeto_exon = lista_exones[j]
                posicion_donante = objeto_exon.end if objeto_exon.strand == '+' else objeto_exon.start
                
                lista_sitios.append({
                    'chrom': objeto_exon.chrom,
                    'pos': posicion_donante,
                    'strand': objeto_exon.strand,
                    'label': 1
                })
    
    tabla_datos = pd.DataFrame(lista_sitios)
    return tabla_datos.drop_duplicates(subset=['chrom', 'pos', 'strand'])

def generar_señuelos_simulados(cantidad_objetivo: int, longitud_total: int = 200) -> pd.DataFrame:
    """
    Genera secuencias sintéticas que imitan la composición del ADN.
    
    Parámetros:
        cantidad_objetivo (int): Número de secuencias a generar.
        longitud_total (int): Tamaño de cada secuencia en nucleótidos.
        
    Retorna:
        pd.DataFrame: Tabla con secuencias marcadas como clase 0.
    """
    print(f"Modo simulación: Generando {cantidad_objetivo} secuencias sintéticas...")
    lista_decoys = []
    base_nucleotidos = ['A', 'C', 'G', 'T']
    
    while len(lista_decoys) < cantidad_objetivo:
        secuencia_lista = []
        for _ in range(longitud_total):
            if random.random() < 0.5:
                secuencia_lista.append(random.choice(['G', 'C']))
            else:
                secuencia_lista.append(random.choice(['A', 'T']))
        
        posicion_gt = random.randint(longitud_total // 4, 3 * longitud_total // 4)
        secuencia_lista[posicion_gt] = 'G'
        secuencia_lista[posicion_gt + 1] = 'T'
        
        cadena_final = ''.join(secuencia_lista)
        if cadena_final not in lista_decoys:
            lista_decoys.append(cadena_final)
            
    return pd.DataFrame({
        'chrom': 'simulado', 'pos': 0, 'strand': '+', 'label': 0, 'sequence': lista_decoys
    })

def obtener_sitios_señuelo(ruta_fasta: Path, cantidad: int, tabla_reales: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Busca sitios GT en el genoma real que no están anotados como funcionales.
    """
    if not ruta_fasta.exists():
        return None

    try:
        objeto_genoma = Fasta(str(ruta_fasta))
    except Exception as error_lectura:
        print(f"Error al leer genoma: {error_lectura}")
        return None

    set_reales = set(zip(tabla_reales['chrom'], tabla_reales['pos'], tabla_reales['strand']))
    lista_negativos = []
    nombres_cromosomas = [c for c in objeto_genoma.keys() if '_' not in c and len(c) < 6]
    
    intentos_realizados = 0
    while len(lista_negativos) < cantidad and intentos_realizados < cantidad * 50:
        intentos_realizados += 1
        nombre_chrom = random.choice(nombres_cromosomas)
        longitud_chrom = len(objeto_genoma[nombre_chrom])
        if longitud_chrom <= 200: continue
        
        pos_azar = random.randint(101, longitud_chrom - 101)
        dinucleotido = objeto_genoma[nombre_chrom][pos_azar : pos_azar + 2].seq.upper()
        sentido_hebra = random.choice(['+', '-'])
        
        if dinucleotido == 'GT':
            if (nombre_chrom, pos_azar, sentido_hebra) not in set_reales:
                lista_negativos.append({
                    'chrom': nombre_chrom, 'pos': pos_azar, 'strand': sentido_hebra, 'label': 0
                })
    
    return pd.DataFrame(lista_negativos) if lista_negativos else None

def extraer_y_limpiar_secuencias(tabla_entrada: pd.DataFrame, ruta_fasta: Path) -> pd.DataFrame:
    """
    Extrae secuencias del genoma, filtra 'N' y elimina duplicados.
    """
    if 'sequence' in tabla_entrada.columns:
        return tabla_entrada.drop_duplicates(subset=['sequence'])

    try:
        objeto_genoma = Fasta(str(ruta_fasta))
    except Exception:
        return pd.DataFrame()

    datos_finales = []
    print(f"Procesando {len(tabla_entrada)} candidatos...")
    
    for _, fila in tabla_entrada.iterrows():
        try:
            secuencia = objeto_genoma[fila['chrom']][fila['pos'] - 100 : fila['pos'] + 100].seq.upper()
            if 'N' not in secuencia and len(secuencia) == 200:
                nueva_fila = fila.to_dict()
                nueva_fila['sequence'] = secuencia
                datos_finales.append(nueva_fila)
        except Exception:
            continue
            
    tabla_resultado = pd.DataFrame(datos_finales)
    if not tabla_resultado.empty:
        total_previo = len(tabla_resultado)
        # Mantener metadatos en la deduplicación
        tabla_resultado = tabla_resultado.drop_duplicates(subset=['sequence'])
        print(f"Limpieza completada: {total_previo} -> {len(tabla_resultado)} secuencias únicas.")
    
    return tabla_resultado
