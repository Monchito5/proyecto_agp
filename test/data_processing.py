from gtfparse import read_gtf

gtf_path = "data/raw/gencode.v47.annotation.gtf"
df = read_gtf(gtf_path)
# Filtrar exones y obtener sitios donantes/aceptores
exons = df[df['feature'] == 'exon']
print(f"Total exons: {len(exons)}")
# Ejemplo: extraer coordenadas de sitios donantes (final de exón)
donor_sites = exons[['seqname', 'start', 'end', 'strand']].copy()
# Para donante: posición end del exón (justo antes del intrón)
print(donor_sites.head())