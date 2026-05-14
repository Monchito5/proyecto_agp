import gffutils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data_preprocessing import obtener_sitios_splicing, extraer_secuencias_ventana

# Configuración de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

DB_FILE = Path("data/raw/gencode.v47.annotation.gtf.db")
OUTPUT_DIR = Path("data/export")

def get_summary_stats(db):
    """Obtiene conteos básicos de tipos de features."""
    print("Obteniendo estadísticas de features...")
    stats = {}
    for feature_type in db.featuretypes():
        count = db.execute(f'SELECT COUNT(*) FROM features WHERE featuretype = "{feature_type}"').fetchone()[0]
        stats[feature_type] = count
    return pd.Series(stats)

def plot_chromosome_distribution(db, output_path):
    """Grafica la distribución de genes por cromosoma usando gffutils."""
    print("Graficando distribución por cromosoma...")
    chrom_counts = {}
    for feature in db.features_of_type('gene'):
        chrom = feature.chrom
        chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
    
    df_chroms = pd.DataFrame(list(chrom_counts.items()), columns=['seqname', 'count'])
    # Ordenar cromosomas de forma natural
    order = sorted(df_chroms['seqname'].unique(), key=lambda x: (len(x), x))
    
    plt.figure()
    sns.barplot(data=df_chroms, x='seqname', y='count', order=order)
    plt.xticks(rotation=45)
    plt.title("Distribución de Genes por Cromosoma")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_feature_lengths(db, feature_type, output_path, limit=2000):
    """Grafica la distribución de longitudes de un tipo de feature."""
    print(f"Calculando longitudes para {feature_type}...")
    lengths = []
    for i, feature in enumerate(db.features_of_type(feature_type)):
        if i >= limit: break
        lengths.append(len(feature))
    
    plt.figure()
    sns.histplot(lengths, bins=50, kde=True)
    plt.title(f"Distribución de Longitudes de {feature_type.capitalize()}s (Muestra de {limit})")
    plt.xlabel("Longitud (pb)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_exons_per_transcript(db, output_path, limit=500):
    """Grafica cuántos exones tiene cada transcrito."""
    print("Calculando exones por transcrito...")
    exon_counts = []
    transcripts = db.features_of_type('transcript')
    for i, transcript in enumerate(transcripts):
        if i >= limit: break
        count = sum(1 for _ in db.children(transcript, featuretype='exon'))
        exon_counts.append(count)
    
    plt.figure()
    sns.histplot(exon_counts, bins=range(min(exon_counts), max(exon_counts) + 2), discrete=True)
    plt.title(f"Distribución de Exones por Transcrito (Muestra de {limit})")
    plt.xlabel("Número de Exones")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_splice_motifs(db, fasta_path, output_path, limit=500):
    """Analiza los motivos de los sitios de splicing (ej. GT para donantes)."""
    print("Analizando motivos de sitios de splicing...")
    df_sitios = obtener_sitios_splicing(db, limit=limit)
    
    if not fasta_path.exists():
        print("Aviso: No se encontró el archivo FASTA. Saltando análisis de motivos.")
        return

    # Extraemos una ventana pequeña para ver solo el sitio de unión (±2pb)
    df_seqs = extraer_secuencias_ventana(df_sitios, fasta_path, ventana=2)
    
    motivos = []
    for _, row in df_seqs.iterrows():
        seq = row['sequence']
        if len(seq) >= 4:
            # Los primeros 2nt del intrón están en la posición 2 y 3 (0-based)
            motivos.append(seq[2:4])
    
    motivo_counts = pd.Series(motivos).value_counts()
    print("\nMotivos de sitios donantes encontrados (Top 5):")
    print(motivo_counts.head())

    plt.figure()
    motivo_counts.head(10).plot(kind='bar', color='skyblue')
    plt.title(f"Top 10 Motivos en Sitios Donantes (Muestra de {limit} genes)")
    plt.xlabel("Dinucleótido (primeros 2nt del intrón)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DB_FILE.exists():
        print(f"Error: No se encontró la base de datos en {DB_FILE}")
        return

    db = gffutils.FeatureDB(str(DB_FILE))
    FASTA_PATH = Path("data/raw/GRCh38.primary_assembly.genome.fa")
    
    # 1. Estadísticas de resumen
    summary = get_summary_stats(db)
    print("\nResumen de Features:")
    print(summary)
    summary.to_csv(OUTPUT_DIR / "feature_counts.csv")

    # 2. Distribución cromosómica
    plot_chromosome_distribution(db, OUTPUT_DIR / "eda_cromosomas_updated.png")
    
    # 3. Longitudes de exones
    plot_feature_lengths(db, 'exon', OUTPUT_DIR / "longitud_exones.png")
    
    # 4. Exones por transcrito
    plot_exons_per_transcript(db, OUTPUT_DIR / "exones_por_transcrito.png")

    # 5. Análisis de motivos de splicing
    analyze_splice_motifs(db, FASTA_PATH, OUTPUT_DIR / "eda_splicing_updated.png")

    print(f"\nEDA completado exitosamente. Resultados en {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
