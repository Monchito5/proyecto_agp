import gffutils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from data_preprocessing import obtener_sitios_splicing, extraer_secuencias_y_limpiar

# Configuración de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

DB_FILE = Path("data/raw/gencode.v47.annotation.gtf.db")
OUTPUT_DIR = Path("data/export")

def save_summary_table(df, filename):
    """Guarda un resumen de texto del DataFrame."""
    with open(OUTPUT_DIR / filename, "w") as f:
        f.write(df.to_string())

def perform_general_analysis(df):
    """Realiza el análisis general del dataset con las visualizaciones solicitadas."""
    print("\n--- Iniciando Análisis General ---")
    
    # 1. Dimensiones y composición de features
    print("Analizando dimensiones...")
    
    # Clasificar columnas por tipo
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    seq_cols = [c for c in cat_cols if 'sequence' in c.lower()]
    
    feat_types = pd.Series({
        'Numéricas': len(num_cols),
        'Categóricas': len(cat_cols) - len(seq_cols),
        'Secuencias': len(seq_cols)
    })
    
    plt.figure()
    feat_types.plot(kind='barh', color='skyblue')
    plt.title("Composición del Dataset por Tipo de Feature")
    plt.xlabel("Cantidad de Columnas")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gen_01_composicion_features.png")
    plt.close()

    # 2. Tipos de datos (Heatmap)
    print("Analizando tipos de datos...")
    unique_types = [str(t) for t in df.dtypes.unique()]
    type_matrix = pd.DataFrame(0, index=df.columns, columns=unique_types)
    for col in df.columns:
        type_matrix.loc[col, str(df[col].dtype)] = 1
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(type_matrix, annot=True, cbar=False, cmap="YlGnBu")
    plt.title("Mapa de Calor de Tipos de Datos")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gen_02_tipos_datos_heatmap.png")
    plt.close()

    # 3. Valores faltantes
    print("Analizando valores faltantes...")
    null_pct = (df.isnull().sum() / len(df)) * 100
    if null_pct.sum() > 0:
        plt.figure()
        null_pct.plot(kind='bar', color='salmon')
        plt.title("Porcentaje de Valores Nulos por Columna")
        plt.ylabel("% de Nulos")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "gen_03_nulos_bar.png")
        plt.close()
    else:
        plt.figure()
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title("Mapa de Completitud (No se detectaron nulos)")
        plt.savefig(OUTPUT_DIR / "gen_03_completitud_heatmap.png")
        plt.close()

    # 4. Estadísticas descriptivas (Numéricas)
    if num_cols:
        print("Generando estadísticas numéricas...")
        col_to_plot = 'pos' if 'pos' in num_cols else num_cols[0]
        fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        sns.boxplot(data=df, x=col_to_plot, ax=ax_box, color='lightgreen')
        sns.histplot(data=df, x=col_to_plot, ax=ax_hist, kde=True, color='teal')
        ax_box.set(yticks=[], title=f"Distribución de {col_to_plot}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"gen_04_dist_{col_to_plot}.png")
        plt.close()

    # 5. Composición de bases por posición
    if seq_cols:
        print("Analizando composición por posición...")
        seq_col = seq_cols[0]
        sample_size = min(5000, len(df))
        seqs = df[seq_col].iloc[:sample_size].apply(list)
        pos_df = pd.DataFrame(seqs.tolist())
        
        comp_pos = pos_df.apply(lambda x: pd.Series(x).value_counts(normalize=True)).fillna(0).T
        for nt in 'ACGT':
            if nt not in comp_pos.columns: comp_pos[nt] = 0
        comp_pos = comp_pos[['A', 'C', 'G', 'T']]

        comp_pos.plot(kind='bar', stacked=True, width=1.0, figsize=(15, 6), color=['#619CFF', '#00BA38', '#F8766D', '#B79F00'])
        plt.title(f"Perfil de Composición de Bases por Posición ({seq_col})")
        plt.xlabel("Posición en la secuencia")
        plt.ylabel("Frecuencia")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(np.arange(0, len(comp_pos), 20), np.arange(0, len(comp_pos), 20))
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "gen_05_composicion_posicional.png")
        plt.close()

def perform_specific_analysis(df, db):
    """Realiza el análisis específico del dataset genómico."""
    print("\n--- Iniciando Análisis Específico ---")
    
    # 1. Balance de clases
    print("Analizando balance de clases...")
    plt.figure(figsize=(7, 7))
    class_counts = df['label'].value_counts()
    labels = [f"Clase {c} (N={n})" for c, n in class_counts.items()]
    plt.pie(class_counts, labels=labels, autopct='%1.1f%%', colors=['#66b3ff','#99ff99'])
    plt.title("Balance de Clases (1: Verdadero, 0: Señuelo)")
    plt.savefig(OUTPUT_DIR / "spec_01_balance_clases.png")
    plt.close()

    # 2. Distribución por cromosoma
    print("Analizando distribución cromosómica...")
    chrom_counts = df['chrom'].value_counts().head(10)
    plt.figure()
    chrom_counts.plot(kind='barh', color='plum')
    plt.title("Top 10 Cromosomas en el Dataset")
    plt.xlabel("Cantidad de Sitios")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spec_02_dist_cromosomas.png")
    plt.close()

    # 3. Composición nucleotídica global
    print("Analizando composición global...")
    all_seqs = "".join(df['sequence'].iloc[:min(10000, len(df))].tolist())
    freqs = {nt: all_seqs.count(nt) for nt in 'ACGT'}
    plt.figure(figsize=(7, 7))
    plt.pie(freqs.values(), labels=freqs.keys(), autopct='%1.1f%%', colors=sns.color_palette('viridis', 4))
    plt.title("Composición Nucleotídica Global")
    plt.savefig(OUTPUT_DIR / "spec_03_composicion_global.png")
    plt.close()

    # 4. Contenido GC por clase
    print("Analizando Contenido GC por clase...")
    def calc_gc(seq):
        return (seq.count('G') + seq.count('C')) / len(seq) if seq else 0
    df['gc_content'] = df['sequence'].apply(calc_gc)
    
    plt.figure()
    sns.violinplot(data=df, x='label', y='gc_content', palette='Set2')
    plt.title("Distribución de Contenido GC por Clase")
    plt.ylabel("Fracción GC")
    plt.savefig(OUTPUT_DIR / "spec_04_gc_content_violin.png")
    plt.close()

    # 5. Correlación
    print("Analizando correlaciones...")
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 1:
        plt.figure()
        sns.heatmap(num_df.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Mapa de Calor de Correlación (Spearman)")
        plt.savefig(OUTPUT_DIR / "spec_05_correlacion_heatmap.png")
        plt.close()

    # 6. Longitud de exones (DB)
    print("Analizando longitudes desde la base de datos...")
    exon_lengths = [len(f) for i, f in enumerate(db.features_of_type('exon')) if i < 5000]
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    sns.boxplot(x=exon_lengths, ax=ax_box, color='gold')
    sns.histplot(x=exon_lengths, ax=ax_hist, kde=True, color='orange', bins=50)
    ax_box.set(yticks=[], title="Distribución de Longitudes de Exones (Muestra DB)")
    plt.xlabel("Longitud (pb)")
    plt.xlim(0, 1000)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spec_06_longitud_exones.png")
    plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DATASET_CSV = OUTPUT_DIR / "dataset_consolidado_balanceado.csv"

    if not DATASET_CSV.exists():
        print(f"Error: No se encontró el dataset en {DATASET_CSV}")
        return

    df = pd.read_csv(DATASET_CSV)
    
    if not DB_FILE.exists():
        print(f"Error: No se encontró la DB en {DB_FILE}")
        return
    db = gffutils.FeatureDB(str(DB_FILE))

    perform_general_analysis(df)
    perform_specific_analysis(df, db)
    print(f"\nEDA y Exportación completados. Ver archivos en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
