import sys
from pathlib import Path
import pandas as pd
import gffutils
from sklearn.model_selection import train_test_split

# Importaciones locales
from data_preprocessing import obtener_sitios_splicing, obtener_sitios_senuelo, generar_decoys_simulados, extraer_secuencias_y_limpiar
from eda import perform_general_analysis, perform_specific_analysis

# Configuración de rutas
ROOT_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_EXPORT_DIR = ROOT_DIR / "data" / "export"

DB_FILE = DATA_RAW_DIR / "gencode.v47.annotation.gtf.db"
FASTA_FILE = DATA_RAW_DIR / "GRCh38.primary_assembly.genome.fa"

class PipelineManager:
    def __init__(self):
        DATA_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.db = None
        self.dataset = None

    def _load_db(self):
        if not DB_FILE.exists():
            raise FileNotFoundError(f"Base de datos no encontrada en {DB_FILE}. Ejecute descarga primero.")
        self.db = gffutils.FeatureDB(str(DB_FILE))
        print("✓ Conexión a GFFUtils DB establecida.")

    def run_preprocessing(self, limit_genes=500, balance=True):
        """Paso 1: Extracción, Limpieza, Muestreo de Negativos y Deduplicación."""
        print("\n=== INICIANDO PREPROCESAMIENTO ===")
        self._load_db()
        
        # 1. Sitios Reales
        df_reales = obtener_sitios_splicing(self.db, limit=limit_genes)
        print(f"Extraídos {len(df_reales)} sitios reales únicos por coordenadas.")

        # 2. Secuencias Reales y Limpieza 'N'
        df_reales_seq = extraer_secuencias_y_limpiar(df_reales, FASTA_FILE)
        
        if balance:
            # 3. Generación de Negativos (Decoys)
            print("\nGenerando balance de clases...")
            n_target = len(df_reales_seq)
            df_negativos = obtener_sitios_senuelo(FASTA_FILE, n_target, df_reales_seq)
            
            if df_negativos is not None:
                df_negativos = extraer_secuencias_y_limpiar(df_negativos, FASTA_FILE)
                if len(df_negativos) < n_target:
                    df_sim = generar_decoys_simulados(n_target - len(df_negativos))
                    df_negativos = pd.concat([df_negativos, df_sim], ignore_index=True)
            else:
                df_negativos = generar_decoys_simulados(n_target)
            
            self.dataset = pd.concat([df_reales_seq, df_negativos], ignore_index=True)
        else:
            self.dataset = df_reales_seq

        # Mezclar y guardar
        self.dataset = self.dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        output_path = DATA_EXPORT_DIR / "dataset_consolidado_balanceado.csv"
        self.dataset.to_csv(output_path, index=False)
        print(f"✓ Dataset consolidado guardado en {output_path}")
        print(f"Composición: {self.dataset['label'].value_counts().to_dict()}")

    def run_eda(self):
        """Paso 2: Análisis Exploratorio exhaustivo."""
        print("\n=== EJECUTANDO EDA ===")
        if self.dataset is None:
            path = DATA_EXPORT_DIR / "dataset_consolidado_balanceado.csv"
            if not path.exists():
                print("Error: No hay dataset para analizar.")
                return
            self.dataset = pd.read_csv(path)
        
        if self.db is None: self._load_db()
        
        perform_general_analysis(self.dataset)
        perform_specific_analysis(self.dataset, self.db)
        print("✓ Reporte visual generado en data/export/")

    def run_split(self, test_size=0.2):
        """Paso 3: Partición estratificada para entrenamiento."""
        print(f"\n=== PARTICIÓN DE DATOS (Test size: {test_size}) ===")
        if self.dataset is None:
            self.dataset = pd.read_csv(DATA_EXPORT_DIR / "dataset_consolidado_balanceado.csv")
            
        train_df, test_df = train_test_split(
            self.dataset, 
            test_size=test_size, 
            stratify=self.dataset['label'],
            random_state=42
        )
        
        train_df.to_csv(DATA_EXPORT_DIR / "dataset_train.csv", index=False)
        test_df.to_csv(DATA_EXPORT_DIR / "dataset_test.csv", index=False)
        print(f"✓ Partición completada: Train({len(train_df)}), Test({len(test_df)})")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Genomic Splicing Pipeline")
    parser.add_argument("--step", choices=['all', 'preprocess', 'eda', 'split'], default='all')
    parser.add_argument("--limit", type=int, default=500, help="Límite de genes para procesar")
    
    args = parser.parse_args()
    pipeline = PipelineManager()

    if args.step in ['all', 'preprocess']:
        pipeline.run_preprocessing(limit_genes=args.limit)
    
    if args.step in ['all', 'eda']:
        pipeline.run_eda()
        
    if args.step in ['all', 'split']:
        pipeline.run_split()

if __name__ == "__main__":
    main()
