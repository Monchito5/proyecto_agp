import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

# Importar módulos locales
from dataset import SplicingDataset, collate_fn
from model import SplicingCNN1D, count_parameters


# Configuración de rutas
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_EXPORT_DIR = PROJECT_ROOT / "data" / "export"

# Archivos
DATASET_FILE = DATA_EXPORT_DIR / "dataset_consolidado_balanceado.csv"

# Hiperparámetros
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Semilla para reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_and_split_dataset(
    csv_path: Path,
    train_ratio: float = 0.8,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
   
    print(f"Cargando el dataset: {csv_path}")
    
    # Cargar DataFrame para partición estratificada
    df = pd.read_csv(csv_path)
    
    # Verificar balance
    label_counts = df['label'].value_counts()
    print(f"Total muestras: {len(df)}")
    print(f"Clase 1: {label_counts.get(1, 0)}")
    print(f"Clase 0: {label_counts.get(0, 0)}")
    
    # Partición estratificada 80/20
    train_df, val_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df['label'],
        random_state=SEED
    )
    
    print(f"Train: {len(train_df)} muestras")
    print(f"Val: {len(val_df)} muestras")
    
    # Crear datasets y loaders
    # Nota: SplicingDataset espera ruta CSV, creamos archivos temporales
    train_path = DATA_EXPORT_DIR / "temp_train.csv"
    val_path = DATA_EXPORT_DIR / "temp_val.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    train_dataset = SplicingDataset(train_path)
    val_dataset = SplicingDataset(val_path)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    # Limpiar archivos temporales
    os.remove(train_path)
    os.remove(val_path)
    
    return train_loader, val_loader


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    
    predictions = (outputs >= 0.5).float()
    correct = (predictions == labels.float()).sum().item()
    total = labels.size(0)
    return correct / total * 100


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    for batch_idx, (sequences, labels) in enumerate(loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Gradientes a cero
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        outputs = outputs.squeeze(-1)  # (batch, 1) -> (batch,)
        
        # Calcular pérdida
        loss = criterion(outputs, labels.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Acumular métricas
        total_loss += loss.item()
        total_acc += calculate_accuracy(outputs, labels)
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(sequences)
            outputs = outputs.squeeze(-1)
            
            # Calcular pérdida
            loss = criterion(outputs, labels.float())
            
            # Acumular métricas
            total_loss += loss.item()
            total_acc += calculate_accuracy(outputs, labels)
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    output_dir: Path
) -> None:
    
    epochs = range(1, len(train_losses) + 1)
    
    # Configuración de estilo
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Gráfica de Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Loss (BCE)', fontsize=12)
    ax.set_title('Curva de Pérdida (Loss) durante Entrenamiento', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    loss_path = output_dir / 'curva_loss.png'
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[GRÁFICA] Guardada: {loss_path}")
    
    # Gráfica de Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Curva de Precisión (Accuracy) durante Entrenamiento', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    acc_path = output_dir / 'curva_accuracy.png'
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafica guardada: {acc_path}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str,
    output_dir: Path
) -> None:
    
    print(f"\nIniciando en dispositivo: {device}")
    print(f"Épocas: {num_epochs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 70)
    
    # Listas para guardar métricas
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Mejor modelo
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Guardar métricas
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Imprimir progreso
        print(f"Época {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    print("=" * 70)
    print(f"[ENTRENAMIENTO] Mejor Val Loss: {best_val_loss:.4f}")
    
    # Guardar mejor modelo
    if best_model_state is not None:
        model_path = output_dir / 'best_model.pth'
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, model_path)
        print(f"[MODELO] Mejor modelo guardado: {model_path}")
    
    # Generar gráficas
    print("\n[GRÁFICAS] Generando curvas de rendimiento...")
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs, output_dir
    )


def main():
    """
    Ejecución principal del pipeline de entrenamiento.
    """
    print("=" * 70)
    print("MÓDULO 5: ENTRENAMIENTO DEL MODELO CNN 1D")
    print("=" * 70)
    print(f"Directorio de trabajo: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_FILE}")
    print(f"Disponible GPU: {torch.cuda.is_available()}")
    print("=" * 70)
    
    try:
        # Paso 1: Cargar y particionar datos
        print("\n[1] CARGA Y PARTICIÓN DE DATOS...")
        train_loader, val_loader = load_and_split_dataset(
            DATASET_FILE,
            train_ratio=0.8,
            batch_size=BATCH_SIZE
        )
        
        # Paso 2: Instanciar modelo
        print("\n[2] INSTANCIANDO MODELO...")
        model = SplicingCNN1D(
            input_length=200,
            input_channels=4,
            dropout_rate_conv=0.4,
            dropout_rate_fc=0.5
        ).to(DEVICE)
        
        total_params, trainable_params = count_parameters(model)
        print(f"    Parámetros totales: {total_params:,}")
        print(f"    Parámetros entrenables: {trainable_params:,}")
        
        # Paso 3: Configurar pérdida y optimizador
        print("\n[3] CONFIGURANDO ENTRENAMIENTO...")
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5
        )
        
        # Paso 4: Entrenar modelo
        print("\n[4] INICIANDO ENTRENAMIENTO...")
        train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            NUM_EPOCHS,
            DEVICE,
            DATA_EXPORT_DIR
        )
        
        print("\n" + "=" * 70)
        print("[OK] Pipeline de entrenamiento completado exitosamente.")
        print("=" * 70)
        print(f"\nResultados guardados en: {DATA_EXPORT_DIR}")
        print("  - curva_loss.png")
        print("  - curva_accuracy.png")
        print("  - best_model.pth")
        print("\nProyecto completado exitosamente!")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()