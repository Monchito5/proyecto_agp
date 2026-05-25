# 🧬 Notebook Unificado - Pipeline de Splicing CNN 1D

## Descripción

**Archivo principal:** `01_pipeline_unificado.ipynb`

Este notebook contiene la implementación lista para Google Colab del pipeline de splicing con CNN 1D.

**Nota importante:** Para la versión completa con optimización evolutiva, interpretabilidad y generación de reportes, usa los módulos de Python en `../src/`:
- `main.py` - Pipeline completo
- `optimizer.py` - Optimización evolutiva de hiperparámetros
- `interpretabilidad.py` - Saliency maps y PWMs
- `generar_reporte.py` - Generación de reportes

---

## Características

✅ **Dos modos de operación:**
- Entrenar desde cero
- Cargar modelo pre-entrenado desde Google Drive

✅ **Soporte para Google Colab:**
- Google Drive para persistencia
- GPU automática (T4)
- Lazy loading para datasets grandes (>100MB)

✅ **Funcionalidades completas:**
- Codificación one-hot
- Entrenamiento con métricas
- Visualización de curvas
- Saliency maps
- Extracción de motivos PWM
- Predicciones

---

## Uso Rápido

### En Google Colab

1. **Sube tus datasets a Google Drive:**
   ```
   Google Drive/
   ├── dataset_entrenamiento.csv
   └── dataset_prueba.csv
   ```

2. **Abre el notebook:**
   - Ve a https://colab.research.google.com
   - Abre `01_pipeline_unificado.ipynb`

3. **Ejecuta en orden:**
   - Runtime → Restart runtime (limpiar)
   - Ejecuta todas las celdas

### Cargar Modelo Pre-entrenado

Si ya tienes un modelo en Drive:

```python
# En la celda "0. Cargar Modelo Pre-entrenado"
if DRIVE_ROOT and os.path.exists(os.path.join(DRIVE_ROOT, "modelo_splicing_best.pth")):
    checkpoint = torch.load(
        os.path.join(DRIVE_ROOT, "modelo_splicing_best.pth"),
        map_location="cpu"  # Compatible CPU/GPU
    )
    modelo.load_state_dict(checkpoint["model_state_dict"])
    print("✓ Modelo cargado desde Drive")
else:
    print("⚠ No hay modelo en Drive. Entrena uno nuevo.")
```

---

## Estructura del Notebook

| Sección | Descripción |
|---------|-------------|
| 0. Cargar Modelo (Opcional) | Carga modelo pre-entrenado |
| 1. Configuración | Importaciones y setup |
| 2. Montar Google Drive | Persistencia de datos |
| 3. Verificar Archivos | Valida datasets |
| 4. Cargar Datasets | Lazy loading |
| 5. Arquitectura CNN | Definición del modelo |
| 6. Entrenamiento | Pipeline completo |
| 7. Evaluación | Métricas y gráficas |
| 8. Saliency Maps | Interpretabilidad |
| 9. Extracción PWM | Motivos biológicos |
| 10. Genes de Interés | Aplicación práctica |

---

## Solución de Problemas

### Error: "FileNotFoundError"

**Causa:** No encuentra los archivos en Drive.

**Solución:**
```python
# Verificar que los archivos están en la raíz de Drive
# Google Drive (Mi unidad) -> dataset_entrenamiento.csv
# Google Drive (Mi unidad) -> dataset_prueba.csv
```

### Error: "CUDA out of memory"

**Causa:** GPU sin suficiente memoria.

**Solución:**
```python
# Reducir batch_size
BATCH_SIZE = 16  # En lugar de 32
```

### Error: "RuntimeError: CUDA device but torch.cuda.is_available() is False"

**Causa:** Intentas cargar modelo de GPU en CPU.

**Solución:**
```python
# Usar map_location='cpu'
checkpoint = torch.load('modelo.pth', map_location='cpu')
```

---

## Funciones Principales

### Entrenar Modelo
```python
modelo, historial = entrenar_modelo(
    train_path='./dataset_entrenamiento.csv',
    val_path='./dataset_prueba.csv',
    num_epochs=50,
    lr=0.001,
    batch_size=32
)
```

### Predecir
```python
# Secuencia individual
prob, clase, etiqueta = predecir_secuencia(modelo, "ACGT"*50)

# Lote de secuencias
resultados = predecir_lote(modelo, secuencias)
```

### Extraer Métricas
```python
# Del historial
train_loss = historial['train_loss']
val_acc = historial['val_acc']

# Estadísticas
import numpy as np
print(f"Mejor Val Loss: {np.min(val_loss):.4f}")
print(f"Época óptima: {np.argmin(val_loss)}")
```

---

## Archivos Generados

Después del entrenamiento, en Google Drive tendrás:

| Archivo | Descripción |
|---------|-------------|
| `modelo_splicing_best.pth` | Modelo entrenado |
| `curvas_entrenamiento.png` | Gráficas de loss/accuracy |
| `matriz_confusion.png` | Matriz de confusión |
| `curva_roc.png` | Curva ROC |

---

## Requisitos

### Google Colab (Recomendado)
- Cuenta de Google
- 15 GB de espacio en Drive
- Conexión a internet

### Local
```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn
```

---

## Referencias

- **Documentación PyTorch:** https://pytorch.org/docs/
- **Google Colab:** https://colab.research.google.com/
- **Tutorial CNN:** https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

---

## Soporte

Para dudas o problemas:
1. Revisa la sección "Solución de Problemas"
2. Verifica las rutas de los archivos
3. Asegúrate de tener las dependencias instaladas

---

**Última actualización:** Mayo 2026  
**Versión:** 1.0.0  
**Autores:** César Martínez | Guillermo Zaragoza
