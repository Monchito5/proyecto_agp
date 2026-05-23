# 🧬 Predicción de Sitios de Splicing mediante CNN Interpretativa

Este proyecto implementa un *pipeline* de alto rendimiento para la identificación de sitios de splicing (donantes) en el genoma humano (GRCh38). Utiliza redes neuronales convolucionales (CNN) optimizadas mediante algoritmos evolutivos para lograr una alta precisión y robustez biológica.

---

## 🏗️ Arquitectura del Pipeline

El sistema está diseñado bajo una arquitectura modular y profesional, orquestada por un punto de entrada único. Cada componente tiene responsabilidades independientes:

1.  **Preprocesamiento y Limpieza** (`src/data_preprocessing.py`): 
    *   Extracción ultra-rápida de secuencias mediante **pyfaidx**.
    *   Generación de sitios señuelo (negativos) biológicamente realistas buscando dinucleótidos `GT` no funcionales.
    *   Deduplicación estricta en dos niveles (coordenadas y secuencias) para evitar *data leakage*.
    *   Filtro de calidad exhaustivo para eliminar ruido (nucleótidos ambiguos 'N').

2.  **Análisis Exploratorio (EDA)** (`src/eda.py`):
    *   Generación de 11+ visualizaciones técnicas siguiendo estándares de producción.
    *   Análisis de composición nucleotídica posicional, balance de clases y contenido GC.
    *   Validación de la jerarquía genómica utilizando **GFFUtils**.

3.  **Optimización Evolutiva** (`src/optimizer.py`):
    *   Búsqueda de hiperparámetros mediante **Evolución Diferencial (DE/rand/1/bin)**.
    *   Ajuste automático de la arquitectura (filtros, kernels) y parámetros de entrenamiento (LR, Dropout).

4.  **Modelo y Entrenamiento** (`src/model.py`, `src/train.py`):
    *   CNN 1D profunda con capas de normalización de lote (BatchNorm) y regularización avanzada.
    *   Carga de datos optimizada en PyTorch con codificación **One-Hot** en tiempo real.

---

## 🚀 Guía de Uso (Nivel Producción)

### 1. Preparación del Entorno
Instale las dependencias necesarias:
```powershell
pip install torch pandas matplotlib seaborn gffutils pyfaidx scikit-learn numpy
```

### 2. Descarga de Recursos Genómicos
Descargue el genoma de referencia y las anotaciones GENCODE:
```powershell
python src/download_data.py
```

### 3. Ejecución del Pipeline Completo
El script `main.py` permite orquestar todas las etapas:

```powershell
# Flujo completo: Preprocesamiento -> EDA -> Partición
python src/main.py --paso todo

# Búsqueda de hiperparámetros óptimos mediante algoritmo evolutivo
python src/main.py --paso opt
```

**Parámetros principales:**
*   `--paso`: Etapa a ejecutar (`pre`, `eda`, `opt`, `part`, `todo`).
*   `--limite`: Número de genes a procesar (0 para el genoma completo).

---

## 📊 Estándares de Ingeniería

El código cumple estrictamente con 24 reglas de escritura de software:
*   **Idioma**: Código y documentación 100% en español con ortografía impecable.
*   **Identificadores**: Nomenclatura significativa con palabras $\ge 3$ caracteres.
*   **Modularidad**: Funciones limitadas a 50 líneas para asegurar mantenibilidad.
*   **Robustez**: Manejo de excepciones con trazas de pila completas y cero código muerto.

---

## 👥 Autores
**Ingeniería Informática – (CUCEI) UdeG**
- César Alexander Martínez Pérez
- Guillermo Daniel Zaragoza Castro
