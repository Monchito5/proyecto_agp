> [!NOTE] proyecto_agp
Proyecto Análisis Genómico y Proteómico Equipo con más flow de qcei

---

# 🧬 Predicción de Sitios de Splicing mediante CNN Interpretable

Este proyecto implementa un *pipeline* de aprendizaje profundo para clasificar sitios de splicing verdaderos vs. señuelos, con un enfoque en la **interpretabilidad biológica** de los motivos de secuencia aprendidos por el modelo.

El propósito principal es **entender qué patrones de ADN identifica una CNN** al distinguir sitios de splicing funcionales de aquellos que no lo son.

---

## 📑 Tabla de Contenidos

- [Visión General y Objetivos](#visión-general-y-objetivos)
- [Arquitectura del Pipeline](#arquitectura-del-pipeline)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos e Instalación](#requisitos-e-instalación)
- [Preparación de Datos](#preparación-de-datos)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [Interpretación de Motivos](#interpretación-de-motivos)
- [Ejecución Rápida](#ejecución-rápida)
- [Validación y Métricas](#validación-y-métricas)
- [Limitaciones y Trabajo Futuro](#limitaciones-y-trabajo-futuro)
- [Referencias](#referencias)

---

> ## Visión General y Objetivos

### 1. Introducción

El *splicing alternativo* es un proceso celular fundamental que permite generar múltiples isoformas de ARN mensajero a partir de un mismo gen. La correcta identificación de los sitios de splicing (donante y aceptor) es crucial, ya que errores en este proceso están asociados a numerosas enfermedades genéticas.

### 2. Objetivos Específicos

1. **Construir** un dataset equilibrado de secuencias de ADN que flanquean sitios de splicing verdaderos y señuelos.
2. **Entrenar** un clasificador CNN para distinguir sitios verdaderos de falsos.
3. **Interpretar** los motivos de secuencia aprendidos mediante *saliency maps* y PWMs.
4. **Validar** los motivos descubiertos con bases de datos biológicas (SFMetaDB, RBPDB).

### 3. Alcance y limitaciones

- **Enfoque**: Humano (*Homo sapiens*, ensamblaje GRCh38).
- **Limitación principal**: No se incluyen otras especies ni validación proteómica. El análisis se limita a secuencias de ADN y anotaciones GENCODE.

---

> ## Arquitectura del proyecto

Arquitectura modular de procesos secuenciales:

1. **Adquisición y Preprocesamiento**  
   - Descarga de anotaciones GENCODE (GTF) y genoma de referencia (FASTA).  
   - Uso principal de **`GFFUtils`** para limpieza, extracción y consulta jerárquica de regiones de interés.  
   - Codificación *one-hot* de las secuencias.

2. **Generación de Dataset Balanceado**  
   - **Sitios verdaderos**: extraídos de la columna `feature = "exon"` en el GTF.  
   - **Sitios señuelo**: muestreo de dinucleótidos `GT`/`AG` no anotados.  
   - Ventanas de **200 pb** (100 pb exón + 100 pb intrón).

3. **Entrenamiento de CNN**  
   - Arquitectura: 2 capas convolucionales 1D + *pooling* + densa.  
   - Optimizador: Adam.  
   - Función de pérdida: entropía cruzada binaria.

4. **Interpretabilidad**  
   - *Saliency maps* para cada nucleótido.  
   - Conversión de pesos de filtros a PWMs.  
   - Comparación con bases de datos de motivos (JASPAR, CISBP‑RNA).

5. **Validación Biológica**  
   - Verificación de la presencia del tracto de polipirimidina y punto de ramificación en las regiones de alta importancia.

---

> ## Estructura del Proyecto

El proyecto se estructura de la siguiente manera:

```
proyecto_agp/
├── data/
│   ├── raw/                     # Datos sin procesar (GTG, FASTA)
│   ├── processed/               # Ventanas extraídas, one-hot, splits
│   └── external/                # Bases de datos externas (ALTssDB, SFMetaDB)
├── docs/                        # Documentación técnica y de usuario
├── notebooks/                   # Exploración y prototipado (Jupyter)
├── src/                         # Código fuente del pipeline
│   ├── data_preprocessing.py    # Uso principal de GFFUtils
│   ├── eda.py                   # Análisis Exploratorio de Datos
│   ├── model.py                 # Definición de la CNN
│   ├── train.py                 # Entrenamiento y early stopping
│   ├── interpret.py             # Saliency maps y PWMs
│   └── utils.py                 # Funciones auxiliares
├── models/                      # Modelos guardados (.h5, SavedModel)
├── results/                     # Figuras, logs, matrices de motivos
├── requirements.txt             # Dependencias Python
├── README.md                    # Este archivo
└── LICENSE                      # MIT u open-source
```

---

> ## Requisitos e Instalación

El proyecto se configura mediante **entornos virtuales de Python** (`venv`) y scripts nativos para cada sistema operativo. Deseable utilizar conda.

### Herramientas principales

| Herramienta       | Versión    | Propósito                                                                 |
|-------------------|------------|---------------------------------------------------------------------------|
| Python            | ≥3.9       | Lenguaje base                                                             |
| TensorFlow / Keras| 2.13       | Implementación de la CNN                                                  |
| **GFFUtils**      | 0.12.0     | Manipulación, limpieza y consulta de archivos GTF/GFF (compatible Python 3) |
| gtfparse          | 1.3.0      | Lectura alternativa rápida a DataFrame (opcional)                         |
| pyfaidx           | 0.7.1      | Indexado y extracción rápida de secuencias desde FASTA                    |
| logomaker         | 0.8        | Visualización de PWMs                                                     |

### Instalación paso a paso

#### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/proyecto_agp.git
cd proyecto_agp
```

#### 2. Crear y activar entorno virtual

**Linux / macOS (bash):**
```bash
python3 -m venv splicing_env
source splicing_env/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv splicing_env
.\splicing_env\Scripts\Activate.ps1
```

#### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Scripts de configuración automatizada

Se incluyen scripts para crear el entorno e instalar todo automáticamente:

- **Linux/macOS (`setup_env.sh`)**:
  ```bash
  chmod +x setup_env.sh
  ./setup_env.sh
  ```

- **Windows (`setup_env.ps1`)**:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\setup_env.ps1
  ```

Estos scripts realizan la creación del `venv`, la activación y la instalación de los requisitos.

---

> ## Preparación de Datos

### 1. Descarga de datos

El *pipeline* utiliza:

- **Anotaciones**: [GENCODE v47](https://www.gencodegenes.org/human/) (archivo `gencode.v47.annotation.gtf.gz`).
- **Genoma**: [GRCh38 primary assembly](https://www.gencodegenes.org/human/release_47.html).

Para automatizar la descarga, ejecutar:

```bash
python src/download_data.py
```

### 2. Procesamiento principal con GFFUtils

**GFFUtils** es la herramienta central para manejar el GTF. Crea una base de datos SQLite que permite navegar relaciones jerárquicas (gen → transcript → exon) y filtrar por atributos de manera eficiente.

**Ventajas de GFFUtils**:
- Permite consultas como `db.children(id, featuretype='exon')` para obtener los exones de un transcript específico.
- Mantiene la fidelidad del GTF original.
- Completamente compatible con Python 3 (versiones ≥ 0.10.0).

**Uso opcional de `gtfparse`**: Para operaciones simples que requieran un DataFrame plano (estadísticas rápidas), se puede utilizar `gtfparse` como complemento, pero todo el flujo principal se basa en GFFUtils.

### 3. Extracción de ventanas y generación de dataset balanceado

El script `src/data_preprocessing.py` realiza:

- Extracción de ventanas de ±100 pb alrededor de cada sitio donante/aceptor utilizando las coordenadas extraídas con GFFUtils.
- Muestreo de señuelos (posiciones GT/AG no anotadas).
- Codificación one-hot y partición 80/20.

---

> ## Entrenamiento del Modelo

### Arquitectura CNN (src/model.py)

```python
model = Sequential([
    Conv1D(filters=32, kernel_size=7, activation='relu', input_shape=(200, 4)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### Hiperparámetros

| Parámetro          | Valor  |
|--------------------|--------|
| *Learning rate*    | 0.001  |
| *Batch size*       | 64     |
| *Épocas*           | 30 (con *early stopping*) |
| *Loss function*    | Binary Crossentropy |
| *Optimizer*        | Adam   |

### Entrenamiento (src/train.py)

```bash
python src/train.py --data data/processed --epochs 30 --batch 64
```

El mejor modelo se guarda en `models/best_model.h5`.

---

> ## Interpretación de Motivos

### Generación de *Saliency Maps*

```python
from src.interpret import compute_saliency

saliency = compute_saliency(model, X_test, y_test)
```

### Conversión de filtros a PWMs

```python
from src.interpret import filters_to_pwm

pwms = filters_to_pwm(model.layers[0].get_weights()[0])
```

### Comparación con bases de datos

Los motivos extraídos se comparan con matrices de posición de **JASPAR** y **CISBP‑RNA** para identificar similitudes con factores de splicing conocidos.

---

> ## Ejecución Rápida

1. **Clonar el repositorio**  
   ```bash
   git clone https://github.com/tu-usuario/proyecto_agp.git
   cd proyecto_agp
   ```

2. **Configurar el entorno** (usando scripts según tu SO)  
   - En Linux/macOS: `./setup_env.sh`  
   - En Windows: `.\setup_env.ps1`

3. **Descargar y preprocesar los datos**  
   ```bash
   python src/download_data.py
   python src/data_preprocessing.py
   ```

4. **Entrenar el modelo**  
   ```bash
   python src/train.py
   ```

5. **Interpretar los motivos**  
   ```bash
   python src/interpret.py --model models/best_model.h5
   ```

---

> ## Validación y Métricas

| Métrica            | Propósito                                                                 |
|--------------------|---------------------------------------------------------------------------|
| **Exactitud (Accuracy)**   | Porcentaje de clasificaciones correctas.                                 |
| **Precisión (Precision)**  | De todas las predicciones positivas, ¿cuántas son realmente correctas?   |
| **Recall (Sensibilidad)**  | De todos los sitios verdaderos, ¿cuántos detectó el modelo?              |
| **F1‑Score**               | Media armónica entre precisión y *recall*.                               |
| **Matriz de confusión**     | Visualización de aciertos y errores.                                     |
| **Curva ROC / AUC**         | Capacidad discriminativa del modelo.                                     |

Además, la **validación biológica** se realiza comparando los PWMs obtenidos con las matrices de posición conocidas de factores de splicing.

---

### Limitaciones actuales

- **Dependencia de anotaciones**: El modelo solo aprende de sitios anotados en GENCODE; puede no generalizar a eventos de splicing no canónicos.
- **Falsos negativos/positivos**: El muestreo de señuelos no es perfecto; algunos sitios no anotados podrían ser funcionales en condiciones celulares específicas.
- **Rendimiento computacional**: El uso de GFFUtils requiere la creación de una base de datos SQLite, que puede ser lenta en archivos GTF muy grandes (>5 GB). Se recomienda ejecutar en un sistema con al menos 8 GB de RAM.

---

> ## Referencias

1. **GENCODE**. (2025). *GENCODE - FAQ*. Recuperado de https://www.gencodegenes.org/pages/faq.html
2. **GFFUtils Documentation**. (n.d.). https://gffutils.readthedocs.io/ (Versión actual compatible con Python 3)
3. **Carranza, F., Shenasa, H., & Hertel, K. J.** (2022). Splice site proximity influences alternative exon definition. *RNA Biology*, 19(1), 829-840.
4. **Hertel, K., Carranza, F., & Shenasa, H.** (2022). Database of human alternative 5’ and 3’ splice sites. *Dryad*. https://doi.org/10.7280/D12108
5. **Buratti, E., Chivers, M., Hwang, G., & Vorechovsky, I.** (2011). DBASS3 and DBASS5: Databases of aberrant 3’- and 5’-splice sites. *Nucleic Acids Research*, 39(suppl_1), D86-D91.
6. **Sundararajan, M., Taly, A., & Yan, Q.** (2017). Axiomatic attribution for deep networks. *Proceedings of the 34th International Conference on Machine Learning*, 70, 3319–3328.
7. **Simonyan, K., Vedaldi, A., & Zisserman, A.** (2013). Deep inside convolutional networks: Visualising image classification models and saliency maps. *arXiv preprint arXiv:1312.6034*.
8. **gtfparse Documentation**. (n.d.). https://pypi.org/project/gtfparse/

---

> ## Autores
**Ingeniería Informática – (CUCEI) UdeG**

- **César Alexander Martínez Pérez** – 219758478  
- **Guillermo Daniel Zaragoza Castro** – 219873153  

