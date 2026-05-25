# CENTRO UNIVERSITARIO DE CIENCIAS EXACTAS E INGENIERÍAS
## UNIVERSIDAD DE GUADALAJARA

**Asignatura:** ANÁLISIS GENÓMICO Y PROTEÓMICO  
**Mtro.** MOISES SOTELO RODRIGUEZ  
**Sección:** D01  
**Entregable 1.** Selección de tema y propuesta metodológica  

**Autores:**
- César Alexander Martínez Pérez (219758478)
- Guillermo Daniel Zaragoza Castro (219873153)

**Ingeniería informática**

---

## Introducción

El splicing alternativo es un proceso celular fundamental que permite generar múltiples isoformas de ARN mensajero a partir de un mismo gen, aumentando la diversidad del proteoma. La correcta identificación de los sitios de splicing (donante y aceptor) es crucial, ya que errores en este proceso están asociados a numerosas enfermedades genéticas. Sin embargo, la mera predicción de estos sitios no revela qué características de la secuencia guían la decisión del espliceosoma.

El presente proyecto se enfoca en interpretar qué aprende un modelo computacional al distinguir sitios de splicing verdaderos de señuelos. Utilizando un clasificador simple basado en redes neuronales convolucionales y herramientas de visualización de rasgos, se busca identificar motivos biológicos conocidos, como el tracto de polipirimidina y el punto de ramificación, así como descubrir nuevos patrones relevantes en las regiones flanqueantes de los sitios de splicing.

---

## Dataset

Para construir un conjunto de datos equilibrado de secuencias de ADN que flanquean sitios de splicing:

- **Sitios verdaderos (positivos):** Se extraerán anotaciones de GENCODE (versión humana, GRCh38) para obtener sitios donantes (GT-AG) constitutivos y alternativos. Se considerarán ventanas de 200 nucleótidos alrededor del sitio (100 nt hacia el exón y 100 nt hacia el intrón).

- **Sitios falsos (negativos):** Se generarán muestreando posiciones con dinucleótidos GT o AG que no correspondan a sitios de splicing anotados, asegurando que no estén en exones validados, o sea, sean estos señuelos. Se mantendrá una proporción 1:1.

Se considera de forma opcional, con el objetivo de enriquecer el contexto de eventos alternativos reales, integrar la base de datos ALTssDB (sitios 5' y 3' alternativos). No se incluirán datos de otras especies ni de validación proteómica, limitando el análisis a humano.

---

## Metodología

Se implementará un clasificador binario basado en CNN distribuido en los siguientes puntos:

1. **Codificación de secuencias:** Las secuencias de ADN de longitud fija (ej. 200 pb) se codificarán mediante one-hot encoding (4 canales: A, T, C, G).

2. **Arquitectura CNN:**
   - Arquitectura base configurable: 3 bloques convolucionales 1D con BatchNorm, ReLU y Dropout.
   - Filtros y tamaños de kernel optimizados mediante algoritmo evolutivo (rango típico: 64-256 filtros, kernels 7-15).
   - Max-pooling estratificado para reducción dimensional.
   - Capas densas finales con dropout y activación sigmoide para clasificación binaria.

3. **Entrenamiento:** Se utilizará el algoritmo evolutivo Adam como optimizador, función de pérdida de entropía cruzada binaria. Se espera gestionar un 80% entrenamiento y 20% prueba.

4. **Interpretación de motivos:** Una vez entrenado, se aplicarán técnicas de atribución de características:
   - **Saliency maps:** cálculo del gradiente de la salida respecto a los nucleótidos de entrada, identificando posiciones críticas.
   - **Visualización de los pesos** de la primera capa convolucional como position weight matrices (PWMs) para extraer motivos candidatos.

Estos métodos permitirán responder preguntas clave sobre la secuencia.

---

## Pipeline

El flujo de trabajo se planea realizar en las siguientes etapas:

### 1. Adquisición y preprocesamiento
- Descarga de anotaciones GENCODE (archivo GTF) y secuencias genómicas (FASTA).
- Extracción de ventanas centradas en sitios donantes (GT) y aceptores (AG). Balance del dataset con señuelos.
- **Propagación de metadatos:** Se extrae y almacena el `gene_id` y `gene_name` de cada secuencia para facilitar el análisis posterior por gen.
- **Hard-Negative Mining (opcional):** El muestreo de señuelos permite generar ejemplos difíciles (GT cercanos a sitios reales pero no anotados) para mejorar el aprendizaje del modelo.
- One-hot encoding y partición entrenamiento/prueba.

### 2. Entrenamiento del modelo CNN
- Implementación en Python con PyTorch.
- Entrenamiento con early stopping y validación.
- Optimización de hiperparámetros mediante algoritmo evolutivo (opcional).

### 3. Extracción de motivos interpretables
- Cálculo de **Saliency Maps** para identificar nucleótidos críticos en la secuencia.
- Implementación de **Integrated Gradients** (Sundararajan et al., 2017) para atribuciones robustas del impacto de cada nucleótido en la predicción.
- Agregación por posición: promedio de importancia por nucleótido y visualización como logo de secuencia utilizando `logomaker`.
- **Extracción de PWMs:** Análisis de los filtros de la primera capa convolucional (capa_conv1). Los pesos se transforman en Position Weight Matrices (PWMs) y se visualizan como logos de secuencia para identificar motivos candidatos.
- Comparación prospectiva con bases de datos de motivos conocidos (JASPAR, CISBP-RNA).

### 4. Validación biológica
- Comparar los motivos descubiertos con bases de datos de elementos reguladores de splicing (SFMetaDB, RBPDB).
- Verificar la presencia de motivos conocidos en las regiones de alta importancia.

> **Nota:** Este punto es actualmente opcional, de ser posible.

### 5. Aplicación a genes de interés
- El modelo entrenado se ejecuta sobre secuencias de genes específicos (ej. BRCA1, TP53) mediante el script `src/predict_gene.py`. Este script realiza una ventana deslizante sobre la región del gen, predice cada ventana y genera visualizaciones de las regiones que el modelo considera más relevantes (incluyendo Saliency Maps dinámicos).
- Se facilita la identificación de posibles sitios alternativos no anotados y la priorización de candidatos para validación experimental.

---

## Bibliografía

- GENCODE. (2025). GENCODE - FAQ. Recuperado de https://www.gencodegenes.org/pages/faq.html
- Carranza, F., Shenasa, H., & Hertel, K. J. (2022). Splice site proximity influences alternative exon definition. *RNA Biology*, 19(1), 829-840.
- Hertel, K., Carranza, F., & Shenasa, H. (2022). Database of human alternative 5’ and 3’ splice sites. *Dryad*. https://doi.org/10.7280/D12108
- Buratti, E., Chivers, M., Hwang, G., & Vorechovsky, I. (2011). DBASS3 and DBASS5: Databases of aberrant 3’- and 5’-splice sites. *Nucleic Acids Research*, 39(suppl_1), D86-D91.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *Proceedings of the 34th International Conference on Machine Learning*, 70, 3319–3328. (Integrated Gradients)
- Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside convolutional networks: Visualising image classification models and saliency maps. *arXiv preprint arXiv:1312.6034*.
- Garmendia, I. E. (2015). *Proteogenómica y splicing alternativo*. Universidad Autónoma de Madrid. (Contexto biológico, no metodológico)