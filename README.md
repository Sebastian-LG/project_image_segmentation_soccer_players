# Segmentación de jugadores de fútbol en imágenes

Este proyecto implementa métodos clásicos y modelos preentrenados para la **segmentación de jugadores de fútbol en imágenes**. Permite preprocesar imágenes, segmentar jugadores, aplicar postprocesamiento y evaluar los resultados.  

---

## Estructura del proyecto

```

project\_image\_segmentation\_soccer\_players/
│
├── .dvc                  # Archivos de configuración DVC
├── .dvcignore            # Ignorar archivos para DVC
├── .gitignore            # Ignorar archivos para Git
├── src/                  # Código fuente principal
├── classical.py          # Implementación de métodos clásicos (KMeans, Otsu, etc.)
├── demo.ipynb            # Notebook de demostración y experimentos
├── evaluation.py         # Funciones de evaluación de segmentación
├── postprocessing.py     # Funciones de postprocesamiento de máscaras
├── preprocessing.py      # Funciones de preprocesamiento de imágenes
├── pretrained.py         # Uso de modelos preentrenados
├── utils.py              # Funciones auxiliares
└── README.md             # Este archivo
````

---

## Requisitos

- Python 3.9 o superior  
- Librerías principales:
  - `numpy`
  - `opencv-python`
  - `scikit-learn`
  - `matplotlib`
  - `torch` (para modelos preentrenados)
  - `torchvision`

Puedes instalar las dependencias usando:

```bash
pip install -r requirements.txt
````

---

## Uso

### 1. Preprocesamiento

```python
from preprocessing import preprocess_image

img_preprocesada = preprocess_image("ruta/a/imagen.jpg")
```

### 2. Segmentación clásica

```python
from classical import segment_kmeans

mask = segment_kmeans(img_preprocesada, k=2)
```

### 3. Segmentación con modelo preentrenado

```python
from pretrained import segment_with_model

mask = segment_with_model("ruta/a/imagen.jpg")
```

### 4. Postprocesamiento

```python
from postprocessing import refine_mask

mask_refinada = refine_mask(mask)
```

### 5. Evaluación

```python
from evaluation import evaluate_segmentation

metrics = evaluate_segmentation(mask_refinada, mask_ground_truth)
print(metrics)
```

### 6. Notebook de demostración

Abre `demo.ipynb` para ver ejemplos completos de uso, visualización de resultados y comparativa entre métodos clásicos y preentrenados.

---

## Flujo de trabajo recomendado

1. Preprocesar las imágenes (`preprocessing.py`)
2. Aplicar segmentación clásica (`classical.py`) o preentrenada (`pretrained.py`)
3. Refinar máscaras con postprocesamiento (`postprocessing.py`)
4. Evaluar resultados (`evaluation.py`)
5. Ajustar parámetros y repetir según sea necesario

---

## Licencia

Proyecto bajo licencia MIT.

---

