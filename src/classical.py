import cv2
import numpy as np

# ==========================
# 1. Otsu Thresholding
# ==========================
def otsu_segmentation(image):
    """Segmentación binaria usando Otsu."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


# ==========================
# 2. K-Means Clustering
# ==========================
def kmeans_segmentation(image, k=2):
    """Segmentación usando clustering K-Means en el espacio de color."""
    # Si es color, convertir a gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Asegurar tipo uint8
    gray_uint8 = (gray * 255).astype(np.uint8) if gray.dtype == np.float32 else gray.astype(np.uint8)

    # Aplicar Otsu
    _, mask = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


# ==========================
# 3. Canny Edge Detection
# ==========================
def canny_segmentation(image, low_threshold=100, high_threshold=200):
    """Detección de bordes con Canny."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


# ==========================
# 4. Watershed Algorithm
# ==========================
def watershed_segmentation(image):
    """Segmentación basada en Watershed."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Operaciones morfológicas
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Background seguro
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Foreground seguro
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Áreas desconocidas
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marcadores
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)

    # Regiones de foreground = 255, fondo = 0
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255

    return mask


import numpy as np
import cv2
from sklearn.cluster import KMeans

def kmeans_segmentation_lab(image, k=2):
    # Convertir BGR → LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Aplanar la imagen en un array de features
    pixel_vals = lab.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixel_vals)

    # Reconstruir la segmentación
    segmented = labels.reshape((image.shape[0], image.shape[1]))

    # Normalizar para que sea 0-255
    segmented = (segmented * (255 // (k-1))).astype(np.uint8)

    return segmented

def kmeans_segmentation_hsv(image, k=2):
    """
    Segmentación usando KMeans en el espacio HSV.
    Retorna una imagen segmentada en 0-255 según el cluster.
    """
    # Convertir BGR → HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Aplanar la imagen en un array de features
    pixel_vals = hsv.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixel_vals)

    # Reconstruir la segmentación
    segmented = labels.reshape((image.shape[0], image.shape[1]))

    # Normalizar para que sea 0-255
    segmented = (segmented * (255 // (k-1))).astype(np.uint8)

    return segmented

# ==========================
# Wrapper para ejecutar todos los segmentadores clásicos
# ==========================
def run_segmenters(images, method="kmeans", k=2, space="bgr"):
    masks_pred = []

    for img in images:
        if method.lower() == "otsu":
            pred = otsu_segmentation(img)
        elif method.lower() == "kmeans":
            if space == "lab":
                pred = kmeans_segmentation_lab(img, k=k)
            elif space == 'hsv':
                pred = kmeans_segmentation_hsv(img, k=k)
            else:  # default BGR
                pred = kmeans_segmentation(img, k=k)
        elif method.lower() == "canny":
            pred = canny_segmentation(img)
        elif method.lower() == "watershed":
            pred = watershed_segmentation(img)
        else:
            raise ValueError(f"Método {method} no soportado")

        masks_pred.append(pred)

    return masks_pred
