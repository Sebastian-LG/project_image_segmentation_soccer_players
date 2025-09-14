import cv2
import numpy as np

def postprocess_mask(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    Postprocesa una máscara binaria:
    - Convierte a binaria (0,255)
    - Aplica operaciones morfológicas
    - Elimina regiones pequeñas

    Args:
        mask (np.ndarray): máscara binaria (0-255 o 0-1)
        min_area (int): área mínima para conservar una región

    Returns:
        np.ndarray: máscara procesada (0-255)
    """
    # Asegurar formato binario 0/255
    mask = (mask > 127).astype(np.uint8) * 255  

    # Morfología suave: kernel pequeño
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # rellena huecos
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # elimina ruido fino

    # Eliminar regiones extremadamente pequeñas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 255

    return cleaned_mask
