import cv2
import numpy as np

def preprocess_image(image, resize_factor=0.5, use_hsv=False):
    """
    Preprocesa la imagen antes de segmentación.
    - Redimensiona
    - Aplica suavizado
    - Convierte a HSV opcionalmente
    - Normaliza valores
    
    Args:
        image (np.ndarray): imagen original BGR
        resize_factor (float): escala de reducción (0.5 = mitad)
        use_hsv (bool): convertir a HSV si True, quedarse en BGR si False
        
    Returns:
        np.ndarray: imagen preprocesada lista para clustering
    """
    # 1. Redimensionar
    h, w = image.shape[:2]
    image_resized = cv2.resize(image, (int(w * resize_factor), int(h * resize_factor)))

    # 2. Suavizado (suaviza césped y público pero mantiene bordes de jugadores)
    image_blur = cv2.bilateralFilter(image_resized, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. Espacio de color
    if use_hsv:
        image_color = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
    else:
        image_color = image_blur  # se queda en BGR

    # 4. Normalizar
    image_norm = image_color.astype(np.float32) / 255.0

    return image_norm
