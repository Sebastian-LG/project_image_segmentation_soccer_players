import torch
import torchvision
import cv2
import numpy as np

# Cargar modelo Mask R-CNN preentrenado en COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # modo evaluación

# Función para procesar una imagen
def segment_players(image, threshold=0.5):
    """
    image: imagen BGR (OpenCV)
    threshold: umbral de confianza para mantener predicciones
    Returns:
        mask: máscara binaria (jugadores=255, fondo=0)
    """
    # Convertir BGR → RGB y normalizar
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2,0,1).unsqueeze(0)

    # Predicción
    with torch.no_grad():
        outputs = model(img_tensor)

    masks = outputs[0]['masks']  # shape (N,1,H,W)
    labels = outputs[0]['labels']  # categorías COCO
    scores = outputs[0]['scores']

    # COCO label 1 = persona
    mask_total = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask, label, score in zip(masks, labels, scores):
        if label.item() == 1 and score.item() > threshold:
            mask_np = mask[0].numpy()
            mask_total[mask_np > 0.5] = 255

    return mask_total