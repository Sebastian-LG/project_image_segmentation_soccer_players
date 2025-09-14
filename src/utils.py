import json, os
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
import cv2
import random, shutil

def load_dataset(images_dir, masks_dir, limit=None):
    images = []
    masks = []
    files = sorted(os.listdir(images_dir))

    for i, fname in enumerate(files):
        if fname.endswith((".jpg", ".png")):
            img_path = os.path.join(images_dir, fname)
            mask_path = os.path.join(masks_dir, os.path.splitext(fname)[0] + "_mask.png")

            if os.path.exists(mask_path):
                image = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                images.append(image)
                masks.append(mask)

        if limit and i >= limit - 1:
            break

    print(f"✅ Dataset cargado: {len(images)} pares imagen/máscara")
    return images, masks

import matplotlib.pyplot as plt

def visualize_sample(image, mask_gt, mask_pred, mask_post, idx=None):
    """
    Muestra imagen, máscara ground truth y predicción lado a lado.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image)
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(mask_gt, cmap="gray")
    axes[1].set_title("Máscara GT")
    axes[1].axis("off")

    axes[2].imshow(mask_pred, cmap="gray")
    axes[2].set_title("Predicción")
    axes[2].axis("off")

    axes[3].imshow(mask_post, cmap="gray")
    axes[3].set_title("PostProcesado")
    axes[3].axis("off")

    if idx is not None:
        fig.suptitle(f"Ejemplo {idx}", fontsize=14)

    plt.show()

def sample_coco(json_path, images_dir, out_dir, N=50, seed=42):
    random.seed(seed)
    with open(json_path, "r") as f:
        data = json.load(f)

    # Lista de todas las imágenes
    all_images = data["images"]
    print(f"Total de imágenes: {len(all_images)}")
    
    # Selección aleatoria
    sample_images = random.sample(all_images, N)
    sample_ids = {img["id"] for img in sample_images}

    # Filtrar anotaciones correspondientes
    sample_annotations = [ann for ann in data["annotations"] if ann["image_id"] in sample_ids]

    # Construir el JSON reducido
    sampled_data = {
        "licenses": data.get("licenses", []),
        "info": data.get("info", {}),
        "categories": data["categories"],
        "images": sample_images,
        "annotations": sample_annotations
    }

    # Crear carpetas de salida
    out_images = os.path.join(out_dir, "images")
    os.makedirs(out_images, exist_ok=True)

    # Copiar imágenes seleccionadas
    for img in sample_images:
        src = os.path.join(images_dir, img["file_name"])
        dst = os.path.join(out_images, img["file_name"])
        shutil.copy(src, dst)

    # Guardar JSON reducido
    out_json = os.path.join(out_dir, "annotations.json")
    with open(out_json, "w") as f:
        json.dump(sampled_data, f, indent=2)

    print(f"Muestreo completado ✅ {N} imágenes copiadas a {out_images}")
    print(f"JSON reducido guardado en {out_json}")

def fix_coco_ids(input_json, output_json):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reasignar IDs basados en el nombre de archivo
    id_map = {}
    for img in data["images"]:
        new_id = int(img["file_name"].split(".")[0])  # ej: 204.jpg -> 204
        id_map[img["id"]] = new_id
        img["id"] = new_id

    # Corregir las anotaciones
    for ann in data["annotations"]:
        if ann["image_id"] in id_map:
            ann["image_id"] = id_map[ann["image_id"]]

    # Guardar JSON corregido
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ JSON corregido guardado en {output_json}")

def coco_to_masks(annotations_file, images_dir, masks_dir):
    # Crear carpeta de salida si no existe
    os.makedirs(masks_dir, exist_ok=True)

    # Cargar anotaciones
    coco = COCO(annotations_file)

    # Crear mapa image_id -> file_name
    img_id_to_filename = {img["id"]: img["file_name"] for img in coco.dataset["images"]}

    print("Procesando imágenes y anotaciones...")
    for img in coco.dataset["images"]:
        img_id = img["id"]
        file_name = img["file_name"]

        # Obtener todas las anotaciones para esa imagen
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        if not anns:
            print(f"{file_name} → 0 anotaciones")
            continue

        # Crear máscara vacía
        mask = np.zeros((img["height"], img["width"]), dtype=np.uint8)

        # Dibujar cada anotación en la máscara
        for ann in anns:
            if "segmentation" in ann:
                rle = coco.annToRLE(ann)
                m = maskUtils.decode(rle)
                mask = np.maximum(mask, m * 255)

        # Guardar máscara
        mask_filename = os.path.splitext(file_name)[0] + "_mask.png"
        cv2.imwrite(os.path.join(masks_dir, mask_filename), mask)

        print(f"{file_name} → {len(anns)} anotaciones → máscara generada")

    print(f"✅ Máscaras guardadas en {masks_dir}")

# Ejemplo de uso:
# coco_to_masks("data/sample/annotations.json", "data/sample/images", "data/sample/masks")
