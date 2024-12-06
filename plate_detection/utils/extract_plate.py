import os

import cv2
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_BASE_PATH = ".dataset"
DATASETS_PATH = ("polish",)
DATASET_SECTIONS = ("train", "valid", "test")


def extract_image_boundary(meta_path) -> np.array:
    boundaries = []
    with open(meta_path, "r") as f:
        for line in f:
            boundaries.append(line.strip().split(" ")[1:])

    return np.array(boundaries, dtype=np.float32)

def poly_cord2bbox(cords, image):
    """
    Convert polynomial coordinates to bounding box coordinates
    """

    x_center, y_center, width, height  = cords
    x = x_center - width/2
    y = y_center - height/2

    x = x * image.width
    y = y * image.height
    w = width * image.width
    h = height * image.height

    return round(x), round(y), round(w), round(h)


def extract_plate_from_dataset():
    for dataset_path in DATASETS_PATH:
        for section in DATASET_SECTIONS:
            data_path = os.path.join(BASE_DIR, DATASET_BASE_PATH, dataset_path, section)
            for img_path in os.listdir(data_path):
                if not img_path.endswith(".jpg"):
                    continue
                
                boundaries = extract_image_boundary(os.path.join(data_path, f'{img_path.split(".jpg")[0]}.txt'))
                for boundary in boundaries:
                    image = Image.open(os.path.join(data_path, img_path))

                    box = poly_cord2bbox(boundary, image)
                    roi = np.array(image)[box[1]: box[1] +box[3], box[0]:box[0] + box[2]]
                    roi_image = Image.fromarray(roi)
                    roi_image.save(os.path.join(BASE_DIR, DATASET_BASE_PATH, "samples", section, img_path))


extract_plate_from_dataset()