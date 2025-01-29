import subprocess
import os
import keras
import platform
from PIL import Image

import shutil
import json
import numpy as np
from dataclasses import dataclass
from plate_detection.blocks import YoloV3
from plate_detection.yolo.const import NUM_CLASSES, YOLO_LAYERS
from plate_detection.yolo.utils import load_yolo_weights, load_image_as_tf, get_bboxes, get_original_bbox


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float


class PlateDetectionService:
    def __init__(self, use_our_implementation: bool = False):
        self.model = YoloV3(num_classes=NUM_CLASSES)()
        model_base_path = os.path.join(BASE_DIR, ".weights")

        if use_our_implementation:
            # weights trained on custom YOLOv3 implementation in Python
            model_path = os.path.join(model_base_path, "yolov3-our-implementation.weights.h5")
            self.model.load_weights(model_path)
        else:
            # weights trained by darknet
            load_yolo_weights(os.path.join(model_base_path, "yolov3-custom_final.weights"), self.model, YOLO_LAYERS)

    def detect(self, image) -> tuple[np.array, Box]:
        scale, _image = load_image_as_tf(image)
        output = self.model(_image)

        boxes, _class, conf = get_bboxes(output)
        x1, y1, x2, y2 = get_original_bbox(scale, boxes[0])

        plate = image.crop((x1, y1, x2, y2))
        plate = np.array(plate)

        return plate, Box(x1, y1, x2, y2, float(conf[0]))
