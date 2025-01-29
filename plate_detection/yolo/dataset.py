import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageFile
from plate_detection.yolo.utils import calculate_area_iou
from plate_detection.yolo.const import IMAGE_SIZE

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YoloDataset:
    def __init__(self, obj_data_file, anchors, anchor_mask, image_size=416, S=[13, 26, 52], data_type="train"):
        self.read_config(obj_data_file)
        self.image_size = image_size
        self.S = S
        self.anchors = np.concatenate([
            np.array(anchors)[anchor_mask[idx]] / image_size
            for idx in range(len(S))
        ], dtype=np.float32)
        self.num_anchors = self.anchors.shape[0]
        self.ignore_iou_thresh = 0.5
        self.num_anchors_per_scale = 3
        self.data_type = data_type

    
    def read_config(self, path):
        if not os.path.exists(path):
            raise ValueError("obj_data_file does not exist.")
        
        with open(path, 'r') as f:
            config = self.load_config(f.read())
        
        assert os.path.exists(os.path.join(config["base_dir"], config["train"]))
        assert os.path.exists(config["backup"])
        if config.get("valid", None):
            assert os.path.exists(os.path.join(config["base_dir"], config["valid"]))
            self.has_valid = True
        else:
            self.has_valid = False
        
        self.config = config

    def load_config(self, data):
        config = {}
        for line in data.split('\n'):
            key, value = line.split('=')
            config[key.strip()] = value.strip()

        return config

    def get_image(self, image_path):
        # To be removed
        image_path = "."+image_path
        return np.array(Image.open(os.path.join(self.config["base_dir"], image_path)).convert("RGB"), dtype=np.float32) /  255

    def get_label(self, label_path):
        # To be removed
        label_path = "."+label_path
        with open(os.path.join(self.config['base_dir'], label_path.replace(".jpg", ".txt"))) as file:
            if os.stat(file.fileno()).st_size:
                label = np.loadtxt(file, delimiter=" ")
            else:
                return None
        
        if label.ndim == 1 and label.shape[0] != 0:
            return label.reshape((1, label.shape[0]))
        else:
            return None
    
    def read_txt(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        lines = [line.strip() for line in lines]
        return lines
    
    def get_item(self, path):
        image = self.get_image(path)
        bboxes = self.get_label(path)
        if bboxes is not None and bboxes.shape[0] != 0:
            bboxes = np.roll(bboxes, 4, axis=1)
        else:
            bboxes =[]

        targets = [np.zeros((S, S, self.num_anchors // 3, 6), dtype=np.float32) for S in self.S]

        for box in bboxes:
            iou = calculate_area_iou(np.array(box[2:4]), self.anchors).numpy()
            anchors_indices = iou.argsort(axis=0)[::-1]
            x, y, w, h, _class = box
            has_anchors = [False, False, False]

            for anchor_idx in anchors_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S*x), int(S*y)

                anchor_taken = targets[scale_idx][j, i, anchor_on_scale,  4]
                if not anchor_taken and not has_anchors[scale_idx]:
                    targets[scale_idx][j, i, anchor_on_scale, 4] = 1
                    x_cell, y_cell = S*x - i, S*y - j
                    width_cell, height_cell = (w, h)

                    box_coords = np.array([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][j, i, anchor_on_scale, :4] = box_coords 
                    targets[scale_idx][ j, i, anchor_on_scale, 5] = _class 

                elif not anchor_taken and iou[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][ j, i, anchor_on_scale, 4] = -1
                    
        return image, tuple(targets)
    
    def load_image_and_labels(self):
        if self.data_type not in ["train", "test", "valid"]:
            raise ValueError("data_type should be 'train', 'test', or 'valid'.")
        
        img_paths = self.read_txt(os.path.join(self.config['base_dir'], self.config[self.data_type]))
        
        for path in img_paths:
            img, outputs = self.get_item(path)
    
            yield img, (
                tf.convert_to_tensor(outputs[0]),
                tf.convert_to_tensor(outputs[1]),
                tf.convert_to_tensor(outputs[2]),
            )

    def __call__(self):        
        return tf.data.Dataset.from_generator(
            self.load_image_and_labels, 
            output_signature=(
                tf.TensorSpec(shape=(self.image_size, self.image_size, 3), dtype=tf.float32), 
                (
                    tf.TensorSpec(shape=( self.S[0], self.S[0], self.num_anchors // 3, 6), dtype=tf.float32), 
                    tf.TensorSpec(shape=( self.S[1], self.S[1], self.num_anchors // 3, 6), dtype=tf.float32), 
                    tf.TensorSpec(shape=( self.S[2], self.S[2], self.num_anchors // 3, 6), dtype=tf.float32),
                )
            )
        )
