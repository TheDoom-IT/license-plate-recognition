import os
import random
import shutil

import xml.etree.ElementTree as ET
import uuid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_BASE_PATH = ".dataset"

def read_annotation_xml(data_path):
    tree = ET.parse(data_path)
    root = tree.getroot()

    childs = [child for child in root]
    images = filter(lambda x: x.tag == "image", childs)

    return list(images)


def create_metadata(image, dst, filename):
    lines = []
    for box in image:
        xtl = float(box.attrib.get("xtl"))
        ytl = float(box.attrib.get("ytl"))
        xbr = float(box.attrib.get("xbr"))
        ybr = float(box.attrib.get("ybr"))

        w = xbr - xtl
        h = ybr - ytl

        x_center = xtl + w/2
        y_center = ytl + h/2

        width, height = float(image.attrib["width"]), float(image.attrib["height"])
        x_center = x_center / width
        y_center = y_center / height
        w = w / width
        h = h / height

        lines.append(f"0 {x_center} {y_center} {w} {h}")

    with open(os.path.join(dst, f"{filename}.txt"), "w") as f:
        f.write("\n".join(lines))


def copy_img(src, dst, images):
    file_names = []

    for image in images:
        file_name = uuid.uuid4().hex
        src_img_path = os.path.join(src, image.attrib["name"])
        dst_img_path = os.path.join(dst, f"{file_name}.jpg")
        
        shutil.copy(src_img_path, dst_img_path)
        create_metadata(image, dst, file_name)
        file_names.append((file_name, image))

    return file_names


def store_labels(path, files, dtype):
    with open(os.path.join(path, f"labels-{dtype}.txt"), "w") as f:
        for name, img in files:
            for box in img:
                attribute = [attr for attr in box][0]
        
            f.write(f"{name}.jpg {attribute.text}\n")


def convert_coco_to_darknet(path, dst, anotation, src_image):
    image_elts = read_annotation_xml(os.path.join(path, anotation))
    random.shuffle(image_elts)

    size = len(image_elts)
    
    # 70% of dataset to training set
    train = image_elts[:int(size * 0.7)]
    # 20% of dataset to validation set
    val = image_elts[int(size * 0.7): int(size * 0.9)]
    # 10% of dataset to training set
    test = image_elts[int(size * 0.9):]

    file_names = copy_img(os.path.join(path, src_image), os.path.join(dst, "train"),train)
    store_labels(path, file_names, dtype="train")

    file_names = copy_img(os.path.join(path, src_image), os.path.join(dst, "valid"),val)
    store_labels(path, file_names, dtype="valid")

    file_names = copy_img(os.path.join(path, src_image), os.path.join(dst, "test"),test)
    store_labels(path, file_names, dtype="test")


