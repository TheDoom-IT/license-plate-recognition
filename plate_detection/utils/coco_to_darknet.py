import os
import json
import argparse

import uuid
import shutil


def coco_to_darknet(src, dst):
    data_types = ["train", "test", "valid"]
    for dtype in data_types: 
        src_path = os.path.join(src, dtype) 
        dst_path = os.path.join(dst, dtype)
        if dtype == "valid" and not os.path.exists(src_path):
            continue
        
        with open(os.path.join(src_path, "_annotations.coco.json")) as f:  
            metadata = json.load(f)

        for image, annotation in zip(metadata["images"], metadata["annotations"]):
            new_file_name = uuid.uuid4().hex
            shutil.copy(os.path.join(src_path, image["file_name"]), os.path.join(dst_path, f"{new_file_name}.jpg"))
            with open(os.path.join(dst_path, f"{new_file_name}.txt"), "w") as f:
                x, y, w, h = annotation["bbox"]

                x = ((x+w)/2)/image["width"]
                y = ((y+h)/2)/image["height"]
                w = w/image["width"]
                h = h/image["height"]

                f.write(f"{annotation['category_id']} {x} {y} {w} {h}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert coco to darknet dataset.")
    parser.add_argument("src_dir", type=str, help="Source directory of the coco dataset.")
    parser.add_argument("dst_dir", type=str, help="Destination directory to save to darknet dataset.")


    args = parser.parse_args()
    coco_to_darknet(args.src_dir, args.dst_dir)


if __name__ == "__main__":
    main()