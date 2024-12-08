import os
import argparse

from PIL import Image, ImageDraw


DATASET_PARTS = ["train", "valid", "test"]


def load_image(filepath) -> Image.Image:
    """
    Load an image from the specified file.

    :param filepath: Path to the image file.
    :return: Loaded image.
    """
    return Image.open(filepath)

def process_metadata(filepath):
    """
    Process metadata from the corresponding .txt file.

    :param filepath: Path to the image file.
    :return: List of metadata strings.
    """
    boxes = []
    meta = filepath.split(".jpg")[0]
    with open(f"{meta}.txt", 'r') as f:
        for line in f:
            boxes.append(line.strip())
    return boxes

def update_metadata(filepath, metadata):
     """
    Update the metadata file corresponding to the image.

    :param filepath: Path to the image file.
    :param metadata: List of updated metadata.
    """
     meta = filepath.split(".jpg")[0]
     with open(f"{meta}.txt", 'w') as f:
        f.write("\n".join(metadata))
    

def resize_image(filepath):
    """
    Resize the image and adjust bounding boxes to fit the new size.

    :param filepath: Path to the image file.
    :return: Resized image and updated metadata.
    """
    image = load_image(filepath)
    width, height = image.size
    cropped = image.copy()
    f_data = process_metadata(filepath)
    new_metadata = []

    x_min, y_min, x_max, y_max = 0, 0, 0, 0
    x_diff, y_diff = 0, 0
    for metadata in f_data:
        x, y, w, h = map(float, metadata.split(" ")[1:])
        _x, _y, _w, _h = x*width, y*height, w*width, h*height

        if x_min == x_max:
            x_min, y_min, x_max, y_max = _x - _w/2, _y - _h/2, _x + _w/2, _y + _h/2 
        else:
            x_min, y_min, x_max, y_max = min(x_min, _x), min(y_min, _y), max(x_max, _x), max(y_max, _y)

    diff = abs(width - height)
    if width > height:
        if x_min < diff/2:
            if x_max > width - diff:
                raise ValueError(f"Can't crop the image both on left and right side: {filepath}")
            else:
                cropped = image.crop((0, 0, width-diff, height))
                width -= diff
        elif x_max > width - diff/2:
            if x_min < diff:
                raise ValueError(f"Can't crop the image both on left and right side: {filepath}")
            else:
                x_diff = -diff
                cropped = image.crop((diff, 0, width, height))
                width -= diff
        else:
            x_diff = -diff/2
            cropped = image.crop((diff/2, 0, width - diff/2, height))
            width -= diff
    elif height > width:
        if y_min < diff/2:
            if y_max > height - diff:
                raise ValueError(f"Can't crop the image both on top and bottom side: {filepath}")
            else:
                cropped = image.crop((0, 0, width, height-diff))
                height -= diff
        elif y_max > height - diff/2:
            if y_min < diff:
                raise ValueError(f"Can't crop the image both on top and bottom side: {filepath}")
            else:
                y_diff = -diff
                cropped = image.crop((0, diff, width, height))
                height -= diff
        else:
            y_diff = -diff/2
            cropped = image.crop((0, diff/2, width, height - diff/2))
            height -= diff

    h_convertor = 1/height
    w_convertor = 1/width

    for metadata in f_data:
        ow, oh = image.size
        c, x, y, w, h = map(float, metadata.split(" "))
        _x, _y, _w, _h = x*ow, y*oh, w*ow, h*oh

        _x += x_diff
        _y += y_diff

        _x *= w_convertor
        _y *= h_convertor   
        _w *= w_convertor
        _h *= h_convertor
        
        new_metadata.append(" ".join(map(str, [c, _x, _y, _w, _h])))

    resized = cropped.resize((416, 416))
    
    return resized, new_metadata
        

def resize_dataset(dataset_path):
    """
    Resize all images in the dataset and update the corresponding metadata.

    :param dataset_path: Path to the dataset folder.
    """
    for country in os.listdir(dataset_path):
        for part in DATASET_PARTS:
            if not os.path.exists(os.path.join(dataset_path, country, part)):
                continue
            
            for image_path in os.listdir(os.path.join(dataset_path, country, part)):
                if image_path.endswith(".jpg"):
                    im_path = os.path.join(dataset_path, country, part, image_path)
                    resized_image, metadata = resize_image(im_path)
                    
                    resized_image.save(im_path)
                    update_metadata(im_path, metadata)
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize image to 416x416.")
    parser.add_argument("dir", type=str, help="Directory of dataset.")

    args = parser.parse_args()
    resize_dataset(args.dir)
