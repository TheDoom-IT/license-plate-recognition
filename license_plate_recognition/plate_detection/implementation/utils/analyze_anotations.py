import cv2
import os
import argparse

IMAGE_SIZE = 416
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def draw_bbox(image, annotations):
    """
    Draw bounding boxes on the image based on annotations.

    :param image: The image to draw annotations.
    :param annotations: A list of annotations.
    :return: Image with bounding boxes drawn.
    """
    for annotation in annotations:

        class_id, x_center, y_center, bbox_width, bbox_height = map(float, annotation.split())
        
        x1 = int((x_center - bbox_width / 2) * IMAGE_SIZE) 
        y1 = int((y_center - bbox_height / 2) * IMAGE_SIZE)
        x2 = int((x_center + bbox_width / 2) * IMAGE_SIZE)
        y2 = int((y_center + bbox_height / 2) * IMAGE_SIZE)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Class {int(class_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image

def main(path):
    """
    Display the image with bounding box from dataset given the source path.

    :param path: Path to the train.txt file.
    """    
    with open(path, 'r') as f:
        image_paths = f.read().strip().split('\n')

    for selected_image_path in image_paths[132:]:
        selected_image_path = selected_image_path.replace("dataset", ".dataset")
        selected_image_path = selected_image_path.replace("/", "\\")
        base_path = os.path.splitext(selected_image_path)[0]
        annotation_path = f"{base_path}.txt"

        image = cv2.imread(os.path.join(BASE_DIR, selected_image_path))
        if image is None:
            print(f"Failed to load image: {selected_image_path}")
            continue

        try:
            with open(os.path.join(BASE_DIR, annotation_path), 'r') as f:
                annotations = f.read().strip().split('\n')
        except FileNotFoundError:
            print(f"Annotation Error: {selected_image_path}")
            continue

        print(f"Showing image: {selected_image_path}")
        image_with_bbox = draw_bbox(image, annotations)

        # Display the image with bounding boxes
        cv2.imshow("YOLOv3 Annotation Check", image_with_bbox)
        
        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis of our correct annotations")
    parser.add_argument("path", type=str, help="Path to entry file. (train.txt, test.txt, valid.txt, etc)")

    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise ValueError("Could not find the entry file.")
    
    main(args.path)
