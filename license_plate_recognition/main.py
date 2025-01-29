import sys
import os.path
import cv2
import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageFile
from .recognition.service import RecognitionService
from .plate_detection.service import PlateDetectionService, Box
from .segmentation.service import SegmentationService


def read_file(filename: str) -> ImageFile:
    if not os.path.isfile(filename):
        print(f"File '{filename}' does not exist")
        sys.exit(1)

    image = Image.open(filename)
    # convert to RGB to make sure that only 3 layers are used (PNG has 4 layers)
    # plate detection part requires 3 layers only
    return image.convert('RGB')


def show_result(img: cv2.typing.MatLike, box: Box, plate: np.ndarray, characters: list[np.ndarray], plate_number: str):
    gs = GridSpec(3, len(characters))

    fig = plt.figure()
    fig.suptitle(plate_number, fontsize=16)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(img)

    ax1.add_patch(plt.Rectangle((box.x1, box.y1), box.x2 - box.x1, box.y2 - box.y1, fill=False, edgecolor='red', lw=2))

    ax2 = fig.add_subplot(gs[1, :])
    ax2.imshow(plate)

    for index, character in enumerate(characters):
        ax = fig.add_subplot(gs[2, index])
        ax.imshow(character, cmap='gray', interpolation='nearest')

    plt.tight_layout()
    plt.show()


def handle_args(params: list[str]) -> [str, bool]:
    return sys.argv[1], "--our-implementation" in params


HELP = """
Usage: python main.py <filename>

Options:
    --our-implementation    Use YOLOv3 weights trained on custom YOLOv3 implementation in Python.
                            By default, YOLOv3 weights trained by darknet are used.
"""

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(HELP)
        sys.exit(1)

    filename, use_our_implementation = handle_args(sys.argv)

    img = read_file(filename)

    plate_detection_service = PlateDetectionService(use_our_implementation)
    segmentation_service = SegmentationService()
    recognition_service = RecognitionService()

    plate, box = plate_detection_service.detect(img)
    characters = segmentation_service.segment(plate)
    if len(characters) == 0:
        print("No characters found")
        sys.exit(1)
    plate_number = recognition_service.recognize(characters)

    show_result(img, box, plate, characters, plate_number)
