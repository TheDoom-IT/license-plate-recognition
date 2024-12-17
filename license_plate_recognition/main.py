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

    return Image.open(filename)


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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    img = read_file(filename)

    plate_detection_service = PlateDetectionService(filename)
    segmentation_service = SegmentationService()
    recognition_service = RecognitionService()

    plate, box = plate_detection_service.detect()
    characters = segmentation_service.segment(plate)
    if len(characters) == 0:
        print("No characters found")
        sys.exit(1)
    plate_number = recognition_service.recognize(characters)

    show_result(img, box, plate, characters, plate_number)
