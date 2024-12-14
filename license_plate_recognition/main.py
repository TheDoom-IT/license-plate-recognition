import sys
import os.path
import cv2
from .recognition.service import RecognitionService
from .plate_detection.service import PlateDetectionService
from .segmentation.service import SegmentationService


def read_file(filename: str) -> cv2.typing.MatLike:
    if not os.path.isfile(filename):
        print(f"File '{filename}' does not exist")
        sys.exit(1)

    return cv2.imread(filename, cv2.IMREAD_COLOR)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    img = read_file(filename)

    plate_detection_service = PlateDetectionService()
    segmentation_service = SegmentationService()
    recognition_service = RecognitionService()

    plate = plate_detection_service.detect(img)
    characters = segmentation_service.segment(plate)
    result = recognition_service.recognize(characters)

    print(f"The image contains the following license plate: {result}")
