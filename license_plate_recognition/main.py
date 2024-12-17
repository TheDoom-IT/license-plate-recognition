import sys
import os.path
import cv2
from .recognition.service import RecognitionService
from .plate_detection.service import PlateDetectionService, Box
from .segmentation.service import SegmentationService


def read_file(filename: str) -> cv2.typing.MatLike:
    if not os.path.isfile(filename):
        print(f"File '{filename}' does not exist")
        sys.exit(1)

    return cv2.imread(filename, cv2.IMREAD_COLOR)


def show_result(img: cv2.typing.MatLike, box: Box, plate_number: str):
    cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), (0, 0, 255), 2)
    cv2.putText(img, plate_number, (box.x1, box.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imshow("License Plate Recognition", img)
    cv2.waitKey(0)


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
    plate_number = recognition_service.recognize(characters)

    show_result(img, box, plate_number)
