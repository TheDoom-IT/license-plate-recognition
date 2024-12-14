import cv2
from dataclasses import dataclass


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


class PlateDetectionService:
    def __init__(self):
        # TODO: load model here
        pass

    def detect(self, img: cv2.typing.MatLike) -> tuple[cv2.typing.MatLike, Box]:
        # TODO: implement plate detection

        height, width, _ = img.shape
        return None, Box(
            int(width / 2),
            int(height / 2),
            int(width / 2 + 100),
            int(height / 2 + 100),
        )
