# This is a test script for the purpose of visualizing the segmentation of one image
import os

from service import *

if __name__ == '__main__':

    # Initialize the segmentation service
    service = SegmentationService()

    # Load a plate image (replace this to fit the path of targeted image)
    plate_image = cv2.imread(os.path.dirname(__file__)+"\\test\\1.jpg\\plate.png")

    # Segment the characters
    characters = service.segment(plate_image)

    # Visualize the segmented characters
    for idx, char_img in enumerate(characters):
        cv2.imshow(f"Character {idx}", char_img)
        cv2.waitKey(500)

    cv2.destroyAllWindows()
