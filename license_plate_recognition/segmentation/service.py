import cv2
import numpy as np

class SegmentationService:
    # Implements the class for the segmentation module
    def __init__(self):
        # Initialize the segmentation service, e.g., load pre-trained models here if needed
        pass

    @staticmethod
    def _apply_closing_opening(binary):
        # Reduce small noise with opening and reconstruct characters with closing
        # Reduce small noise with opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        # Reconstruct characters with closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        # Dilate a bit
        binary = cv2.dilate(binary, kernel, iterations=1)
        # Then close
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary

    def _process_binary_contours(self, plate):
        # Convert to grayscale
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        plate_gray = clahe.apply(plate_gray)

        # Apply Gaussian blur to reduce noise
        plate_gray = cv2.GaussianBlur(plate_gray, (5, 5), 0)

        # Find binary image using Otsu's thresholding
        _, binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)  # Invert colors: characters white, background black

        # binary = cv2.adaptiveThreshold(
        #     plate_gray, 255,
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY_INV,
        #     11, 2
        # )

        # Apply morphology operations to clean noise
        binary = self._apply_closing_opening(binary)
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Add black border so the plate is not stick to the border
        binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

        # Update the original image accordingly
        plate = cv2.copyMakeBorder(plate, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Reapply closing and opening
        binary = self._apply_closing_opening(binary)

        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return plate, binary, contours

    @staticmethod
    def _extract_char_boxes(plate, contours):
        # Filter and extract characters
        char_boxes = []
        all_heights = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            # if 0.2 < aspect_ratio < 1.0 and h > 0.5 * plate.shape[0]:  # Adjust size and ratio filters
            # if 0.1 < aspect_ratio < 1.2 and h > 0.5 * plate.shape[0]: # Increase boundaries to include narrow characters like "I" and "1"
            # if 0.1 < aspect_ratio < 1.2 and h > 0.3 * plate.shape[0] and w > 0.05 * plate.shape[1]:
            # if 0.1 < aspect_ratio < 1.0 and h > 0.5 * plate.shape[0] and w > 0.06 * plate.shape[1]: ##
            if 0.2 < aspect_ratio < 1.0 and h > 0.3 * plate.shape[0] and w > 0.05 * plate.shape[1]:
                char_boxes.append((x, y, w, h))
                all_heights.append(h)

        # Align bboxes to have roughly the same height
        # if len(char_boxes) not in [7, 8]:
        new_char_boxes = []
        mean_height = np.mean(all_heights)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if abs(h - mean_height) <= 12:
                new_char_boxes.append((x, y, w, h))

        # Take the bboxes most aligned
        char_boxes = new_char_boxes if len(new_char_boxes) in [7, 8] or (
                    len(new_char_boxes) > len(char_boxes) and len(new_char_boxes) <= 8) else char_boxes

        # Sort bounding boxes from left to right
        char_boxes = sorted(char_boxes, key=lambda b: b[0])
        return char_boxes

    @staticmethod
    def _extract_char_images(binary, char_boxes):
        characters = []
        for (x, y, w, h) in char_boxes:
            char_roi = binary[y:y + h, x:x + w]
            characters.append(char_roi)
        return characters

    def segment(self, plate) -> list:
        """
        Segments the input plate image and returns a list of images,
        each containing a single character.
        """
        # Step 1: Process the plate and find contours
        plate, binary, contours = self._process_binary_contours(plate)

        # Step 2: Extract bounding boxes of characters
        char_boxes = self._extract_char_boxes(plate, contours)

        # Step 3: Extract and return individual character images
        characters = self._extract_char_images(binary, char_boxes)

        # Return a list of images, each containing a single character
        return characters
