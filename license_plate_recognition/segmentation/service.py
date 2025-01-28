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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
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
        binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

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
        # if len(char_boxes) not in [6, 8]:
        new_char_boxes = []
        mean_height = np.mean(all_heights)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if abs(h - mean_height) <= 12:
                new_char_boxes.append((x, y, w, h))

        # Take the bboxes most aligned
        char_boxes = new_char_boxes if len(new_char_boxes) in [6, 8] or (
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

    @staticmethod
    def _find_skew(img):
        # Convert to grayscale if needed
        if len(img.shape) == 3:  # Color image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape

        # Preprocessing: reduce noise and enhance edges
        blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Smooth the image
        edges = cv2.Canny(blurred, 50, 150)  # Detect edges

        # Remove small connected components to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=w / 5.0,  # Dynamically adjusted based on width
            maxLineGap=h / 10.0  # Dynamically adjusted based on height
        )

        if lines is None or len(lines) == 0:
            return 0.0  # Fallback: No lines detected

        # Collect all angles of the detected lines
        angles = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)

        # Filter out extreme angles (e.g., noise)
        filtered_angles = [a for a in angles if -45 <= a <= 45]

        if len(filtered_angles) == 0:
            return 0.0  # Fallback: No valid angles detected

        # Calculate the dominant angle using a custom mode-like approach
        bins = np.arange(-45, 46, 1)  # 1-degree bins
        hist, _ = np.histogram(filtered_angles, bins=bins)
        dominant_angle_index = np.argmax(hist)
        dominant_angle = bins[dominant_angle_index]

        return dominant_angle

    @staticmethod
    def _rotate_pic(img, angle):
        # Get the center of the image
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # Compute the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the affine transformation (rotation)
        rotated = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    # Corrects the skew of an image by rotating it
    def deskew(self, img):
        # Find the skew angle and rotate the image accordingly
        return self._rotate_pic(img, self._find_skew(img))

    @staticmethod
    def normalize_plate(plate_img):
        """Normalize plate image to the range [0, 1]."""
        return plate_img / 255.0

    @staticmethod
    def _resize_with_padding(plate_img, target_width, target_height):
        """Resize plate while maintaining aspect ratio, add padding to match 4:1."""
        h, w = plate_img.shape[:2]
        aspect_ratio = target_width / target_height
        input_ratio = w / h

        if input_ratio > aspect_ratio:  # Plate is too wide
            new_w = target_width
            new_h = int(target_width / input_ratio)
        else:  # Plate is too tall
            new_h = target_height
            new_w = int(target_height * input_ratio)

        resized_plate = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Add padding to ensure final size matches the 800x200 target
        delta_w = target_width - new_w
        delta_h = target_height - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        padded_plate = cv2.copyMakeBorder(resized_plate, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded_plate

    @staticmethod
    def _sharpen_plate(plate_img):
        """Sharpen plate image using an unsharp mask."""
        gaussian_blurred = cv2.GaussianBlur(plate_img, (5, 5), 0)
        sharpened = cv2.addWeighted(plate_img, 1.5, gaussian_blurred, -0.5, 0)
        return sharpened

    def _process_plate(self, plate_img, target_width, target_height):
        """Process license plate image to resize, sharpen and pad."""

        # Resize and pad to maintain 4:1 aspect ratio
        resized_plate = self._resize_with_padding(plate_img, target_width, target_height)

        # Sharpen
        sharpened_plate = self._sharpen_plate(resized_plate)

        return sharpened_plate

    def segment(self, plate) -> list:
        """
        Segments the input plate image and returns a list of images,
        each containing a single character.
        """
        target_width = 800 # 4:1 aspect ratio
        target_height = 200 # 4:1 aspect ratio

        # Step 1: Process license plate image to resize, sharpen, and pad
        processed_plate = self._process_plate(plate, target_width, target_height)

        # Step 2: Correct the skew in the plate image
        fixed_plate = self.deskew(processed_plate)

        # Step 3: Process binary contours and return the contours, the binary image as well as the updated fixed_plate
        fixed_plate, binary, contours = self._process_binary_contours(fixed_plate)

        # Step 4: Extract bounding boxes of characters
        char_boxes = self._extract_char_boxes(plate, contours)

        # Step 5: Extract and return individual character images
        characters = self._extract_char_images(binary, char_boxes)

        # Return a list of images, each containing a single character
        return characters
