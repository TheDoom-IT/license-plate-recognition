# !pip install opencv-python

# !pip install imutils

import os
import cv2
import numpy as np

from bs4 import BeautifulSoup


# Read annotations
current_path = os.path.dirname(__file__)
dataset_path = current_path+"\\dataset"
with open(dataset_path+"\\annotations.xml", "r", encoding="utf-8") as file:
    xml_content = file.read()

# Analyse XML content
soup = BeautifulSoup(xml_content, "xml")

# Initialiser une structure pour stocker les résultats
bbox_data = []

# Parcourir chaque élément <image>
for image in soup.find_all("image"):
    image_id = image.get("id")
    image_name = image.get("name")
    width = image.get("width")
    height = image.get("height")

    # Parcourir les <box> associés à cette image
    for box in image.find_all("box"):
        label = box.get("label")
        source = box.get("source")
        occluded = box.get("occluded")
        z_order = box.get("z_order")
        xtl = float(box.get("xtl"))
        ytl = float(box.get("ytl"))
        xbr = float(box.get("xbr"))
        ybr = float(box.get("ybr"))

        # Ajouter les données à la structure
        bbox_data.append({
            "image_id": image_id,
            "image_name": image_name,
            "width": int(width),
            "height": int(height),
            "label": label,
            "source": source,
            "occluded": bool(int(occluded)),
            "z_order": int(z_order),
            "xtl": xtl,
            "ytl": ytl,
            "xbr": xbr,
            "ybr": ybr
        })


# Afficher les données extraites
for bbox in bbox_data:
    print(bbox)

# Directory helper function
def delete_files_in_directory(directory_path):
    print("Proceed with files deletion.")
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError:
        print("Error occurred while deleting files.")

# Image processing functions

def read_image(PROJECT_DATA_PATH, image_name):
    return cv2.imread(f"{PROJECT_DATA_PATH}\\{image_name}")  # Replace with your plate extraction code if needed


def extract_plate(original, xtl, ytl, xbr, ybr):
    # Get corners coordinates of plate
    # corners_1 = [(1318.15, 537.20), (2015.94, 704.15)]
    corners_1 = [(xtl, ytl), (xbr, ybr)]
    top_left = corners_1[0]
    top_left = tuple(int(num) for num in top_left)
    bottom_right = corners_1[1]
    bottom_right = tuple(int(num) for num in bottom_right)
    plate = original[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return plate


def apply_closing_opening(binary):
    # Reduce small noise with opening
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Reconstruct characters with closing
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    # Dilate a bit
    binary = cv2.dilate(binary, kernel, iterations=1)
    # Then close
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary


def process_binary_contours(plate):
    # Get greyscale of the image
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # plate_gray = cv2.equalizeHist(plate_gray) # try equalization

    # Increase the contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    plate_gray = clahe.apply(plate_gray)

    # Apply Gaussian blur to reduce noise
    plate_gray = cv2.GaussianBlur(plate_gray, (5, 5), 0)
    # plate_gray = cv2.medianBlur(plate_gray, 7)

    # Find the optimal threshold using Otsu algorithm
    _, binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)  # Invert colors: characters white, background black

    # binary = cv2.adaptiveThreshold(
    #     plate_gray, 255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY_INV,
    #     11, 2
    # )

    # Apply closing and opening to reduce noise
    binary = apply_closing_opening(binary)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Add black border so the plate is not stick to the border
    binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Update the original image accordingly
    plate = cv2.copyMakeBorder(plate, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Reapply closing and opening
    binary = apply_closing_opening(binary)

    # Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return plate, binary, contours

def extract_char_boxes(plate, contours):
    # Filter and extract characters
    char_boxes = []
    all_boxes = []
    all_heights = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        all_boxes.append((x, y, w, h))
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
    char_boxes = new_char_boxes if len(new_char_boxes) in [6, 8] or (len(new_char_boxes) > len(char_boxes) and len(new_char_boxes) <= 8) else char_boxes

    # Sort bounding boxes from left to right
    char_boxes = sorted(char_boxes, key=lambda b: b[0])
    return char_boxes, all_boxes


def save_contours_rectangles_plate(plate, char_boxes, all_boxes=None):
    for x, y, w, h in char_boxes:
        cv2.rectangle(plate, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.imshow("Contours", plate)
    # cv2.waitKey(0)
    cv2.imwrite(f'contours.png', plate)

    # Optionnally save all contours for debugging
    if all_boxes:
        copy_plate = plate.copy()
        for x, y, w, h in all_boxes:
            cv2.rectangle(plate, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(f'all_contours.png', plate)


def extract_save_single_characters(binary, char_boxes):
    # Extract and save characters
    characters = []
    for i, (x, y, w, h) in enumerate(char_boxes):
        char_roi = binary[y:y+h, x:x+w]
        characters.append(char_roi)
        # cv2.imshow("char_roi", char_roi)
        # cv2.waitKey(500)
        cv2.imwrite(f'ROI_{i}.png', char_roi)

    # Optionally visualize segmented characters
    # for char in characters:
    #     cv2.imshow("Character", char)
    #     cv2.waitKey(500)


def find_skew(img):
    # Convert to grayscale if needed
    if len(img.shape) == 3:  # Color image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape

    # Preprocessing: reduce noise and enhance edges
    blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Smooth the image
    edges = cv2.Canny(blurred, 50, 150)         # Detect edges

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
        maxLineGap=h / 10.0    # Dynamically adjusted based on height
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


def rotate_pic(img, angle):
    # Get the center of the image
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the affine transformation (rotation)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# Corrects the skew of an image by rotating it
def deskew(img):
    # Find the skew angle and rotate the image accordingly
    return rotate_pic(img, find_skew(img))


def normalize_plate(plate_img):
    """Normalize plate image to the range [0, 1]."""
    return plate_img / 255.0


def resize_with_padding(plate_img, target_width, target_height):
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


def sharpen_plate(plate_img):
    """Sharpen plate image using an unsharp mask."""
    gaussian_blurred = cv2.GaussianBlur(plate_img, (5, 5), 0)
    sharpened = cv2.addWeighted(plate_img, 1.5, gaussian_blurred, -0.5, 0)
    return sharpened


def process_plate(plate_img, target_width, target_height):
    """Process license plate image to resize, sharpen and pad."""

    # Resize and pad to maintain 4:1 aspect ratio
    resized_plate = resize_with_padding(plate_img, target_width, target_height)

    # Sharpen
    sharpened_plate = sharpen_plate(resized_plate)

    return sharpened_plate


# Process extraction on all images

PROJECT_DATA_PATH = dataset_path+"\\photos"
save_dir = dataset_path+"\\extracted_characters\\test_extraction"
target_width = 800
target_height = 200

for plate_bbox in bbox_data:
    image_name = plate_bbox['image_name']
    original_img = read_image(PROJECT_DATA_PATH, image_name)

    # Extract the plate according to specific bounding boxes information provided
    plate = extract_plate(original_img, plate_bbox['xtl'], plate_bbox['ytl'], plate_bbox['xbr'], plate_bbox['ybr'])

    # Process license plate image to resize, sharpen, and pad
    processed_plate = process_plate(plate, target_width, target_height)

    # Correct the skew in the plate image
    fixed_plate = deskew(processed_plate)

    # Process binary contours and return the contours, the binary image as well as the updated fixed_plate
    fixed_plate, binary, contours = process_binary_contours(fixed_plate)

    if not os.path.exists(f"{save_dir}\\{image_name}"):
        os.makedirs(f"{save_dir}\\{image_name}")
    else:
        print("Folder already created, refreshing the content")
        delete_files_in_directory(f"{save_dir}\\{image_name}")
    os.chdir(f"{save_dir}\\{image_name}")
    cv2.imwrite(f'binary.png', binary)
    cv2.imwrite(f'processed_plate.png', processed_plate)
    cv2.imwrite(f'fixed_skew.png', fixed_plate)

    char_boxes, all_boxes = extract_char_boxes(fixed_plate, contours)

    os.chdir(f"{save_dir}\\{image_name}")
    save_contours_rectangles_plate(fixed_plate, char_boxes, all_boxes)

    extract_save_single_characters(binary, char_boxes)
    print(f"Processed file {image_name}")

    cv2.destroyAllWindows()

    '''
    TODO:
    - try opening and closing properly to remove scratches
    - try rotation of image to get all contours aligned horizontally and remove extra noise
    '''