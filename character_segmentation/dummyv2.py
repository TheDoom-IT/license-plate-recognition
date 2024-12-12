import cv2
import numpy as np

# Read the plate image
PROJECT_DATA_PATH = "C:\\Cours\\E5\\IASR\\project_files\\dataset\\photos\\"
file1 = "1.jpg"
original = cv2.imread(PROJECT_DATA_PATH+file1)  # Replace with your plate extraction code if needed
nom_fenetre = "original1"
cv2.imshow(nom_fenetre, original)

cv2.waitKey(0)

# Get corners coordinates of plate
corners_1 = [(1318.15, 537.20), (2015.94, 704.15)]
top_left = corners_1[0]
top_left = tuple(int(num) for num in top_left)
bottom_right = corners_1[1]
bottom_right = tuple(int(num) for num in bottom_right)


# Show extracted plate image
plate = original[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
print("Extracted image")
cv2.imshow("plate", plate)
cv2.waitKey(0)

plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

# Preprocess the image
_, binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary = cv2.bitwise_not(binary)  # Invert colors: characters white, background black

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter and extract characters
char_boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    if 0.2 < aspect_ratio < 1.0 and h > 0.5 * plate.shape[0]:  # Adjust size and ratio filters
        char_boxes.append((x, y, w, h))

# Sort bounding boxes from left to right
char_boxes = sorted(char_boxes, key=lambda b: b[0])

# Extract and save characters
characters = []
for i, (x, y, w, h) in enumerate(char_boxes):
    char_roi = binary[y:y+h, x:x+w]
    characters.append(char_roi)
    cv2.imwrite(f'/mnt/data/char_{i}.png', char_roi)

# Optionally visualize segmented characters
for char in characters:
    cv2.imshow("Character", char)
    cv2.waitKey(500)

cv2.destroyAllWindows()