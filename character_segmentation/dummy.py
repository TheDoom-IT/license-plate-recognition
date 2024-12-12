#Python version (3.10.12)
from platform import python_version
print(python_version())

# Import libraries
import cv2

PROJECT_DATA_PATH = "C:\\Cours\\E5\\IASR\\project_files\\dataset\\photos\\"
ANNOTATION_PATH = "C:\\Cours\\E5\\IASR\\project_files\\dataset\\annotations.xml"
file1 = "1.jpg"


image = cv2.imread(PROJECT_DATA_PATH+file1)
nom_fenetre = "image1"
cv2.imshow(nom_fenetre, image)

cv2.waitKey(0)

import numpy as np

# Get corners coordinates of plate
corners_1 = [(1318.15, 537.20), (2015.94, 704.15)]
top_left = corners_1[0]
top_left = tuple(int(num) for num in top_left)
bottom_right = corners_1[1]
bottom_right = tuple(int(num) for num in bottom_right)


# Show extracted plate image
plate = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
print("Extracted image")
cv2.imshow("plate", plate)
cv2.waitKey(0)

'''
# Show gray scale image
print("Grayscale image")
# Use the cvtColor() function to grayscale the image
gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray plate", gray_plate)
cv2.waitKey(0)

# Apply closing/opening filter on image
kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(gray_plate, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closed", closing)
cv2.waitKey(0)

# Apply Canny edge detector filter
from matplotlib import pyplot as plt

img = closing
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

'''

from imutils import contours

mask = np.zeros(plate.shape, dtype=np.uint8)
gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)

# Equalization from histogram
equ = cv2.equalizeHist(gray)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# Apply Gaussian Blur
# blurred_image = cv2.GaussianBlur(equ, (15, 15), 0)

closing = cv2.morphologyEx(equ, cv2.MORPH_CLOSE, kernel)

blurred_image = cv2.bilateralFilter(equ, 20, 75, 75)


closing = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel)
closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
img=closing

'''
edges = cv2.Canny(img,160,310) # need automatic adjustments using Otsu
imagem = cv2.bitwise_not(edges)
kernel = np.ones((5,5), np.uint8)
open2 = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, kernel)
edges2 = cv2.bitwise_not(open2)
'''

thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts)==2 else cnts[1]
(cnts,_) = contours.sort_contours(cnts,method="left-to-right")
ROI_number = 0

for c in cnts:
    area = cv2.contourArea(c)
    if 3100 > area > 100:
        x,y,w,h = cv2.boundingRect(c)
        ROI = 255 - thresh[y:y+h, x:x+w]
        cv2.drawContours(mask, [c], -1, (255,255,255), -1)
        cv2.imwrite('ROI_{}.png'.format(ROI_number),ROI)
        ROI_number += 1


print("equ")
cv2.imshow("equ", equ)
cv2.waitKey(0)

print("blurred image")
cv2.imshow("blurred image", blurred_image)
cv2.waitKey(0)

print("closing image")
cv2.imshow("closing image", closing)
cv2.waitKey(0)

print("thresh")
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

'''
print("edges")
cv2.imshow("edges", edges)
cv2.waitKey(0)

print("edges2")
cv2.imshow("edges2", edges2)
cv2.waitKey(0)
'''

print("mask")
cv2.imshow("mask", mask)
cv2.waitKey(0)