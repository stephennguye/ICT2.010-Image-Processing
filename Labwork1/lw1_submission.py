import cv2
import matplotlib.pyplot as plt
import numpy as np

# Task 1: Read and Display Image
img = cv2.imread("TYCT-69345.jpg")   
if img is None:
    print("Error: Image not found")
    exit()

# OpenCV loads in BGR, convert to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6,6))
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis("off")
plt.show()

# Task 2: Resize Image
resized = cv2.resize(img_rgb, (200, 200))  # Resize to 200x200
plt.title("Resized Image (200x200)")
plt.imshow(resized)
plt.axis("off")
plt.show()

# Task 3: Brightness Adjustment
# Formula: img_processed = a*f(x,y) + b
a = 1.2   # contrast scale factor
b = 50    # brightness shift

bright_img = cv2.convertScaleAbs(img_rgb, alpha=a, beta=b)
plt.title("Brightness Adjusted Image")
plt.imshow(bright_img)
plt.axis("off")
plt.show()

# Task 4: Histogram Equalization
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Original histogram
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure()
plt.title("Histogram of Original Image")
plt.plot(hist, color='black')
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# Equalize histogram
equalized = cv2.equalizeHist(gray)
hist_eq = cv2.calcHist([equalized],[0],None,[256],[0,256])

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Equalized Image")
plt.imshow(equalized, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Equalized Histogram")
plt.plot(hist_eq, color='black')
plt.show()

# Task 5: Thresholding
# Ensure grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Global thresholding
thresh_val = 127  # midpoint of 0-255
_, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Grayscale Image")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Binarized Image")
plt.imshow(binary, cmap="gray")
plt.axis("off")
plt.show()
