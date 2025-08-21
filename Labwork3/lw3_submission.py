import cv2
import numpy as np
import matplotlib.pyplot as plt

# Task 1: Load Image + Laplacian + Sobel
img = cv2.imread("TYCT-69345.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Image not found")
    exit()

# Laplacian filter
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# Sobel filter (x and y gradients)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1), plt.imshow(img,cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(laplacian,cmap="gray"), plt.title("Laplacian"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(sobel,cmap="gray"), plt.title("Sobel"), plt.axis("off")
plt.show()

# Task 2: Canny Edge Detection
edges = cv2.Canny(img, 100, 200) 

plt.figure(figsize=(12,4))
plt.subplot(1,3,1), plt.imshow(laplacian,cmap="gray"), plt.title("Laplacian"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(sobel,cmap="gray"), plt.title("Sobel"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(edges,cmap="gray"), plt.title("Canny"), plt.axis("off")
plt.show()

# Task 3: Hough Line Detection-
# First use Canny to get edges
edges = cv2.Canny(img, 100, 200)

# Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)  # rho=1, theta=1deg, threshold=150
img_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img_lines,(x1,y1),(x2,y2),(0,0,255),2)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.imshow(edges,cmap="gray"), plt.title("Edges"), plt.axis("off")
plt.subplot(1,2,2), plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB)), plt.title("Hough Lines"), plt.axis("off")
plt.show()

# Task 4: Image Segmentation
# Apply a few thresholds (k values)
thresh1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
thresh2 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]
thresh3 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]

plt.figure(figsize=(12,4))
plt.subplot(1,4,1), plt.imshow(img,cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(1,4,2), plt.imshow(thresh1,cmap="gray"), plt.title("Thresh=100"), plt.axis("off")
plt.subplot(1,4,3), plt.imshow(thresh2,cmap="gray"), plt.title("Thresh=150"), plt.axis("off")
plt.subplot(1,4,4), plt.imshow(thresh3,cmap="gray"), plt.title("Thresh=200"), plt.axis("off")
plt.show()
