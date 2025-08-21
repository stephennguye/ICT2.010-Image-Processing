import cv2
import numpy as np
import matplotlib.pyplot as plt

# Task 1: Load Image & Histogram
img = cv2.imread("TYCT-69345.jpg", cv2.IMREAD_GRAYSCALE)  
if img is None:
    print("Error: Image not found.")
    exit()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Histogram")
plt.hist(img.ravel(), bins=256, range=[0,256], color="black")
plt.show()

# ------------------------------
# Task 2: Apply Filters
# ------------------------------
avg_blur = cv2.blur(img, (5,5))                 # Averaging filter
gauss_blur = cv2.GaussianBlur(img, (5,5), 1.0) # Gaussian filter
median_blur = cv2.medianBlur(img, 5)           # Median filter

plt.figure(figsize=(12,4))
plt.subplot(1,4,1), plt.imshow(img, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(1,4,2), plt.imshow(avg_blur, cmap="gray"), plt.title("Averaging"), plt.axis("off")
plt.subplot(1,4,3), plt.imshow(gauss_blur, cmap="gray"), plt.title("Gaussian"), plt.axis("off")
plt.subplot(1,4,4), plt.imshow(median_blur, cmap="gray"), plt.title("Median"), plt.axis("off")
plt.show()

# Task 3: Add Noise
# Gaussian noise
def add_gaussian_noise(image, mean=0, var=20):
    row,col = image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    noisy = image + gauss
    noisy = np.clip(noisy,0,255).astype(np.uint8)
    return noisy

# Salt & Pepper noise
def add_salt_pepper_noise(image, prob=0.02):
    noisy = np.copy(image)
    rnd = np.random.rand(*image.shape)
    noisy[rnd < prob/2] = 0
    noisy[rnd > 1 - prob/2] = 255
    return noisy

# Periodic noise
def add_periodic_noise(image):
    row, col = image.shape
    X, Y = np.meshgrid(np.arange(col), np.arange(row))
    sinusoid = 50 * np.sin(0.1*X + 0.1*Y)
    noisy = image + sinusoid
    noisy = np.clip(noisy,0,255).astype(np.uint8)
    return noisy

gauss_noisy = add_gaussian_noise(img)
sp_noisy = add_salt_pepper_noise(img)
per_noisy = add_periodic_noise(img)

def show_img_hist(image, title):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.hist(image.ravel(), bins=256, range=[0,256], color="black")
    plt.title(title+" Histogram")
    plt.show()

show_img_hist(gauss_noisy, "Gaussian Noise")
show_img_hist(sp_noisy, "Salt & Pepper Noise")
show_img_hist(per_noisy, "Periodic Noise")

# Task 4: Apply Filters to Noisy Images
def apply_filters(noisy, title):
    avg = cv2.blur(noisy,(5,5))
    gauss = cv2.GaussianBlur(noisy,(5,5),1.0)
    median = cv2.medianBlur(noisy,5)

    plt.figure(figsize=(12,4))
    plt.subplot(1,4,1), plt.imshow(noisy,cmap="gray"), plt.title(title+" Noisy"), plt.axis("off")
    plt.subplot(1,4,2), plt.imshow(avg,cmap="gray"), plt.title("Averaging"), plt.axis("off")
    plt.subplot(1,4,3), plt.imshow(gauss,cmap="gray"), plt.title("Gaussian"), plt.axis("off")
    plt.subplot(1,4,4), plt.imshow(median,cmap="gray"), plt.title("Median"), plt.axis("off")
    plt.show()

apply_filters(gauss_noisy, "Gaussian")
apply_filters(sp_noisy, "Salt & Pepper")
apply_filters(per_noisy, "Periodic")

# Task 5: Fourier Transform on Periodic Noise
f = np.fft.fft2(per_noisy)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift)+1)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1), plt.imshow(per_noisy, cmap="gray"), plt.title("Periodic Noisy"), plt.axis("off")
plt.subplot(1,2,2), plt.imshow(magnitude_spectrum, cmap="gray"), plt.title("Fourier Spectrum"), plt.axis("off")
plt.show()

# Frequency Filtering 
rows, cols = per_noisy.shape
crow, ccol = rows//2 , cols//2
mask = np.ones((rows,cols), np.uint8)
mask[crow-10:crow+10, ccol-10:ccol+10] = 1   # keep low freq
# Example notch filters: block some areas
mask[crow-30:crow+30, ccol-80:ccol-60] = 0
mask[crow-30:crow+30, ccol+60:ccol+80] = 0

fshift_filtered = fshift * mask
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1), plt.imshow(per_noisy, cmap="gray"), plt.title("Periodic Noisy"), plt.axis("off")
plt.subplot(1,2,2), plt.imshow(img_back, cmap="gray"), plt.title("Filtered (Freq Domain)"), plt.axis("off")
plt.show()
