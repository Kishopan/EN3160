import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# import shells image
shells = cv.imread('/shells.tif',cv.IMREAD_GRAYSCALE)
assert shells is not None

def histogram_equalization(f):
    # Number of intensity levels (0 to 255)
    L = 256
    # Get image dimensions (height M and width N)
    M, N = f.shape

    # Calculate the histogram of the input image
    hist = cv.calcHist([f], [0], None, [L], [0, L])
    # Compute the cumulative distribution function (CDF) from the histogram
    cdf = hist.cumsum()

    # Create the histogram equalization mapping (transformation function)
    # Scale the CDF to the range [0, L-1] and normalize by total number of pixels (M*N)
    t = np.array([(L - 1) / (M * N) * cdf[k] for k in range(L)]).astype("uint8")

    # Map the original image pixels through the equalization transformation and return result
    return t[f]

# Perform histogram equalization on the grayscale image using the custom function
equalized = histogram_equalization(shells)

# Create a figure with two subplots to display the original and equalized images side by side
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Display the original grayscale image
axs[0].imshow(shells, cmap='gray', vmin=0, vmax=255)
axs[0].set_title('Original')
axs[0].axis("off")  

# Display the histogram-equalized image
axs[1].imshow(equalized, cmap='gray', vmin=0, vmax=255)
axs[1].set_title('Equalized using function')
axs[1].axis("off") 

plt.tight_layout()
plt.show()

# Calculate histograms for the original and equalized grayscale images
hist1 = cv.calcHist([shells], [0], None, [256], [0, 256])
hist2 = cv.calcHist([equalized], [0], None, [256], [0, 256])

# Flatten the image arrays for plotting histograms using matplotlib
shells_flat = shells.flatten()
equalized_flat = equalized.flatten()

# Create a figure with two subplots for side-by-side histogram comparison
plt.figure(figsize=(10, 5))

# Plot histogram of the original image
plt.subplot(1, 2, 1)
plt.hist(shells_flat, bins=256, range=(0, 256), color='black', alpha=0.9)
plt.title('Before Equalization')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])

# Plot histogram of the histogram-equalized image
plt.subplot(1, 2, 2)
plt.hist(equalized_flat, bins=256, range=(0, 256), color='black', alpha=0.9)
plt.title('After Equalization')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])

# Adjust layout spacing and display the histograms
plt.tight_layout()
plt.show()


