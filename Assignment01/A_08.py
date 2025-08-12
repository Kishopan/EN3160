import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def zoom(image, technique, scale=4):
  if technique == 'nn':
        # Nearest Neighbor interpolation (fast but blocky result)
        return cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
  elif technique == 'bilinear':
        # Bilinear interpolation (slower but smoother result)
        return cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

def norm_SSD(image1, image2):
  if image1.shape != image2.shape:
        # Ensure both images have the same shape for comparison
        raise ValueError("Images must have the same dimensions")
  # Compute SSD and normalize by total number of pixels
  return np.sum((image1 - image2) ** 2) / image1.size

# Import 1st image
im01 = cv.imread('/im01.png')
assert im01 is not None
im01_small = cv.imread('/im01small.png')
assert im01_small is not None

# Zoom the smaller image using Nearest Neighbour interpolation
im01_zoomed_nn = zoom(im01_small, technique='nn')

# Zoom the smaller image using Bilinear interpolation
im01_zoomed_bilinear = zoom(im01_small, technique='bilinear')

# Calculate Normalized SSD between the original image and the NN-zoomed image
nn_SSD = norm_SSD(im01, im01_zoomed_nn)

# Calculate Normalized SSD between the original image and the Bilinear-zoomed image
bilinear_SSD = norm_SSD(im01, im01_zoomed_bilinear)

# Print out the calculated SSD values for comparison
print('Normalized SSD for Nearest Neighbour: ', nn_SSD)
print('Normalized SSD for Bilinear: ', bilinear_SSD)

# Create a figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 8))

# Plot the second image
axs[0].imshow(cv.cvtColor(im01_zoomed_nn, cv.COLOR_BGR2RGB))
axs[0].set_title(f'Nearest Neighbour (SSD: {nn_SSD:.5f})')
axs[0].axis('off')  # Turn off the axis

# Plot the second image
axs[1].imshow(cv.cvtColor(im01_zoomed_bilinear, cv.COLOR_BGR2RGB))
axs[1].set_title(f'Bilinar Interpolation (SSD: {bilinear_SSD:.5f})')
axs[1].axis('off')  # Turn off the axis

# Show the plot
plt.tight_layout()
plt.show()

# Try other images also
# Import 2nd image
im02 = cv.imread('/im02.png')
assert im02 is not None
im02_small = cv.imread('/im02small.png')
assert im02_small is not None

# Zoom the smaller image using Nearest Neighbour interpolation
im02_zoomed_nn = zoom(im02_small, technique='nn')

# Zoom the smaller image using Bilinear interpolation
im02_zoomed_bilinear = zoom(im02_small, technique='bilinear')

# Calculate Normalized SSD between the original image and the NN-zoomed image
nn_SSD = norm_SSD(im02, im02_zoomed_nn)

# Calculate Normalized SSD between the original image and the Bilinear-zoomed image
bilinear_SSD = norm_SSD(im02, im02_zoomed_bilinear)

# Print out the calculated SSD values for comparison
print('Normalized SSD for Nearest Neighbour: ', nn_SSD)
print('Normalized SSD for Bilinear: ', bilinear_SSD)

# Create a figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 8))

# Plot the second image
axs[0].imshow(cv.cvtColor(im02_zoomed_nn, cv.COLOR_BGR2RGB))
axs[0].set_title(f'Nearest Neighbour (SSD: {nn_SSD:.5f})')
axs[0].axis('off')  # Turn off the axis

# Plot the second image
axs[1].imshow(cv.cvtColor(im02_zoomed_bilinear, cv.COLOR_BGR2RGB))
axs[1].set_title(f'Bilinar Interpolation (SSD: {bilinear_SSD:.5f})')
axs[1].axis('off')  # Turn off the axis

# Show the plot
plt.tight_layout()
plt.show()

# Import 3rd image
im03 = cv.imread('/im03.png')
assert im03 is not None
im03_small = cv.imread('/im03small.png')
assert im03_small is not None

# Zoom the smaller image using Nearest Neighbour interpolation
im03_zoomed_nn = zoom(im03_small, technique='nn')

# Zoom the smaller image using Bilinear interpolation
im03_zoomed_bilinear = zoom(im03_small, technique='bilinear')

# Calculate Normalized SSD between the original image and the NN-zoomed image
nn_SSD = norm_SSD(im03, im03_zoomed_nn)

# Calculate Normalized SSD between the original image and the Bilinear-zoomed image
bilinear_SSD = norm_SSD(im03, im03_zoomed_bilinear)

# Print out the calculated SSD values for comparison
print('Normalized SSD for Nearest Neighbour: ', nn_SSD)
print('Normalized SSD for Bilinear: ', bilinear_SSD)

# Create a figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 8))

# Plot the Nearest Neighbour result
axs[0].imshow(cv.cvtColor(im03_zoomed_nn, cv.COLOR_BGR2RGB))
axs[0].set_title(f'Nearest Neighbour (SSD: {nn_SSD:.5f})')
axs[0].axis('off')  # Turn off the axis

# Plot the Bilinear Interpolation result
axs[1].imshow(cv.cvtColor(im03_zoomed_bilinear, cv.COLOR_BGR2RGB))
axs[1].set_title(f'Bilinear Interpolation (SSD: {bilinear_SSD:.5f})')
axs[1].axis('off')  # Turn off the axis

# Show the plot
plt.tight_layout()
plt.show()



