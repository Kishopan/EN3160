import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# import spiderman image
spider_bgr = cv.imread('/spider.png')
assert spider_bgr is not None
spider_hsv = cv.cvtColor(spider_bgr, cv.COLOR_BGR2HSV)
spider_rgb = cv.cvtColor(spider_bgr, cv.COLOR_BGR2RGB)

# Split the HSV image into its three separate channels: Hue (H), Saturation (S), and Value (V)
H, S, V = cv.split(spider_hsv)

# Create a figure with three subplots to display the Hue, Saturation, and Value channels separately
fig, axs = plt.subplots(1, 3, figsize=(12, 8))

# Display the Hue channel in grayscale
axs[0].imshow(H, cmap='gray', vmin=0, vmax=255)
axs[0].set_title('Hue')
axs[0].axis("off")  # Hide axis ticks and labels

# Display the Saturation channel in grayscale
axs[1].imshow(S, cmap='gray', vmin=0, vmax=255)
axs[1].set_title('Saturation')
axs[1].axis("off")  # Hide axis ticks and labels

# Display the Value channel in grayscale
axs[2].imshow(V, cmap='gray', vmin=0, vmax=255)
axs[2].set_title('Value')
axs[2].axis("off")  # Hide axis ticks and labels

# Adjust layout spacing and show the plot
plt.tight_layout()
plt.show()

# Define an intensity transformation function using a Gaussian-modulated additive term
a = 0.6
sigma = 70.0
x = np.arange(0, 256)  # Input intensity values from 0 to 255

# Compute the transformed intensity: add a scaled Gaussian curve centered at 128, capped at 255
f = np.minimum(x + a * 128 * np.exp(-((x - 128)**2) / (2 * sigma**2)), 255).astype('uint8')

# Plot the transformation function f(x) showing how input intensities are mapped to output intensities
plt.figure(figsize=(5, 5))
plt.plot(x, f)
plt.title(f'Intensity Transformation f(x) with a = {a}')
plt.xlabel('Input Intensity (x)')
plt.ylabel('Output Intensity (f(x))')
plt.grid(True)
plt.xlim([0, 255])
plt.show()
# Apply the intensity transformation function to the Saturation channel using a lookup table
S_modified = cv.LUT(S, f)

# Merge the original Hue and Value channels with the modified Saturation channel
merged = cv.merge([H, S_modified, V])

# Convert the modified HSV image back to RGB color space for visualization
spider_modified = cv.cvtColor(merged, cv.COLOR_HSV2RGB)

# Create a figure with two subplots to compare the original and vibrance-adjusted images
fig, axs = plt.subplots(1, 2, figsize=(12, 8))

# Display the original RGB image
axs[0].imshow(spider_rgb)
axs[0].set_title('Original')
axs[0].axis('off')  # Hide axis ticks and labels

# Display the image after vibrance adjustment (modified saturation)
axs[1].imshow(spider_modified)
axs[1].set_title('vibrance-enhanced')
axs[1].axis('off')  # Hide axis ticks and labels

# Adjust subplot spacing and show the comparison plot
plt.tight_layout()
plt.show()




