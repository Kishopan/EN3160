import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#import brain image
brain = cv.imread('/brain_proton_density_slice.png', cv.IMREAD_GRAYSCALE)
assert brain is not None

# Generate a Gaussian-shaped intensity transformation curve
mu = 150      # Mean (center) of the Gaussian
sigma = 20    # Standard deviation (width/spread) of the Gaussian
x = np.linspace(0, 255, 256)  # Input intensity values (0–255)
t = 255 * np.exp(-((x - mu)**2) / (2 * sigma**2))  # Gaussian function scaled to 255

# Limit all values to stay within the valid image intensity range [0, 255]
t = np.clip(t, 0, 255)

print(t.shape)

# Plot the Gaussian transformation curve
plt.figure(figsize=(5, 5))
plt.plot(t)
plt.xlabel("Input intensity")
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.ylabel("Output intensity")
plt.grid(True)
plt.show()


g = t[brain]

# Display the transformed image
plt.figure(figsize=(5, 5))
plt.imshow(g, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()

mu = 200
sigma = 15
x = np.linspace(0, 255, 256)
t = 255 * np.exp(-((x-mu)**2)/(2*sigma**2))
t = np.clip(t, 0 ,255)

print(t.shape)

plt.figure(figsize=(5, 5))
plt.plot(t)
plt.xlabel("Input intensity")
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.ylabel("Output intensity")
plt.grid(True)
plt.show()

g = t[brain]

# Display the transformed image
plt.figure(figsize=(5, 5))
plt.imshow(g, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()
