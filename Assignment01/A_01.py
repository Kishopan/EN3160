import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Import Emma image
emma = cv.imread('/emma.jpg', cv.IMREAD_GRAYSCALE)
assert emma is not None

# Define piecewise transformation function
t1 = np.linspace(0, 50, num=51).astype('uint8')
t2 = np.linspace(100, 255, num=100).astype('uint8')
t3 = np.linspace(150, 255, num=105).astype('uint8')

# Concatenate all segments to create the transformation array
t = np.concatenate((t1, t2, t3), axis=0).astype('uint8')
print(t.shape)

# Plot the array
plt.figure(figsize=(5, 5))
plt.plot(t)
plt.xlabel("Input intensity")
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.ylabel("Output intensity")
plt.grid(True)
plt.show()


g = t[emma]

# Display the transformed image
plt.figure(figsize=(6, 6))
plt.imshow(g, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()

# Create a figure with two side-by-side subplots to display the original and transformed images
fig, axs = plt.subplots(1, 2, figsize=(8, 6))
axs[0].imshow(emma, cmap='gray', vmin=0, vmax=255)
axs[0].set_title('Original')
axs[0].axis("off")
axs[1].imshow(g, cmap='gray', vmin=0, vmax=255)
axs[1].set_title('Transformed')
axs[1].axis("off")

