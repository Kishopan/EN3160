import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#import einstein image
einstein = cv.imread('/einstein.png', cv.IMREAD_GRAYSCALE)
assert einstein is not None

# Define the Sobel-X filter
sobel_x = np.array([[1, 0, -1], 
                    [2, 0, -2], 
                    [1, 0, -1]])

# Define the Sobel-Y filter
sobel_y = np.array([[1, 2, 1], 
                    [0, 0, 0], 
                    [-1, -2, -1]])

# Apply the Sobel filter kernel in the horizontal (X) direction using filter2D
sobel_x_filtered = cv.filter2D(einstein, cv.CV_64F, sobel_x)

# Apply the Sobel filter kernel in the vertical (Y) direction using filter2D
sobel_y_filtered = cv.filter2D(einstein, cv.CV_64F, sobel_y)

# Create a figure with two subplots to display the filtered images
fig, ax = plt.subplots(1, 2, figsize=(12, 8))

# Display the X-direction Sobel filtered image
ax[0].imshow(sobel_x_filtered, cmap='gray')
ax[0].set_title('Sobel X (Using filter2D)')
ax[0].axis("off")

# Display the Y-direction Sobel filtered image
ax[1].imshow(sobel_y_filtered, cmap='gray')
ax[1].set_title('Sobel Y (Using filter2D)')
ax[1].axis("off")

# Adjust layout for better spacing and show the plots
plt.tight_layout()
plt.show()


def apply_filter(image, filter):
    [rows, columns] = np.shape(image) # Get rows and columns of the image
    filtered_image = np.zeros(shape=(rows, columns)) # Create empty image
    
    for i in range(rows - 2):
        for j in range(columns - 2): # Process 2D convolution
            value = np.sum(np.multiply(filter, image[i:i + 3, j:j + 3])) 
            filtered_image[i + 1, j + 1] = value
    
    return filtered_image


# Apply the Sobel filter in the X direction
sobel_x_filtered = apply_filter(einstein, sobel_x)

# Apply the Sobel filter in the Y direction
sobel_y_filtered = apply_filter(einstein, sobel_y)
# Create the figure for plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 8))

ax[0].imshow(sobel_x_filtered, cmap='gray')
ax[0].set_title('Sobel X (Using custom function)')
ax[0].axis("off")
ax[1].imshow(sobel_y_filtered, cmap='gray')
ax[1].set_title('Sobel Y (Using custom function)')
ax[1].axis("off")

plt.tight_layout()
plt.show()


# Sobel x filter seperated
sobel_x_vertical = np.array([[1], [2], [1]])
sobel_x_horizontal = np.array([[1, 0, -1]])

# Sobel y filter seperated
sobel_y_vertical = np.array([[1], [0], [-1]])
sobel_y_horizontal = np.array([[1, 2, 1]])

# Apply the vertical and horizontal filters consecutively
x_mid = cv.filter2D(einstein, cv.CV_64F, sobel_x_horizontal)
x_filtered_image = cv.filter2D(x_mid, cv.CV_64F, sobel_x_vertical)

y_mid = cv.filter2D(einstein, cv.CV_64F, sobel_y_vertical)
y_filtered_image = cv.filter2D(y_mid, cv.CV_64F, sobel_y_horizontal)

print(sobel_x_vertical @ sobel_x_horizontal)
print(sobel_y_vertical @ sobel_y_horizontal)

# Create a figure with 1 row and 4 columns for displaying images
fig, axs = plt.subplots(1, 4, figsize=(12, 8))

# Display the intermediate Sobel X step
axs[0].imshow(x_mid, cmap='gray')
axs[0].set_title('Sobel X intermediate step')
axs[0].axis("off")

# Display the final Sobel X image
axs[1].imshow(x_filtered_image, cmap='gray')
axs[1].set_title('Sobel X final image')
axs[1].axis("off")

# Display the intermediate Sobel Y step
axs[2].imshow(y_mid, cmap='gray')
axs[2].set_title('Sobel Y intermediate step')
axs[2].axis("off")

# Display the final Sobel Y image
axs[3].imshow(y_filtered_image, cmap='gray')
axs[3].set_title('Sobel Y final image')
axs[3].axis("off")

# Adjust layout to prevent overlapping titles and labels
plt.tight_layout()
plt.show()



