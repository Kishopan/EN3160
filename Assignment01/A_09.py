import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# import daisy image
daisy = cv.imread('/daisy.jpg')
assert daisy is not None

# Create an initial mask the same height & width as the input image, initialized to 0 (background)
mask = np.zeros(daisy.shape[:2],np.uint8)

# Temporary arrays used internally by the GrabCut algorithm to store background/foreground models 
bgdModel = np.zeros((1,65),np.float64) # Background model
fgdModel = np.zeros((1,65),np.float64) # Foreground model

# Define a rectangle (x, y, width, height) around the object of interest
rect = (50,100,550,490)

# Apply GrabCut segmentation
# Parameters:
# daisy    - input image
# mask     - initial mask
# rect     - rectangle containing the foreground
# bgdModel - background model (will be updated)
# fgdModel - foreground model (will be updated)
# 5        - number of iterations
# cv.GC_INIT_WITH_RECT - initialize using the rectangle
cv.grabCut(daisy,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

# Convert the mask so that foreground pixels = 1, background pixels = 0
# mask==2 or mask==0 → background (set to 0)
# mask==1 or mask==3 → foreground (set to 1)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

# Extract the foreground by multiplying the mask with the image
foreground = daisy * mask2[:, :, np.newaxis]

# Extract the background by subtracting the foreground from the original image
background = cv.subtract(daisy, foreground)

# Show the results
fig, axs = plt.subplots(1, 3, figsize=(12, 6))

axs[0].imshow(mask2, cmap='gray')
axs[0].set_title('Segmentation Mask')
axs[0].axis('off')

axs[1].imshow(cv.cvtColor(foreground, cv.COLOR_BGR2RGB))
axs[1].set_title('Foreground Image')
axs[1].axis('off')

axs[2].imshow(cv.cvtColor(background, cv.COLOR_BGR2RGB))
axs[2].set_title('Background Image')
axs[2].axis('off')

plt.tight_layout()
plt.show()

# Apply Gaussian blur to the extracted background to create a depth-of-field effect
blurred_background = cv.GaussianBlur(background, (25, 25), 3)

# Combine the sharp foreground with the blurred background
blurred = cv.add(foreground, blurred_background)

# Display the original and processed images side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 8))

# Show the original image
axs[0].imshow(cv.cvtColor(daisy, cv.COLOR_BGR2RGB))
axs[0].set_title('Original')
axs[0].axis('off')

# Show the image with background blur applied
axs[1].imshow(cv.cvtColor(blurred, cv.COLOR_BGR2RGB))
axs[1].set_title('Background Blurred Image')
axs[1].axis('off')

# Adjust spacing and display the figure
plt.tight_layout()
plt.show()


