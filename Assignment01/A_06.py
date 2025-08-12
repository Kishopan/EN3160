import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# import jennifer image
jennifer_bgr = cv.imread('/jeniffer.jpg')
assert jennifer_bgr is not None

jennifer_hsv = cv.cvtColor(jennifer_bgr,cv.COLOR_BGR2HSV)
jennifer_rgb = cv.cvtColor(jennifer_bgr,cv.COLOR_BGR2RGB)

# Split the HSV image into its three separate channels: Hue (H), Saturation (S), and Value (V)
H, S, V = cv.split(jennifer_hsv)

# Create a figure with three subplots to display the Hue, Saturation, and Value channels separately
fig, axs = plt.subplots(1, 3, figsize=(12, 8))

# Display the Hue channel in grayscale
axs[0].imshow(H, cmap='gray', vmin=0, vmax=255)
axs[0].set_title('Hue')
axs[0].axis("off")  

# Display the Saturation channel in grayscale
axs[1].imshow(S, cmap='gray', vmin=0, vmax=255)
axs[1].set_title('Saturation')
axs[1].axis("off")  

# Display the Value channel in grayscale
axs[2].imshow(V, cmap='gray', vmin=0, vmax=255)
axs[2].set_title('Value')
axs[2].axis("off") 

# Adjust layout spacing and show the plot
plt.tight_layout()
plt.show()

# Threshold the Saturation channel to create a binary mask
_, mask = cv.threshold(S, 12, 255, cv.THRESH_BINARY)

# Apply the mask to the original image to isolate the foreground
foreground = cv.bitwise_and(jennifer_bgr, jennifer_bgr, mask=mask)

# Plot the isolated foreground image
plt.figure(figsize=(10, 5))
plt.imshow(cv.cvtColor(foreground, cv.COLOR_BGR2RGB))
plt.title('Extracted Foreground')
plt.axis('off')
plt.show()


plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')
plt.show()

# Convert the foreground image from BGR to HSV color space
foreground_hsv = cv.cvtColor(foreground, cv.COLOR_BGR2HSV)

# Separate the Hue, Saturation, and Value channels
H_fg, S_fg, V_fg = cv.split(foreground_hsv)

# Compute the histogram for the Value (V) channel using the provided mask
hist = cv.calcHist([V_fg], [0], mask, [256], [0, 256])

# Create an array of x-axis positions corresponding to intensity values
x_positions = np.arange(len(hist))

# Plot the Value channel histogram as a bar graph
plt.figure()
plt.bar(x_positions, hist.flatten(), color='black', width=1)  # width=1 → 1 bar per intensity value
plt.title('Histogram of Value Channel for Foreground')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])  
plt.grid(True)
plt.show()

# Compute the cumulative distribution function (CDF) from the histogram
cdf = hist.cumsum()

# Plot the histogram
plt.figure()
plt.plot(cdf, color='black')
plt.title('Cumulative Sum of the Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Total number of pixels in the masked area
pixels = cdf[-1]

# Create transformation function for histogram equalization
t = np.array([(256 - 1) / pixels * cdf[k] for k in range(256)]).astype("uint8")

# Apply the transformation to the Value channel
V_eq = t[V_fg]

# Compute histogram of the equalized Value channel (masked region only)
hist = cv.calcHist([V_eq], [0], mask, [256], [0, 256])

# X-axis positions for histogram bars
x_positions = np.arange(len(hist))

# Plot the equalized histogram
plt.figure()
plt.bar(x_positions, hist.flatten(), color='black', width=1)  # Width=1 ensures no gaps between bars
plt.title('Equalized Histogram of Value Channel for Foreground')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])  
plt.grid(True)
plt.show()

# Merge the modified HSV channels into a single image
merged = cv.merge([H_fg, S_fg, V_eq])
foreground_modified = cv.cvtColor(merged, cv.COLOR_HSV2RGB)

# Isolate the background using the inverted mask
background = cv.bitwise_and(jennifer_bgr, jennifer_bgr, mask=cv.bitwise_not(mask))

# Combine the equalized foreground with the original background
result = cv.add(cv.cvtColor(background, cv.COLOR_BGR2RGB), foreground_modified)

# Create a figure with 2 side-by-side subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 8))

# Display the original image
axs[0].imshow(jennifer_rgb)
axs[0].set_title('Original')
axs[0].axis('off')  

# Display the processed image with equalized foreground
axs[1].imshow(result)
axs[1].set_title('Foreground Equalized')
axs[1].axis('off') 
# Adjust layout and show the figure
plt.tight_layout()
plt.show()



