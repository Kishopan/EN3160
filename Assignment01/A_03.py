# import image
im_bgr = cv.imread('/highlights_and_shadows.jpg')
assert im_bgr is not None

# Convert the image from BGR to LAB color space
im_lab = cv.cvtColor(im_bgr, cv.COLOR_BGR2LAB)

# Convert the image from BGR to RGB color space (for correct display in matplotlib)
im_rgb = cv.cvtColor(im_bgr, cv.COLOR_BGR2RGB)

# Separate the LAB image into its individual channels: L (lightness), a, and b
L, a, b = cv.split(im_lab)

# Define a gamma correction transformation and apply it to the L (lightness) channel
gamma = 0.75
t = np.array([(i / 255.0) ** gamma * 255 for i in np.arange(0, 256)]).astype('uint8')  # Gamma lookup table
L_modified = cv.LUT(L, t)  # Apply gamma correction using the lookup table

# Recombine the modified L channel with the original a and b channels
merged = cv.merge([L_modified, a, b])

# Convert the modified LAB image back to RGB color space for display
im_modified = cv.cvtColor(merged, cv.COLOR_LAB2RGB)

# Create a figure with two side-by-side subplots for displaying images
fig, axs = plt.subplots(1, 2, figsize=(8, 6))

# Display the original RGB image in the first subplot
axs[0].imshow(im_rgb)
axs[0].set_title('Original')
axs[0].axis('off')  

# Display the gamma-corrected image in the second subplot
axs[1].imshow(im_modified)
axs[1].set_title(f'Gamma Corrected (γ={gamma})')
axs[1].axis('off')  

# Show the plot
plt.tight_layout()
plt.show()

# Calculate histograms for the original and gamma-corrected L (lightness) channels
hist1 = cv.calcHist([L] , [0], None , [256], [0,256])
hist2 = cv.calcHist([L_modified] , [0], None , [256] , [0,256])

# Flatten the L channel arrays for easier histogram plotting with matplotlib
L_flat = L.flatten()
L_modified_flat = L_modified.flatten()

# Create a figure for side-by-side histogram comparison
plt.figure(figsize=(10, 5))

# Plot histogram of the original L channel
plt.subplot(1, 2, 1)
plt.hist(L_flat, bins=256, range=(0, 256), color='black', alpha=0.9)
plt.title('Original L Channel')
plt.xlabel('pixel intensity')
plt.ylabel('frequency')
plt.xlim([0,256])

# Plot histogram of the gamma-corrected L channel
plt.subplot(1, 2, 2)
plt.hist(L_modified_flat, bins=256, range=(0, 256), color='black', alpha=0.9)
plt.title('Gamma-corrected L channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

# Define RGB color channels for plotting histograms
colors = ('r', 'g', 'b')
plt.figure(figsize=(10, 5))

# Plot histograms for the original image's RGB channels
plt.subplot(1, 2, 1)
for i, col in enumerate(colors):
    # Extract and flatten the corresponding color channel
    channel_flat = im_rgb[:, :, i].flatten()
    # Plot histogram for this channel
    plt.hist(channel_flat, bins=256, range=(0, 256), color=col, alpha=0.6)
plt.title('Original image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])

# Plot histograms for the gamma-corrected image's RGB channels
plt.subplot(1, 2, 2)
for i, col in enumerate(colors):
    # Extract and flatten the corresponding color channel
    channel_flat = im_modified[:, :, i].flatten()
    # Plot histogram for this channel
    plt.hist(channel_flat, bins=256, range=(0, 256), color=col, alpha=0.6)
plt.title('Gamma-corrected image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])

# Adjust subplot spacing and display the figure
plt.tight_layout()
plt.show()
