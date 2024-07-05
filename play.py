import cv2
import numpy as np

# Load the binary image
binary_image_path = 'Shadow map_screenshot_05.07.2024.png'
binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
rgb_image = cv2.imread('sample_5/inputs/rgb_image.png', cv2.IMREAD_COLOR)

# Ensure the image is binary (if not already)
_, binary_image = cv2.threshold(binary_image, 128, 255, cv2.THRESH_BINARY)

# Apply morphological operations
kernel = np.ones((7, 7), np.uint8)  # You can adjust the kernel size
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Apply Gaussian blur to smooth the edges
smoothed_image = cv2.GaussianBlur(closed_image, (7, 7), 0)  # Adjust the kernel size as needed

# Threshold again to keep it binary
_, smoothed_binary_image = cv2.threshold(smoothed_image, 128, 255, cv2.THRESH_BINARY)

rgb_image *= smoothed_binary_image[:, :, None]

# Save or display the result
cv2.imwrite('Smoothed_Shadow_Map.png', smoothed_binary_image)
cv2.imshow('Original RGB Image', rgb_image)
cv2.imshow('Original Binary Image', binary_image)
cv2.imshow('Smoothed Binary Image', smoothed_binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
