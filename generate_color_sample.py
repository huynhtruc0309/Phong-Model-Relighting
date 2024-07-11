import cv2
import numpy as np

# Define the colors in RGBA format
colors = [
    (39, 127, 255),   # Orange
    (164, 73, 163),  # Purple
    (76, 177, 34)   # Green
]

# Image dimensions and grid size
image_width = 300
image_height = 300
grid_size = 3  # 3x3 grid

# Calculate the size of each cell
cell_width = image_width // grid_size
cell_height = image_height // grid_size

# Create the grid image
grid_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# Fill the grid image with the specified colors
for i in range(grid_size):
    for j in range(grid_size):
        color = colors[(i * grid_size + j) % len(colors)]
        top_left = (j * cell_width, i * cell_height)
        bottom_right = ((j + 1) * cell_width, (i + 1) * cell_height)
        cv2.rectangle(grid_image, top_left, bottom_right, color, -1)

# Generate depth map (linear depth from top-left to bottom-right)
depth_map = np.zeros((image_height, image_width), dtype=np.float32)

# Generate normal map (normal facing up)
normal_map = np.zeros((image_height, image_width, 3), dtype=np.float32)
normal_map[:, :, 1] = 1.0  # Y component of the normal vector (pointing up)

# Convert normal map to 0-255 range
normal_map = cv2.normalize(normal_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Save and display the images
cv2.imwrite('rgb_image.png', grid_image)

cv2.imshow('Grid Image', grid_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
