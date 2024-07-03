import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image dimensions
height, width = 200, 300

# Create a blue background image
color_image = np.zeros((height, width, 3), dtype=np.uint8)
color_image[:, :] = [112, 57, 6]  # Blue background (BGR format)

# Draw a red rectangle in the center
rect_start = (width // 4, height // 4)
rect_end = (3 * width // 4, 3 * height // 4)
cv2.rectangle(color_image, rect_start, rect_end, (67, 135, 226), -1)  # Red rectangle (BGR format)

# Create a depth map
depth_map = np.zeros((height, width), dtype=np.uint8)
depth_map[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]] = 255

# Create a normal map with all normals pointing in the same direction
# Assuming the normal vector (0, 0, 1) in 3D space (normalized to [128, 128, 255] for 8-bit image)
normal_map = np.zeros((height, width, 3), dtype=np.uint8)
normal_map[:, :] = [128, 128, 255]

# Save the images
cv2.imwrite('sample_6/inputs/rgb_image.png', color_image)
cv2.imwrite('sample_6/inputs/depth_map.png', depth_map)
cv2.imwrite('sample_6/inputs/normal_map.png', normal_map)

# Plot the images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Color Image')
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Depth Map')
plt.imshow(depth_map, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Normal Map')
plt.imshow(cv2.cvtColor(normal_map, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
