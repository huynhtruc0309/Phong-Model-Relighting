import numpy as np
import cv2

sample = 5
normal_map_path = 'sample_' + str(sample) + '/inputs/normal_map.png'

# Load the depth map
depth = np.load('sample_' + str(sample) + '/inputs/depth.npy')

# Normalize depth to [0, 1]
normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())

# Invert the depth values
inverted_depth = 1 - normalized_depth

# Scale to [0, 255]
scaled_depth = (inverted_depth * 255).astype(np.uint8)


# Assuming scaled_depth is your processed depth map from the previous steps
cv2.imwrite('sample_' + str(sample) + '/inputs/new_depth_map.png', scaled_depth)


# Load the normal map
normal_new = np.load('sample_' + str(sample) + '/inputs/normal.npy')

# Normalize to [0, 1]
normalized_normal = (normal_new - normal_new.min()) / (normal_new.max() - normal_new.min())

# Scale to [0, 255]
scaled_normal = (normalized_normal * 255).astype(np.uint8)

# Save the scaled normal map as an image
cv2.imwrite('sample_' + str(sample) + '/inputs/new_normal_map.png', scaled_normal)