import cv2
import numpy as np

def compute_guidance_image(normal_map):
    # Compute the intensity of the normal map for guidance
    # Here we use the magnitude of the normal vector as the intensity
    magnitude = np.sqrt(np.sum(normal_map**2, axis=-1))
    guidance_image = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return guidance_image

def smooth_depth_map(depth_map, normal_map, d=9, sigma_color=75, sigma_space=75):
    # Compute the guidance image from the normal map
    guidance_image = compute_guidance_image(normal_map)

    # Apply bilateral filter
    smoothed_depth_map = cv2.bilateralFilter(depth_map, d, sigma_color, sigma_space)

    # Optionally use guided filter for better results
    # smoothed_depth_map = cv2.ximgproc.guidedFilter(guidance_image, depth_map, radius=5, eps=1e-2)

    return smoothed_depth_map

# Load depth map and normal map
sample = 5
depth_map = cv2.imread('sample_' + str(sample) + '/inputs/depth_map.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
normal_map = cv2.imread('sample_' + str(sample) + '/inputs/normal_map.png', cv2.IMREAD_COLOR).astype(np.float32)

# Smooth depth map
smoothed_depth_map = smooth_depth_map(depth_map, normal_map)

cv2.imshow('Smoothed Depth Map', smoothed_depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
