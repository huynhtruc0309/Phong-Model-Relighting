import cv2
import numpy as np

def normalize(v):
    v = v.astype('float64')
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def visualize_lighting(normal_vector, light_vector):
    """
    Visualize the lighting effect based on the dot product of the normal and light vectors.
    
    Parameters:
    normal_vector (tuple): The normal vector as (nx, ny, nz).
    light_vector (tuple): The light vector as (lx, ly, lz).
    
    Returns:
    image: An image visualizing the lighting effect.
    """
    # Calculate the dot product
    dot_product = np.dot(normalize(normal_vector), normalize(light_vector))
    
    # Calculate the intensity based on the dot product
    intensity = max(dot_product, 0)  # Clamp to zero to avoid negative values

    print("Dot Product: ", dot_product)
    # Normalize the intensity to a range between 0 and 255
    intensity_normalized = np.uint8(255 * intensity)

    # Create an image with the resulting intensity
    height, width = 100, 100
    image = np.full((height, width, 3), intensity_normalized, dtype=np.uint8)
    
    return image, dot_product

# Example normal and light vectors
normal_vectors = [
    np.array([126, 147, 1]),    # Directly facing the light
    np.array([169, 142, 7]),    # Perpendicular to the light
    np.array([85, 139, 1])    # Facing away from the light
]

light_vector = np.array([0, 0, 200])  # Light coming from directly above

# Create a window to display the results
cv2.namedWindow("Lighting Effect", cv2.WINDOW_NORMAL)

# Visualize the lighting effect for each normal vector
for normal_vector in normal_vectors:
    image, dot_product = visualize_lighting(normal_vector, light_vector)
    print(f"Normal Vector: {normal_vector}, Dot Product: {dot_product}")
    cv2.imshow("Lighting Effect", image)
    cv2.waitKey(0)

# Close the window when done
cv2.destroyAllWindows()
