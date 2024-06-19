import cv2
import numpy as np

def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_COLOR)

def load_depth_map(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def load_normal_map(file_path):
    return cv2.imread(file_path, cv2.IMREAD_COLOR)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def relight_image(rgb_image, depth_map, normal_map, light_position, light_color):
    print("Relighting the image...")
    height, width, _ = rgb_image.shape
    new_rgb_image = np.zeros_like(rgb_image)

    # Normalize normal map to [-1, 1]
    normal_map = (normal_map / 255.0) * 2.0 - 1.0

    # Convert depth map to float
    depth_map = depth_map.astype(np.float32) / 255.0

    # Phong reflection model parameters
    ambient_coefficient = 0.1
    diffuse_coefficient = 0.9
    specular_coefficient = 0.5
    shininess = 32

    light_position = np.array(light_position)
    light_color = np.array(light_color) / 255.0

    for y in range(height):
        for x in range(width):
            # Calculate position of the current pixel in 3D space
            position = np.array([x, y, depth_map[y, x]])

            # Calculate vector from pixel to light
            light_vector = light_position - position
            light_vector = normalize(light_vector)

            # Calculate diffuse component
            normal_vector = normal_map[y, x]
            normal_vector = normalize(normal_vector)
            diffuse_intensity = max(np.dot(normal_vector, light_vector), 0)

            # Calculate specular component
            view_vector = np.array([0, 0, 1])
            reflect_vector = 2 * normal_vector * np.dot(normal_vector, light_vector) - light_vector
            reflect_vector = normalize(reflect_vector)
            specular_intensity = max(np.dot(view_vector, reflect_vector), 0) ** shininess

            # Combine components
            ambient = ambient_coefficient * light_color
            diffuse = diffuse_coefficient * diffuse_intensity * light_color
            specular = specular_coefficient * specular_intensity * light_color

            color = ambient + diffuse + specular
            color = np.clip(color, 0, 1)

            # Apply the lighting to the original color
            original_color = rgb_image[y, x] / 255.0
            new_color = original_color * color
            new_rgb_image[y, x] = np.clip(new_color * 255, 0, 255)

    print("Image relighted!")
    return new_rgb_image.astype(np.uint8)

# Paths to your images
rgb_image_path = 'sample_2/inputs/rgb_image.jpg'
depth_map_path = 'sample_2/inputs/depth_map.jpg'
normal_map_path = 'sample_2/inputs/normal_map.png'

# Load images
rgb_image = load_image(rgb_image_path)
depth_map = load_depth_map(depth_map_path)
normal_map = load_normal_map(normal_map_path)

# Define new lighting parameters
# light_position = [674, 304, 0]  # Example light position
light_color = [255, 255, 255]     # Example light color (white)

for light_z in range(0, 260, 5):
    light_position = [963, 328, light_z]
    new_rgb_image = relight_image(rgb_image, depth_map, normal_map, light_position, light_color)
    cv2.imwrite(f'sample_2/output_2/relighted_image_{light_z}.png', new_rgb_image)
