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

    # Convert depth map to float
    depth_map = depth_map.astype(np.float32) 

    # Phong reflection model parameters
    ambient_coefficient = 0.05
    diffuse_coefficient = 0.3
    specular_coefficient = 0
    shininess = 32

    # foreground_position = np.array((308, 148, depth_map[148, 308]))
    # foreground_position_2 = np.array((452, 131, depth_map[131, 452]))
    light_position = np.array(light_position)
    light_color = np.array(light_color) / 255.0

    for y in range(height):
        for x in range(width):
            # Calculate position of the current pixel in 3D space
            position = np.array([x, y, depth_map[y, x]])

            # Calculate vector from pixel to light
            light_vector = abs(light_position - position)
            light_vector = normalize(light_vector)

            # Calculate depth difference
            light_depth = light_position[2] - depth_map[y, x]
            if light_depth > 0:
                depth_factor = (1 - light_depth / 255.0) ** 2
            else:
                depth_factor = 0

            # Calculate diffuse component
            normal_vector = normal_map[y, x]
            normal_vector = normalize(normal_vector)
            diffuse_intensity = np.dot(normal_vector, light_vector) * depth_factor

            # Calculate specular component
            view_vector = np.array([0, 0, 1])
            reflect_vector = 2 * normal_vector * np.dot(normal_vector, light_vector) - light_vector
            reflect_vector = normalize(reflect_vector)
            specular_intensity = max(np.dot(view_vector, reflect_vector), 0) ** shininess  * depth_factor

            # Combine components
            ambient = ambient_coefficient * light_color 
            diffuse = diffuse_coefficient * diffuse_intensity * light_color
            specular = specular_coefficient * specular_intensity * light_color

            color = (ambient + diffuse + specular) 
            # color = np.clip(color, 0, 1)

            if (x == light_position[0] and y == light_position[1]):
                print("===At Blue spot===")
                print("Depth factor: ", depth_factor)
                print("Light position: ", light_position)
                print("Pixel position: ", position)
                print("Light vector: ", light_vector)
                print("Normal vector: ", normal_vector)
                print("Dot product ", np.dot(normal_vector, light_vector))
                print("Depth:", ((255 - depth_map[y, x]) / 255) ** 2)
                print("Diffuse intensity: ", diffuse_intensity)
                print("Diffuse: ", diffuse)
                print("Color: ", color)

            # if (x == foreground_position[0] and y == foreground_position[1]):
            #     print("===At Green spot===")
            #     print("Depth factor: ", depth_factor)
            #     print("Light position: ", light_position)
            #     print("Pixel position: ", position)
            #     print("Light vector: ", light_vector)
            #     print("Normal vector: ", normal_vector)
            #     print("Dot product ", np.dot(normal_vector, light_vector))    
            #     print("Depth:", ((255 - depth_map[y, x]) / 255) ** 2)
            #     print("Diffuse intensity: ", diffuse_intensity)
            #     print("Diffuse: ", diffuse)
            #     print("Color: ", color)

            # if (x == foreground_position_2[0] and y == foreground_position_2[1]):
            #     print("===At Red spot===")
            #     print("Depth factor: ", depth_factor)
            #     print("Light position: ", light_position)
            #     print("Pixel position: ", position)
            #     print("Light vector: ", light_vector)
            #     print("Normal vector: ", normal_vector)
            #     print("Dot product ", np.dot(normal_vector, light_vector))    
            #     print("Depth:", ((255 - depth_map[y, x]) / 255) ** 2)
            #     print("Diffuse intensity: ", diffuse_intensity)
            #     print("Diffuse: ", diffuse)
            #     print("Color: ", color)

            # Apply the lighting to the original color
            original_color = rgb_image[y, x] / 255.0
            new_color = original_color + color
            new_rgb_image[y, x] = np.clip(new_color * 255, 0, 255)

    # draw a circle on the selected position
    cv2.circle(new_rgb_image, (light_position[0], light_position[1]), 5, (255, 0, 0), -1)
    # cv2.circle(new_rgb_image, (int(foreground_position[0]), int(foreground_position[1])), 5, (0, 255, 0), -1)
    # cv2.circle(new_rgb_image, (int(foreground_position_2[0]), int(foreground_position_2[1])), 5, (0, 0, 255), -1)

    return new_rgb_image.astype(np.uint8)

# Paths to your images
sample = 3
rgb_image_path = 'sample_' + str(sample) + '/inputs/rgb_image.png'
depth_map_path = 'sample_' + str(sample) + '/inputs/depth_map.png'
normal_map_path = 'sample_' + str(sample) + '/inputs/normal_map.png'

# Load images
rgb_image = load_image(rgb_image_path)
depth_map = load_depth_map(depth_map_path)
normal_map = load_normal_map(normal_map_path)

# resize the images keep the aspect ratio
scale_percent = 25
width = int(rgb_image.shape[1] * scale_percent / 100)
height = int(rgb_image.shape[0] * scale_percent / 100)
dim = (width, height)
rgb_image = cv2.resize(rgb_image, dim, interpolation = cv2.INTER_AREA)
depth_map = cv2.resize(depth_map, dim, interpolation = cv2.INTER_AREA)
normal_map = cv2.resize(normal_map, dim, interpolation = cv2.INTER_AREA)

# Check if images are loaded correctly
if rgb_image is None or depth_map is None or normal_map is None:
    print("One or more images could not be loaded. Exiting program.")
    exit()

# Define new lighting parameters
white_light = [255, 255, 255]  # Example light color (white)
red_light = [0, 0, 255]  
green_light = [0, 255, 0] 
blue_light = [255, 0, 0] 
light_z = 100                  # Fixed z-coordinate for the light  

# Initialize a global variable to store the selected position
selected_position = None

# Define a mouse callback function to capture the selected position
def mouse_callback(event, x, y, flags, param): 
    global selected_position
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_position = (x, y)

# Create a window and set the mouse callback function
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)
cv2.imshow('Image', rgb_image)

while True:
    # Wait for a key event for 1 ms
    key = cv2.waitKey(1) & 0xFF

    # If the user presses the 'q' key, break the loop
    if key == ord('q'):
        break

    # Check if a position has been selected
    if selected_position:
        # Process the selected position (example: draw a circle on the selected position)
        x, y = selected_position

        # Update the image (you can add your processing logic here)
        light_position = [x, y, light_z]
        print(f"Light position: {light_position}")

        # Relight with white light
        new_rgb_image = relight_image(rgb_image, depth_map, normal_map, light_position, white_light)
        cv2.imshow('Image', new_rgb_image)

        # # Relight with red light
        # new_red_rgb_image = relight_image(rgb_image, depth_map, normal_map, light_position, red_light)
        # cv2.imwrite('sample_' + str(sample) + '/output_1/relighted_image_red.png', new_red_rgb_image)

        # # Relight with green light
        # new_green_rgb_image = relight_image(rgb_image, depth_map, normal_map, light_position, green_light)
        # cv2.imwrite('sample_' + str(sample) + '/output_1/relighted_image_green.png', new_green_rgb_image)

        # # Relight with blue light
        # new_blue_rgb_image = relight_image(rgb_image, depth_map, normal_map, light_position, blue_light)
        # cv2.imwrite('sample_' + str(sample) + '/output_1/relighted_image_blue.png', new_blue_rgb_image)

        # Reset the selected position
        selected_position = None

# Destroy all windows when done
cv2.destroyAllWindows()
