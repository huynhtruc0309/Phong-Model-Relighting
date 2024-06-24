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

def bresenham_line_3d(light_position, position):
    

    points.append((light_position[0], light_position[1], light_position[2]))
    return points

def relight_image(rgb_image, depth_map, normal_map, light_position, light_color):
    print("Relighting the image...")
    height, width, _ = rgb_image.shape
    new_rgb_image = np.zeros_like(rgb_image)

    # Convert sRGB to linear RGB
    rgb_image_linear = sRGB_to_linear(rgb_image / 255.0)
    light_color_linear = sRGB_to_linear(np.array(light_color) / 255.0)

    # Convert depth map to float
    depth_map = depth_map.astype(np.float32) 

    # Phong reflection model parameters
    ambient_coefficient = 0
    diffuse_coefficient = 0.3
    specular_coefficient = 0
    shininess = 32

    foreground_position = np.array((309, 151, depth_map[151, 309]))
    foreground_position_2 = np.array((474, 67, depth_map[67, 474]))
    light_position = np.array(light_position)

    for y in range(height):
        for x in range(width):
            # Calculate position of the current pixel in 3D space
            position = np.array([x, y, depth_map[y, x]])

            # Calculate vector from pixel to light
            light_vector = abs(light_position - position)

            # Calculate depth factor
            depth_factor = (1 - light_vector[2] / 255.0) ** 2

            # Check if the light can reach the pixel
            check_position = light_position.copy()
            if light_position[0] < position[0]:
                sx = 1
            else:
                sx = -1
            if light_position[1] < position[1]:
                sy = 1
            else:
                sy = -1
            if light_position[2] < position[2]:
                sz = 1
            else:
                sz = -1

            # Driving axis is X-axis
            if light_vector[0] >= light_vector[1] and light_vector[0] >= light_vector[2]:
                err1 = light_vector[1] - light_vector[0] / 2
                err2 = light_vector[2] - light_vector[0] / 2

                while check_position[0] != position[0]:
                    if err1 > 0:
                        check_position[1] += sy
                        err1 -= light_vector[0]
                    if err2 > 0:
                        check_position[2] += sz
                        err2 -= light_vector[0]
                    err1 += light_vector[1]
                    err2 += light_vector[2]
                    check_position[0] += sx

                    depth_at_pixel = depth_map[check_position[1], check_position[0]]
                    if depth_at_pixel > light_position[2]:
                        depth_factor = 0
                        break

            # Driving axis is Y-axis
            elif light_vector[1] >= light_vector[0] and light_vector[1] >= light_vector[2]:
                err1 = light_vector[0] - light_vector[1] / 2
                err2 = light_vector[2] - light_vector[1] / 2

                while check_position[1] != position[1]:
                    if err1 > 0:
                        check_position[0] += sx
                        err1 -= light_vector[1]
                    if err2 > 0:
                        check_position[2] += sz
                        err2 -= light_vector[1]
                    err1 += light_vector[0]
                    err2 += light_vector[2]
                    check_position[1] += sy

                    depth_at_pixel = depth_map[check_position[1], check_position[0]]
                    if depth_at_pixel > light_position[2]:
                        depth_factor = 0
                        break

            # Driving axis is Z-axis
            else:
                err1 = light_vector[0] - light_vector[2] / 2
                err2 = light_vector[1] - light_vector[2] / 2

                while check_position[2] != position[2]:
                    if err1 > 0:
                        check_position[0] += sx
                        err1 -= light_vector[2]
                    if err2 > 0:
                        check_position[1] += sy
                        err2 -= light_vector[2]
                    err1 += light_vector[0]
                    err2 += light_vector[1]
                    check_position[2] += sz

                    depth_at_pixel = depth_map[check_position[1], check_position[0]]
                    if depth_at_pixel > light_position[2]:
                        depth_factor = 0
                        break

            # Normalize light vector
            light_vector = normalize(light_vector)

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
            ambient = ambient_coefficient * light_color_linear 
            diffuse = diffuse_coefficient * diffuse_intensity * light_color_linear
            specular = specular_coefficient * specular_intensity * light_color_linear

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
            original_color_linear = rgb_image_linear[y, x]
            new_color_linear = original_color_linear + color

            # Convert back to sRGB for display purposes (if needed)
            new_color_sRGB = np.clip(new_color_linear, 0, 1) ** (1 / 2.2)  # Gamma correction for display

            # Update the image with relighted colors
            new_rgb_image[y, x] = np.clip(new_color_sRGB * 255, 0, 255)

    # draw a circle on the selected position
    cv2.circle(new_rgb_image, (light_position[0], light_position[1]), 5, (255, 0, 0), -1)
    # cv2.circle(new_rgb_image, (int(foreground_position[0]), int(foreground_position[1])), 5, (0, 255, 0), -1)
    # cv2.circle(new_rgb_image, (int(foreground_position_2[0]), int(foreground_position_2[1])), 5, (0, 0, 255), -1)

    print("Image relighted!")
    return new_rgb_image.astype(np.uint8)

def sRGB_to_linear(rgb):
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear

# Paths to your images
sample = 2
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
light_z = 220                  # Fixed z-coordinate for the light  

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
