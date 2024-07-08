import cv2
import numpy as np
import os
from numba import njit

def load_image(file_path):
    if not os.path.isfile(file_path):
        print(f"Image '{file_path}' does not exist.")
        return None
    return cv2.imread(file_path, cv2.IMREAD_COLOR)

def load_depth_map(file_path):
    if not os.path.isfile(file_path):
        print(f"Depth map '{file_path}' does not exist.")
        return None
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def load_normal_map(file_path):
    if not os.path.isfile(file_path):
        print(f"Normal map '{file_path}' does not exist.")
        return None
    return cv2.imread(file_path, cv2.IMREAD_COLOR)

def load_mask(file_path):
    if not os.path.isfile(file_path):
        print(f"Mask '{file_path}' does not exist.")
        return None
    binary_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(binary_mask, 128, 255, cv2.THRESH_BINARY)
    return binary_mask

def refine_visibility_map(visibility_map, mask, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    smoothed_background = cv2.morphologyEx(visibility_map, cv2.MORPH_OPEN, kernel)
    smoothed_foreground = cv2.morphologyEx(visibility_map, cv2.MORPH_CLOSE, kernel)

    blended_visibility_map = np.where(mask, smoothed_foreground, smoothed_background)
    final_smoothed_map = cv2.GaussianBlur(blended_visibility_map, (kernel_size, kernel_size), 0)
    return final_smoothed_map

@njit
def calculate_visibility_map(depth_map, mask, light_position, tolerance=1):
    print("Calculating visibility map...")
    height, width = depth_map.shape
    visibility_map = np.ones_like(depth_map)

    shade_positions, stop_line_positions, depth_positions = [], [], []
    tolerances = []

    for y in range(height):
        for x in range(width):
            # Calculate position of the current pixel in 3D space
            position = np.array([x, y, depth_map[y, x]])

            # Check if the light can reach the pixel
            check_positions = draw_straight_line(light_position, position)
            
            if len(check_positions):
                depth_step = (light_position[2] - position[2]) / len(check_positions)
                check_position_depth = light_position[2]

                for check_position in check_positions:
                    check_position_depth -= depth_step

                    if abs(depth_map[int(check_position[1]), int(check_position[0])] - check_position_depth) < tolerance \
                        and mask[int(check_position[1]), int(check_position[0])] \
                        and abs(depth_map[int(check_position[1]), int(check_position[0])] - position[2]) > 1e-9: 
                            shade_positions.append((x, y))
                            tolerances.append(tolerance)
                            stop_line_positions.append((check_position[0], check_position[1]))
                            depth_positions.append(check_position_depth)
                            visibility_map[y, x] = 0
                            break
            
    print("Visibility map calculated!")
    return visibility_map, shade_positions, stop_line_positions, depth_positions, tolerances

@njit
def normalize(v):
    v = v.astype('float64')
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

@njit
def draw_straight_line(p0, p1):
    x0, y0, _ = p0
    x1, y1, _ = p1
    points = []  # List to hold the points on the line

    dx = abs(x1 - x0)  # Absolute difference in x
    dy = abs(y1 - y0)  # Absolute difference in y
    sx = 1 if x0 < x1 else -1  # Step direction for x
    sy = 1 if y0 < y1 else -1  # Step direction for y
    err = dx - dy  # Error term

    while True:
        e2 = 2 * err  # Double the error term
        if e2 > -dy:  # Adjust error term and x coordinate
            err -= dy
            x0 += sx
        if e2 < dx:  # Adjust error term and y coordinate
            err += dx
            y0 += sy
        if x0 == x1 and y0 == y1:  # Check if the end point is reached
            break
        points.append((x0, y0))  # Add the current point to the list
        
    return points  # Return the list of points

@njit
def sRGB_to_linear(rgb):
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear

@njit
def relight_image(rgb_image, depth_map, normal_map, light_position, light_color):
    print("Relighting the image...")
    height, width, _ = rgb_image.shape
    new_rgb_image = np.zeros_like(rgb_image)
    
    # Phong reflection model parameters
    ambient_coefficient = 0.2
    diffuse_coefficient = 0.2
    specular_coefficient = 0
    shininess = 32

    for y in range(height):
        for x in range(width):
            # Calculate position of the current pixel in 3D space
            position = np.array([x, y, depth_map[y, x]])

            # Calculate vector from pixel to light
            light_vector = np.abs(light_position - position)

            # Calculate depth factor, noting that the light is at maximum 1000
            depth_factor = (1 - abs(light_vector[2]) / 1000.0) ** 2 if visibility_map[y, x] else 0
            
            # Normalize light vector
            light_vector = normalize(light_vector)

            # Calculate diffuse component
            normal_vector = normal_map[y, x]
            normal_vector = normalize(normal_vector)
            diffuse_intensity = np.dot(normal_vector, light_vector) * depth_factor

            # Calculate specular component
            view_vector = np.array([0., 0., 1.])
            reflect_vector = 2 * normal_vector * np.dot(normal_vector, light_vector) - light_vector
            reflect_vector = normalize(reflect_vector)
            specular_intensity = max(np.dot(view_vector, reflect_vector), 0) ** shininess  * depth_factor

            # Combine components
            ambient = ambient_coefficient * light_color 
            diffuse = diffuse_coefficient * diffuse_intensity * light_color
            specular = specular_coefficient * specular_intensity * light_color

            color = (ambient + diffuse + specular) 

            if (x == light_position[0] and y == light_position[1]):
                print("===At Blue spot===")
                print("Light position: ", light_position)
                print("Pixel position: ", position)
                print("Light vector: ", light_vector)
                print("Normal vector: ", normal_vector)
                print("Dot product ", np.dot(normal_vector, light_vector))
                print("Depth factor: ", depth_factor)
                print("Diffuse intensity: ", diffuse_intensity)
                print("Diffuse: ", diffuse)
                print("Color: ", color)

            # Apply the lighting to the original color
            original_color_linear = rgb_image_linear[y, x]
            new_color_linear = original_color_linear + color

            # Convert back to sRGB for display purposes (if needed)
            new_color_sRGB = np.clip(new_color_linear, 0, 1) ** (1 / 2.2)  # Gamma correction for display

            # Update the image with relighted colors
            new_rgb_image[y, x] = np.clip(new_color_sRGB * 255, 0, 255)

    print("Image relighted!")
    return new_rgb_image.astype(np.uint8)


# Paths to your images
sample = 5
rgb_image_path = 'sample_' + str(sample) + '/inputs/rgb_image.png'
depth_map_path = 'sample_' + str(sample) + '/inputs/depth_map_1.png'
normal_map_path = 'sample_' + str(sample) + '/inputs/normal_map.png'
mask_path = 'sample_' + str(sample) + '/inputs/mask.png'

# Load images
rgb_image = load_image(rgb_image_path)
depth_map = load_depth_map(depth_map_path)
cv2.imshow('Depth map', depth_map)
normal_map = load_normal_map(normal_map_path)
mask = load_mask(mask_path)

# resize the images keep the aspect ratio
scale_percent = 100
width = int(rgb_image.shape[1] * scale_percent / 100)
height = int(rgb_image.shape[0] * scale_percent / 100)
dim = (width, height)
rgb_image = cv2.resize(rgb_image, dim, interpolation = cv2.INTER_AREA)
depth_map = cv2.resize(depth_map, dim, interpolation = cv2.INTER_AREA)
normal_map = cv2.resize(normal_map, dim, interpolation = cv2.INTER_AREA)
mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)

# Define new lighting parameters
white_light = [255, 255, 255]  # Example light color (white)
# the oder of the color is BGR
red_light = [0, 0, 255]  
green_light = [0, 255, 0] 
blue_light = [255, 0, 0] 

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
        x, y = selected_position
        # x, y = 245, 489

        light_position = np.array([x, y, 300]).astype(np.float64)
        light_position[2] = 1000 if light_position[2] > 1000 else light_position[2]

        # Convert sRGB to linear RGB
        new_rgb_image = rgb_image.copy()
        rgb_image_linear = sRGB_to_linear(new_rgb_image / 255.0)
        light_color_linear = sRGB_to_linear(np.array(white_light) / 255.0)

        # Convert depth map to float
        depth_map = depth_map.astype(np.float64) 

        # Calculate the visibility map
        visibility_map, shade_positions, stop_line_positions, depth_positions, tolerances = calculate_visibility_map(depth_map, mask, light_position)

        # Refine the visibility map
        visibility_map = refine_visibility_map(visibility_map, mask)

        # Relight with white light
        new_rgb_image = relight_image(rgb_image_linear, depth_map, normal_map, light_position, light_color_linear)

        # draw a circle on the selected position
        cv2.circle(new_rgb_image, (int(light_position[0]), int(light_position[1])), 5, (255, 0, 0), -1)

        cv2.imshow('Image', new_rgb_image)
        cv2.imshow('Shadow map', visibility_map * 255)

        # stop_flag = True
        # for i in range(len(shade_positions)):
        #     cv2.line(new_rgb_image, (int(shade_positions[i][0]), int(shade_positions[i][1])), (int(stop_line_positions[i][0]), int(stop_line_positions[i][1])), (0, 0, 0), 1)
        #     cv2.circle(new_rgb_image, (int(stop_line_positions[i][0]), int(stop_line_positions[i][1])), 1, (0, 255, 0), -1)
        #     cv2.circle(new_rgb_image, (int(shade_positions[i][0]), int(shade_positions[i][1])), 1, (0, 0, 255), -1)

        #     cv2.imshow('Not lighted points', new_rgb_image)
            
        #     print("Stop point:", stop_line_positions[i], depth_map[int(stop_line_positions[i][1]), int(stop_line_positions[i][0])], tolerances[i], mask[int(stop_line_positions[i][1]), int(stop_line_positions[i][0])])
        #     print("Shade point:", shade_positions[i], depth_map[shade_positions[i][1], shade_positions[i][0]])
        #     print("Depth position:", depth_positions[i])

        #     if stop_flag:
        #         key = cv2.waitKey(0) # Wait indefinitely for a key press
        #         if key == 13 or key == 10: # ASCII code for Enter key (13 on some systems, 10 on others)
        #             continue # Proceed to the next iteration of the loop
        #         else:
        #             stop_flag = False

        # Reset the selected position
        selected_position = None

# Destroy all windows when done
cv2.destroyAllWindows()