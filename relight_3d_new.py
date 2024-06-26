import cv2
import numpy as np

def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_COLOR)

def load_depth_map(file_path):
    return np.load(file_path)

def load_normal_map(file_path):
    return np.load(file_path)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def draw_straight_line(x0, y0, z0, x1, y1, z1, depth_map):
    points = []  # List to hold the points on the line
    dx = abs(x1 - x0)  # Absolute difference in x
    dy = abs(y1 - y0)  # Absolute difference in y
    sx = 1 if x0 < x1 else -1  # Step direction for x
    sy = 1 if y0 < y1 else -1  # Step direction for y
    err = dx - dy  # Error term

    while True:
        if depth_map[y0, x0] > min(z0, z1) and depth_map[y0, x0] < max(z0, z1):
            points.append([x0, y0])  # Add the current point to the list
        if x0 == x1 and y0 == y1:  # Check if the end point is reached
            break
        e2 = 2 * err  # Double the error term
        if e2 > -dy:  # Adjust error term and x coordinate
            err -= dy
            x0 += sx
        if e2 < dx:  # Adjust error term and y coordinate
            err += dx
            y0 += sy
    return points  # Return the list of points

def is_point_on_line_segment(p, p0, p1, tolerance=1e-9):
    x, y, z = p
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    
    # Check for degenerate line segment (P0 == P1)
    if (x0 == x1) and (y0 == y1) and (z0 == z1):
        return (x, y, z) == (x0, y0, z0)
    
    # Calculate t for each coordinate
    try:
        tx = (x - x0) / (x1 - x0) if x1 != x0 else None
        ty = (y - y0) / (y1 - y0) if y1 != y0 else None
        tz = (z - z0) / (z1 - z0) if z1 != z0 else None
    except ZeroDivisionError:
        return False
    
    # Check that t values are consistent
    t_values = [t for t in (tx, ty, tz) if t is not None]
    if len(t_values) > 1 and not all(abs(t - t_values[0]) < tolerance for t in t_values[1:]):
        return False
    
    # Check that t is in the range [0, 1]
    t = t_values[0] if t_values else None
    return t is not None and 0 <= t <= 1

def relight_image(rgb_image, depth_map, normal_map, light_position, light_color):
    print("Relighting the image...")
    height, width, _ = rgb_image.shape
    new_rgb_image = np.zeros_like(rgb_image)

    # Convert sRGB to linear RGB
    rgb_image_linear = sRGB_to_linear(rgb_image / 255.0)
    light_color_linear = sRGB_to_linear(np.array(light_color) / 255.0)

    # Convert depth map to float
    # depth_map = depth_map.astype(np.float32) 

    # Phong reflection model parameters
    ambient_coefficient = 0
    diffuse_coefficient = 0.3
    specular_coefficient = 0
    shininess = 32

    light_position = np.array(light_position)

    for y in range(height):
        for x in range(width):
            # Calculate position of the current pixel in 3D space
            position = np.array([x, y, depth_map[y, x]])

            # Calculate vector from pixel to light
            light_vector = abs(light_position - position)

            # # Calculate depth factor
            # depth_factor = ((light_vector[2] - depth_map.min()) / (depth_map.max() - depth_map.min())) ** 2

            # # Check if the light can reach the pixel
            # check_positions = draw_straight_line(light_position[0], light_position[1], light_position[2], x, y, depth_map[y, x], depth_map)
            # save_check_position = None
            # for check_position in check_positions:
            #     # Check if the light can reach the pixel
            #     check_position = np.array([check_position[0], check_position[1], depth_map[check_position[1], check_position[0]]])
            #     if is_point_on_line_segment(check_position, light_position, position):
            #         save_check_position = check_position
            #         depth_factor = 0
            #         break
            
            # Normalize light vector
            light_vector = normalize(light_vector)

            # Calculate diffuse component
            normal_vector = normal_map[y, x]
            normal_vector = normalize(normal_vector)
            diffuse_intensity = np.dot(normal_vector, light_vector) #* depth_factor

            # Calculate specular component
            view_vector = np.array([0, 0, 1])
            reflect_vector = 2 * normal_vector * np.dot(normal_vector, light_vector) - light_vector
            reflect_vector = normalize(reflect_vector)
            specular_intensity = max(np.dot(view_vector, reflect_vector), 0) ** shininess  #* depth_factor

            # Combine components
            ambient = ambient_coefficient * light_color_linear 
            diffuse = diffuse_coefficient * diffuse_intensity * light_color_linear
            specular = specular_coefficient * specular_intensity * light_color_linear

            color = (ambient + diffuse + specular) 
            # color = np.clip(color, 0, 1)

            if (x == light_position[0] and y == light_position[1]):
                print("===At Blue spot===")
                print("Light position: ", light_position)
                print("Pixel position: ", position)
                print("Light vector: ", light_vector)
                print("Normal vector: ", normal_vector)
                print("Dot product ", np.dot(normal_vector, light_vector))
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

    # draw a circle on the selected position
    cv2.circle(new_rgb_image, (light_position[0], light_position[1]), 5, (255, 0, 0), -1)

    print("Image relighted!")
    return new_rgb_image.astype(np.uint8)

def sRGB_to_linear(rgb):
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear

# Paths to your images
sample = 2
rgb_image_path = 'sample_' + str(sample) + '/inputs/rgb_image.png'
depth_map_path = 'sample_' + str(sample) + '/inputs/depth.npy'
normal_map_path = 'sample_' + str(sample) + '/inputs/normal.npy'

# Load images
rgb_image = load_image(rgb_image_path)
depth_map = load_depth_map(depth_map_path)
normal_map = load_normal_map(normal_map_path)

# Check if images are loaded correctly
if rgb_image is None or depth_map is None or normal_map is None:
    print("One or more images could not be loaded. Exiting program.")
    exit()

# resize the images keep the aspect ratio
scale_percent = 25
width = int(rgb_image.shape[1] * scale_percent / 100)
height = int(rgb_image.shape[0] * scale_percent / 100)
rgb_image = cv2.resize(rgb_image, (width, height), interpolation = cv2.INTER_AREA)
depth_map = np.resize(depth_map, (height, width))
normal_map = np.resize(normal_map, (height, width, 3))

# Define new lighting parameters
white_light = [255, 255, 255]  # Example light color (white) and the order of the color is BGR
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
        light_position = [x, y, int(depth_map[y, x])]
        print(f"Light position: {light_position}")

        # Relight with white light
        new_rgb_image = relight_image(rgb_image, depth_map, normal_map, light_position, white_light)
        cv2.imshow('Image', new_rgb_image)

        # Reset the selected position
        selected_position = None

# Destroy all windows when done
cv2.destroyAllWindows()
