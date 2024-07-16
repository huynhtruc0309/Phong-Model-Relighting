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
    return np.load(file_path)

def load_mask(file_path):
    if not os.path.isfile(file_path):
        print(f"Mask '{file_path}' does not exist.")
        return None
    binary_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(binary_mask, 128, 255, cv2.THRESH_BINARY)
    return binary_mask

def refine_visibility_map(visibility_map, mask, kernel_size=5):
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

    for y in range(height):
        for x in range(width):
            position = np.array([x, y, depth_map[y, x]])
            check_positions = draw_straight_line(light_position, position)
            
            if len(check_positions):
                depth_step = (light_position[2] - position[2]) / len(check_positions)
                check_position_depth = light_position[2]

                for check_position in check_positions:
                    check_position_depth -= depth_step

                    if abs(depth_map[int(check_position[1]), int(check_position[0])] - check_position_depth) < tolerance \
                        and mask[int(check_position[1]), int(check_position[0])] \
                        and abs(depth_map[int(check_position[1]), int(check_position[0])] - position[2]) > 1e-9: 
                            visibility_map[y, x] = 0
                            break
            
    print("Visibility map calculated!")
    return visibility_map

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
    points = []  

    dx = abs(x1 - x0) 
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        e2 = 2 * err
        if e2 > -dy: 
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
        if x0 == x1 and y0 == y1: 
            break
        points.append((x0, y0))
        
    return points

@njit
def sRGB_to_linear(rgb):
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear

@njit
def relight_image(rgb_image, depth_map, normal_map, visibility_map, light_position, light_color):
    print("Relighting the image...")
    height, width, _ = rgb_image.shape
    new_rgb_image = np.zeros_like(rgb_image)
    
    ambient_coefficient = 0
    diffuse_coefficient = 0.9
    specular_coefficient = 0
    shininess = 32

    for y in range(height):
        for x in range(width):
            position = np.array([x, y, depth_map[y, x]])
            light_vector = position - light_position
            light_vector = normalize(light_vector)
            depth_factor = (1 - abs(light_vector[2]) / 1000.0) ** 2 if visibility_map[y, x] else 0

            normal_vector = normal_map[y, x]
            normal_vector = normalize(normal_vector)
            diffuse_intensity = max(np.dot(normal_vector, light_vector), 0) * depth_factor

            view_vector = np.array([0., 0., -1.])
            reflect_vector = 2 * normal_vector * np.dot(normal_vector, light_vector) - light_vector
            reflect_vector = normalize(reflect_vector)
            specular_intensity = max(np.dot(view_vector, reflect_vector), 0) ** shininess  * depth_factor

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

            original_color_linear = rgb_image_linear[y, x]
            new_color_linear = (original_color_linear + color) / 2

            new_color_sRGB = np.clip(new_color_linear, 0, 1) ** (1 / 2.2) 
            new_rgb_image[y, x] = np.clip(new_color_sRGB * 255, 0, 255)

    print("Image relighted!")
    return new_rgb_image.astype(np.uint8)

sample = 5
rgb_image_path = 'sample_' + str(sample) + '/inputs/rgb_image.png'
depth_map_path = 'sample_' + str(sample) + '/inputs/depth_map.png'
normal_map_path = 'sample_' + str(sample) + '/inputs/normal.npy'
mask_path = 'sample_' + str(sample) + '/inputs/mask.png'

rgb_image = load_image(rgb_image_path)
depth_map = load_depth_map(depth_map_path)
normal_map = load_normal_map(normal_map_path)
mask = load_mask(mask_path)

scale_percent = 100
width = int(rgb_image.shape[1] * scale_percent / 100)
height = int(rgb_image.shape[0] * scale_percent / 100)
dim = (width, height)
rgb_image = cv2.resize(rgb_image, dim, interpolation = cv2.INTER_AREA)
depth_map = cv2.resize(depth_map, dim, interpolation = cv2.INTER_AREA)
normal_map = np.resize(normal_map, (height, width, 3))
mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)

white_light = [255, 255, 255]  # The oder of the color is BGR

selected_position = None

def mouse_callback(event, x, y): 
    global selected_position
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_position = (x, y)

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)
cv2.imshow('Image', rgb_image)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if selected_position:
        x, y = selected_position

        light_position = np.array([x, y, 300]).astype(np.float64)
        light_position[2] = 1000 if light_position[2] > 1000 else light_position[2]
        depth_map = depth_map.astype(np.float64) 

        visibility_map = calculate_visibility_map(depth_map, mask, light_position)
        visibility_map = refine_visibility_map(visibility_map, mask)

        rgb_image_linear = sRGB_to_linear(rgb_image / 255.0)
        light_color_linear = sRGB_to_linear(np.array(white_light) / 255.0)
        new_rgb_image = relight_image(rgb_image_linear, depth_map, normal_map, visibility_map, light_position, light_color_linear)

        cv2.circle(new_rgb_image, (int(light_position[0]), int(light_position[1])), 5, (255, 0, 0), -1)

        cv2.imshow('Image', new_rgb_image)
        cv2.imshow('Shadow map', visibility_map * 255)

        selected_position = None

cv2.destroyAllWindows()
