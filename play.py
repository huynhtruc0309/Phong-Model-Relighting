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

def refine_visibility_map(visibility_map, mask, kernel_size=7):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    smoothed_background = cv2.morphologyEx(visibility_map, cv2.MORPH_OPEN, kernel)
    smoothed_foreground = cv2.morphologyEx(visibility_map, cv2.MORPH_CLOSE, kernel)

    blended_visibility_map = np.where(mask, smoothed_foreground, smoothed_background)
    final_smoothed_map = cv2.GaussianBlur(blended_visibility_map, (kernel_size, kernel_size), 0)
    return final_smoothed_map

@njit
def calculate_visibility_map(depth_map, mask, light_start, light_direction, spread_angle_deg, tolerance=1):
    height, width = depth_map.shape
    visibility_map = np.ones_like(depth_map)

    light_direction = normalize(light_direction)
    spread_angle_rad = np.deg2rad(spread_angle_deg / 2)

    for y in range(height):
        for x in range(width):
            position = np.array([x, y, depth_map[y, x]])
            light_vector = light_start - position
            light_vector = normalize(light_vector)
            angle = np.arccos(np.dot(light_direction, light_vector))

            if angle <= spread_angle_rad:
                check_positions = draw_straight_line(light_start, position)

                if len(check_positions):
                    depth_step = (light_start[2] - position[2]) / len(check_positions)
                    check_position_depth = light_start[2]

                    for check_position in check_positions:
                        check_position_depth -= depth_step

                        if abs(depth_map[int(check_position[1]), int(check_position[0])] - check_position_depth) < tolerance \
                            and mask[int(check_position[1]), int(check_position[0])] \
                            and abs(depth_map[int(check_position[1]), int(check_position[0])] - position[2]) > 1e-9: 
                                visibility_map[y, x] = 0
                                break
            
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

    while (x0, y0) != (x1, y1):
        points.append((x0, y0))
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
        
    return points

@njit
def sRGB_to_linear(rgb):
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear

@njit
def relight_image(rgb_image, depth_map, normal_map, light_start, light_color, visibility_map):
    height, width, _ = rgb_image.shape
    new_rgb_image = np.zeros_like(rgb_image)
    
    ambient_coefficient = 0
    diffuse_coefficient = 0.7
    specular_coefficient = 0
    shininess = 32

    for y in range(height):
        for x in range(width):
            position = np.array([x, y, depth_map[y, x]])
            light_vector = light_start - position
            depth_factor = (1 - abs(light_vector[2]) / 1000.0) ** 2 if visibility_map[y, x] else 0
            normal_vector = normalize(normal_map[y, x])
            light_vector = normalize(light_vector)
            diffuse_intensity = max(np.dot(normal_vector, light_vector), 0) * depth_factor

            view_vector = np.array([0., 0., 1.])
            reflect_vector = 2 * normal_vector * np.dot(normal_vector, light_vector) - light_vector
            reflect_vector = normalize(reflect_vector)
            specular_intensity = max(np.dot(view_vector, reflect_vector), 0) ** shininess  * depth_factor

            ambient = ambient_coefficient * light_color 
            diffuse = diffuse_coefficient * diffuse_intensity * light_color
            specular = specular_coefficient * specular_intensity * light_color

            color = ambient + diffuse + specular

            original_color_linear = rgb_image[y, x]
            new_color_linear = original_color_linear + color
            new_color_sRGB = np.clip(new_color_linear, 0, 1) ** (1 / 2.2)
            new_rgb_image[y, x] = np.clip(new_color_sRGB * 255, 0, 255)

    return new_rgb_image.astype(np.uint8)

sample = 5
rgb_image_path = f'sample_{sample}/inputs/rgb_image.png'
depth_map_path = f'sample_{sample}/inputs/depth_map_1.png'
normal_map_path = f'sample_{sample}/inputs/normal_map.png'
mask_path = f'sample_{sample}/inputs/mask.png'

rgb_image = load_image(rgb_image_path)
depth_map = load_depth_map(depth_map_path)
normal_map = load_normal_map(normal_map_path)
mask = load_mask(mask_path)

scale_percent = 100
width = int(rgb_image.shape[1] * scale_percent / 100)
height = int(rgb_image.shape[0] * scale_percent / 100)
dim = (width, height)
rgb_image = cv2.resize(rgb_image, dim, interpolation=cv2.INTER_AREA)
depth_map = cv2.resize(depth_map, dim, interpolation=cv2.INTER_AREA)
normal_map = cv2.resize(normal_map, dim, interpolation=cv2.INTER_AREA)
mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)

white_light = [255, 255, 255]
spread_angle_deg = 30
light_z = 300

selected_positions = []

def mouse_callback(event, x, y, flags, param):
    global selected_positions
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_positions.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        selected_positions = []

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)
cv2.imshow('Image', rgb_image)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if len(selected_positions) == 2:
        light_start = np.array([selected_positions[0][0], selected_positions[0][1], light_z]).astype(np.float64)
        light_start[2] = 1000. if light_start[2] > 1000 else light_start[2]
        light_end = np.array([selected_positions[1][0], selected_positions[1][1], depth_map[selected_positions[1][1], selected_positions[1][0]]]).astype(np.float64)
        light_direction = light_start - light_end
        
        rgb_image_linear = sRGB_to_linear(rgb_image / 255.0)
        light_color_linear = sRGB_to_linear(np.array(white_light) / 255.0)

        depth_map = depth_map.astype(np.float64) 

        visibility_map = calculate_visibility_map(depth_map, mask, light_start, light_direction, spread_angle_deg)
        visibility_map = refine_visibility_map(visibility_map, mask)

        new_rgb_image = relight_image(rgb_image_linear, depth_map, normal_map, light_start, light_color_linear, visibility_map)
        cv2.line(new_rgb_image, selected_positions[0], selected_positions[1], (255, 0, 0), 2)

        cv2.imshow('Image', new_rgb_image)
        cv2.imshow('Shadow map', visibility_map * 255)

        selected_positions = []

cv2.destroyAllWindows()
