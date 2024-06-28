import cv2
import numpy as np
from numba import njit, objmode

def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_COLOR)

def load_depth_map(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# @njit
def draw_straight_line(p0, p1, depth_map):
    x0, y0, z0 = p0
    x1, y1, z1 = p1
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

# @njit
def check_point_on_line_segment(p, p0, p1, tolerance=1e-9):
    # Calculate vectors
    line_vec = p1 - p0
    point_vec = p - p0
    
    # Calculate dot product
    dot_product = np.dot(line_vec, point_vec)
    
    # Calculate squared lengths
    line_len_sq = np.dot(line_vec, line_vec)
    point_len_sq = np.dot(point_vec, point_vec)
    
    # Check if the point lies on the infinite line
    if dot_product**2 != point_len_sq * line_len_sq:
        return False    
    return True


# Paths to your images
sample = 5
rgb_image_path = 'sample_' + str(sample) + '/inputs/rgb_image.png'
depth_map_path = 'sample_' + str(sample) + '/inputs/depth_map.png'
mask_path = 'sample_' + str(sample) + '/inputs/mask.png'

# Load images
rgb_image = load_image(rgb_image_path)
depth_map = load_depth_map(depth_map_path)
mask = load_depth_map(mask_path)

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
        new_rgb_image = rgb_image.copy()

        # Update the image (you can add your processing logic here)
        light_position = np.array([10, y, depth_map[y, x]])
        position = np.array([x, y, depth_map[y, x]])
        stop_line_position = None

        check_positions = draw_straight_line(light_position, position, depth_map)
        for check_position in check_positions:
            # Check if the light can reach the pixel
            check_position = np.array([check_position[0], check_position[1], depth_map[check_position[1], check_position[0]]])

            cv2.line(new_rgb_image, (int(check_position[0]), int(check_position[1])), (int(check_position[0]), int(check_position[1] + depth_map[check_position[1], check_position[0]])), (0, 0, 0), 1)

            if check_point_on_line_segment(check_position, light_position, position) and stop_line_position is None and mask[check_position[1], check_position[0]] > 250:
                stop_line_position = check_position
                # break

        print("Stop line position:", stop_line_position, "Position:", position, "Mask value:", mask[check_position[1], check_position[0]])

        cv2.circle(new_rgb_image, (light_position[0], light_position[1]), 5, (255, 0, 0), -1)
        cv2.circle(new_rgb_image, (int(position[0]), int(position[1])), 1, (0, 0, 255), -1)
        # cv2.line(new_rgb_image, (int(position[0]), int(position[1])), (int(light_position[0]), int(light_position[1])), (0, 0, 0), 1)

        if stop_line_position is not None:
            cv2.circle(new_rgb_image, (int(stop_line_position[0]), int(stop_line_position[1])), 1, (0, 255, 0), -1)

        cv2.imshow('Image', new_rgb_image)

        # Reset the selected position
        selected_position = None

# Destroy all windows when done
cv2.destroyAllWindows()
