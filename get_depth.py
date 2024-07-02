import cv2
import numpy as np

def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_COLOR)

def load_depth_map(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def load_normal_map(file_path):
    return cv2.imread(file_path, cv2.IMREAD_COLOR)

def load_mask(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# Paths to your images
sample = 6
rgb_image_path = 'sample_' + str(sample) + '/inputs/rgb_image.png'
depth_map_path = 'sample_' + str(sample) + '/inputs/depth_map.png'
normal_map_path = 'sample_' + str(sample) + '/inputs/normal_map.png'
mask_path = 'sample_' + str(sample) + '/inputs/mask.png'

# Load images
rgb_image = load_image(rgb_image_path)
depth_map = load_depth_map(depth_map_path)
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

# Check if images are loaded correctly
if rgb_image is None or depth_map is None or normal_map is None:
    print("One or more images could not be loaded. Exiting program.")
    exit()

# Initialize a global variable to store the selected position
selected_position = None

# Define a mouse callback function to capture the selected position
def mouse_callback(event, x, y, flags, param): 
    global selected_position
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_position = (x, y)

# Create a window and set the mouse callback function
cv2.namedWindow('Get depth and normal maps')
cv2.setMouseCallback('Get depth and normal maps', mouse_callback)
cv2.imshow('Get depth and normal maps', rgb_image)

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


        print(x, y, depth_map[y, x], normal_map[y, x])

        # Reset the selected position
        selected_position = None

# Destroy all windows when done
cv2.destroyAllWindows()
