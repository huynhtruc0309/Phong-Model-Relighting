import cv2
import numpy as np

# Initialize global variables
positions = []

# Define the callback function for mouse events
def select_position(event, x, y, flags, param):
    global positions
    if event == cv2.EVENT_LBUTTONDOWN:
        positions.append((x, y))
        if len(positions) == 2:
            cv2.circle(image, positions[0], 5, (0, 255, 0), -1)
            cv2.circle(image, positions[1], 5, (0, 255, 0), -1)
            cv2.line(image, positions[0], positions[1], (255, 0, 0), 2)
            cv2.imshow('Image', image)

            # Calculate distance between the two points
            distance = np.sqrt((positions[0][0] - positions[1][0])**2 + (positions[0][1] - positions[1][1])**2)
            print(f"Distance between points: {distance:.2f} pixels")

            # Reset positions list for next selection
            positions = []

# Load an image
image = cv2.imread('sample_5/inputs/rgb_image.png')
cv2.imshow('Image', image)

# Set the mouse callback function to the window
cv2.setMouseCallback('Image', select_position)

# Main loop
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
