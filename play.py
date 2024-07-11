import cv2
import numpy as np

def add_multiple_colors(*colors):
    """
    Add multiple RGB colors.
    
    Parameters:
    colors (tuples): A list of colors as (R, G, B) tuples.
    
    Returns:
    tuple: The resulting color as an (R, G, B) tuple.
    """
    r, g, b = 0, 0, 0
    for color in colors:
        r = (r + color[0]) // 2
        g = (g + color[1]) // 2
        b = (b + color[2]) // 2
    return (r, g, b)

# Example usage
colors = [(245,144,0), (248,36,0), (248,248,248)]
resulting_color = add_multiple_colors(*colors)
print("Resulting color:", resulting_color)

# Create an image with the resulting color
height, width = 100, 100  # Size of the image
image = np.zeros((height, width, 3), dtype=np.uint8)
image[:] = resulting_color[::-1]  # OpenCV uses BGR format, so we reverse the tuple

# Display the image
cv2.imshow("Resulting Color", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
