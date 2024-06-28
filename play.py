import cv2

image = cv2.imread('sample_5/inputs/rgb_image.png')
mask = cv2.imread('sample_5/inputs/mask.png')

# resize the images keep the aspect ratio
scale_percent = 70
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)

cv2.circle(image, (86, 260), 1, (0, 0, 255), -1)
cv2.circle(mask, (86, 260), 1, (0, 0, 255), -1)
cv2.imshow('image', image)
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()