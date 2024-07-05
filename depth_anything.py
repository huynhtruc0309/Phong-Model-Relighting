from transformers import pipeline
from PIL import Image

image = Image.open('sample_' + str(5) + '/inputs/rgb_image.png')
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf")
depth = pipe(image)["depth"]
# save pil image
depth.save('depth_map.png')