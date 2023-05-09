import numpy as np
from PIL import Image

def find_bbox(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

def zoom_centre(img):
    bbox = find_bbox(img)
    img = img[bbox[0]:bbox[1],bbox[2]:bbox[3]]
    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img = pil_img.resize((28, 28), Image.LANCZOS)
    img = np.array(pil_img)
    return img