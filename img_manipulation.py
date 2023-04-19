import numpy as np
from PIL import Image

def shift_image(img: np.array, shift_x: int = 0 , shift_y: int = 0):
    new_img = shift_along_x(img, shift_x)
    new_img = shift_along_y(new_img, shift_y)
    return new_img

def shift_along_x(img: np.array, shift: int = 0):
    if shift == 0:
        return img
    
    new_img = np.zeros_like(img)

    if shift > 0:
        new_img[:, shift:] = np.roll(img, shift, 1)[:, shift:]
        return new_img
    else:
        new_img[:, :shift] = np.roll(img, shift, 1)[:, :shift]
        return new_img

def shift_along_y(img: np.array, shift: int = 0):
    if shift == 0:
        return img
    
    new_img = np.zeros_like(img)

    if shift > 0:
        new_img[shift:, :] = np.roll(img, shift, 0)[shift:, :]
        return new_img
    else:
        new_img[:shift, :] = np.roll(img, shift, 0)[:shift, :]
        return new_img

def rotate_image(img: np.array, rotation: int = 0):
    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img = pil_img.rotate(rotation)
    return np.array(pil_img)

def zoom_image(img: np.array, zoom: float = 1):
    w, h = img.shape
    x, y = w//2, h//2 # centre of img
    pil_img = Image.fromarray(img.astype(np.uint8))

    pil_img = pil_img.crop((x - w / (zoom * 2), 
                            y - h / (zoom * 2), 
                            x + w / (zoom * 2), 
                            y + h / (zoom * 2)))

    pil_img = pil_img.resize((w, h), Image.LANCZOS)
    return np.array(pil_img)