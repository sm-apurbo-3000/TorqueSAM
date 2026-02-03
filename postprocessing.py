import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def process_and_resize_image(image_pil):
    image_np = np.array(image_pil)

    non_zero_coords = np.nonzero(image_np)
    if non_zero_coords[0].size == 0 or non_zero_coords[1].size == 0:
        return image_pil

    y_min, x_min = np.min(non_zero_coords[:2], axis=1)
    y_max, x_max = np.max(non_zero_coords[:2], axis=1)

    y_min, x_min = int(y_min), int(x_min)
    y_max, x_max = int(y_max), int(x_max)

    cropped_image = image_np[y_min:y_max, x_min:x_max]
    resized_image = cv2.resize(cropped_image, (224, 224))
    resized_image_pil = Image.fromarray(resized_image)

    return resized_image_pil

def mask_image(img_path, coords_list):
    img = Image.open(img_path)
    data = np.array(img)
    masked_data = np.zeros_like(data)

    for coords in coords_list:
        x1, y1, x2, y2 = coords
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        masked_data[y1:y2, x1:x2] = data[y1:y2, x1:x2]

    masked_img = Image.fromarray(masked_data).convert('L')
    return masked_img

def display_image(image, title='Image'):
    plt.figure(figsize=(10, 10))
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def mask_image2(img_path, coords_list, plot=True):
    img = Image.open(img_path)
    data = np.array(img)
    masked_data = np.zeros_like(data)

    for coords in coords_list:
        x1, y1, x2, y2 = coords
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        masked_data[y1:y2, x1:x2] = data[y1:y2, x1:x2]

    masked_img = Image.fromarray(masked_data).convert('L')

    if plot:
        display_image(np.array(masked_img), title='Masked Image')

    return masked_img
