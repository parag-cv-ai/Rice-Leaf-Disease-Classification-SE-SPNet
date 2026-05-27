import cv2
import numpy as np
IMG_SIZE = 128
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image

def normalize_image(image):
    image = image.astype("float32") / 255.0
    return image