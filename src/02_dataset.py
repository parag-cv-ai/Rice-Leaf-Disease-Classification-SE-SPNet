import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_image

CLASS_NAMES = [
    "Bacterial Blight",
    "Blast",
    "Brown Spot"
]

def load_dataset(dataset_path):
    X = []
    y = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(dataset_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            try:
                image = preprocess_image(image_path)
                X.append(image)
                y.append(label)
            except:
                continue
    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y, num_classes=len(CLASS_NAMES))
    return X, y

def split_dataset(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )