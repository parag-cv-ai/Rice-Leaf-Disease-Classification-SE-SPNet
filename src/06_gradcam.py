import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from dataset import load_dataset, split_dataset

CLASS_NAMES = [
    "Bacterial Blight",
    "Blast",
    "Brown Spot"
]

IMG_SIZE = 128

X, y = load_dataset("dataset")
X_train, X_test, y_train, y_test = split_dataset(X, y)

model = load_model("../models/trained_model.h5")
last_conv_layer_name = "final_feature_conv"

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer(last_conv_layer_name).output,
        model.output
    ]
)
index = 0
img = X_test[index]
img_array = np.expand_dims(img, axis=0)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    pred_index = tf.argmax(predictions[0])
    class_channel = predictions[:, pred_index]
grads = tape.gradient(class_channel, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
conv_outputs = conv_outputs[0]
heatmap = np.mean(
    conv_outputs * pooled_grads,
    axis=-1
)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
heatmap = cv2.resize(
    heatmap.numpy(),
    (IMG_SIZE, IMG_SIZE)
)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)
original_img = np.uint8(img * 255)

overlay = cv2.addWeighted(
    original_img,
    0.5,
    heatmap,
    0.5,
    0
)
plt.figure(figsize=(10,5))
plt.imshow(overlay)
plt.axis("off")
plt.title("Grad-CAM Visualization")
plt.savefig(
    "../results/gradcam_visualization.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()