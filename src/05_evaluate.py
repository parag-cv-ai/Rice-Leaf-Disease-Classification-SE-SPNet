import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from tensorflow.keras.models import load_model
from dataset import load_dataset, split_dataset
CLASS_NAMES = [
    "Bacterial Blight",
    "Blast",
    "Brown Spot"
]
X, y = load_dataset("dataset")
X_train, X_test, y_train, y_test = split_dataset(X, y)

model = load_model("../models/trained_model.h5")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(
    classification_report(
        y_true,
        y_pred_classes,
        target_names=CLASS_NAMES
    )
)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(
    "../results/confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)
plt.show()