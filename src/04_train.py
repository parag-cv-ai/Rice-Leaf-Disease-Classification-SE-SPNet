from dataset import load_dataset, split_dataset
from model import build_model
import tensorflow as tf

DATASET_PATH = "dataset"
X, y = load_dataset(DATASET_PATH)
X_train, X_test, y_train, y_test = split_dataset(X, y)
model = build_model()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)
model.save("../models/trained_model.h5")
print("\nModel training completed successfully.")