"""
MNIST Image Classification Using a Simple Artificial Neural Network
====================================================================
A basic fully connected neural network (ANN) to classify handwritten
digit images (0-9) from the MNIST dataset.

Covers:
    - Data loading & preprocessing
    - Model construction with Dense layers
    - Training & validation
    - Evaluation on the test set
    - Visualization of training history and sample predictions
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ──────────────────────────────────────────────
# 1. Load the MNIST Dataset
# ──────────────────────────────────────────────
print("=" * 60)
print("  MNIST Image Classification – Simple ANN")
print("=" * 60)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"\nDataset loaded successfully!")
print(f"  Training samples : {x_train.shape[0]}  | Image shape: {x_train.shape[1:]}")
print(f"  Test samples     : {x_test.shape[0]}   | Image shape: {x_test.shape[1:]}")
print(f"  Number of classes: {len(np.unique(y_train))}")

# ──────────────────────────────────────────────
# 2. Data Preprocessing
# ──────────────────────────────────────────────
# 2a. Flatten 28x28 images → 784-element vectors
x_train_flat = x_train.reshape(x_train.shape[0], -1).astype("float32")
x_test_flat  = x_test.reshape(x_test.shape[0], -1).astype("float32")

# 2b. Normalize pixel values from [0, 255] → [0, 1]
x_train_flat /= 255.0
x_test_flat  /= 255.0

# 2c. One-hot encode labels  (e.g. 5 → [0,0,0,0,0,1,0,0,0,0])
num_classes  = 10
y_train_ohe  = keras.utils.to_categorical(y_train, num_classes)
y_test_ohe   = keras.utils.to_categorical(y_test, num_classes)

print(f"\nAfter preprocessing:")
print(f"  x_train shape : {x_train_flat.shape}")
print(f"  x_test  shape : {x_test_flat.shape}")
print(f"  y_train shape : {y_train_ohe.shape}  (one-hot)")

# ──────────────────────────────────────────────
# 3. Build the Model
# ──────────────────────────────────────────────
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,), name="hidden_1"),
    layers.Dense(64,  activation="relu",                     name="hidden_2"),
    layers.Dense(num_classes, activation="softmax",          name="output"),
])

model.summary()

# ──────────────────────────────────────────────
# 4. Compile the Model
# ──────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ──────────────────────────────────────────────
# 5. Train the Model
# ──────────────────────────────────────────────
print("\n🚀 Starting training …\n")

EPOCHS     = 10
BATCH_SIZE = 32

history = model.fit(
    x_train_flat, y_train_ohe,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1,
)

# ──────────────────────────────────────────────
# 6. Evaluate on the Test Set
# ──────────────────────────────────────────────
test_loss, test_accuracy = model.evaluate(x_test_flat, y_test_ohe, verbose=0)

print("\n" + "=" * 60)
print(f"  Test Loss     : {test_loss:.4f}")
print(f"  Test Accuracy : {test_accuracy * 100:.2f}%")
print("=" * 60)

# ──────────────────────────────────────────────
# 7. Visualize Training History
# ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy curves
ax1.plot(history.history["accuracy"],     label="Train Accuracy", linewidth=2)
ax1.plot(history.history["val_accuracy"], label="Val Accuracy",   linewidth=2)
ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Loss curves
ax2.plot(history.history["loss"],     label="Train Loss", linewidth=2)
ax2.plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
print("\n📊 Training history plot saved → training_history.png")

# ──────────────────────────────────────────────
# 8. Sample Predictions
# ──────────────────────────────────────────────
predictions = model.predict(x_test_flat[:10], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)
actual_labels    = y_test[:10]

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i], cmap="gray")
    color = "green" if predicted_labels[i] == actual_labels[i] else "red"
    ax.set_title(
        f"Pred: {predicted_labels[i]}  |  Actual: {actual_labels[i]}",
        fontsize=10, color=color, fontweight="bold",
    )
    ax.axis("off")

plt.suptitle("Sample Predictions (green = correct, red = wrong)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150, bbox_inches="tight")
print("🖼️  Sample predictions plot saved → sample_predictions.png")

print("\n✅ Done! All steps completed successfully.\n")
